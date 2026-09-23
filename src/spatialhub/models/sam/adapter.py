from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from spatialhub.core.runtime import create_ort_session, resolve_model_path
from spatialhub.structures import SegmentationResult
from spatialhub.utils import load_image, non_max_suppression, normalize_image

logger = logging.getLogger(__name__)

SAM_PIXEL_MEAN = np.array([123.675, 116.28, 103.53], dtype=np.float32).reshape(1, 1, 3)
SAM_PIXEL_STD = np.array([58.395, 57.12, 57.375], dtype=np.float32).reshape(1, 1, 3)


def masks_to_boxes(masks: np.ndarray) -> np.ndarray:
    """Compute bounding boxes [x1, y1, x2, y2] from boolean masks of shape (N, H, W)."""
    n = masks.shape[0]
    boxes = np.zeros((n, 4), dtype=np.float32)
    for i in range(n):
        m = masks[i]
        rows = np.any(m, axis=1)
        cols = np.any(m, axis=0)
        if not np.any(rows):
            continue
        ymin, ymax = np.where(rows)[0][[0, -1]]
        xmin, xmax = np.where(cols)[0][[0, -1]]
        boxes[i] = [xmin, ymin, xmax + 1, ymax + 1]
    return boxes


MODEL_REGISTRY: dict[str, dict[str, Any]] = {
    "vit_b": {
        "encoder_filename": "vit_b_encoder.onnx",
        "decoder_filename": "vit_b_decoder.onnx",
        "repo_id": "SpatialHub/sam-onnx",
    },
    "vit_l": {
        "encoder_filename": "vit_l_encoder.onnx",
        "decoder_filename": "vit_l_decoder.onnx",
        "repo_id": "SpatialHub/sam-onnx",
    },
    "vit_h": {
        "encoder_filename": "vit_h_encoder.onnx",
        "decoder_filename": "vit_h_decoder.onnx",
        "repo_id": "SpatialHub/sam-onnx",
    },
}


class SAMAdapter:
    """Automatic Mask Generation (AMG) pipeline using Segment Anything Model (SAM).

    Executes image encoder and mask decoder models independently, grid-sampling
    prompt coordinates across input images to generate segmentation proposals.
    """

    def __init__(
        self,
        encoder_onnx_path: str | Path | None = None,
        decoder_onnx_path: str | Path | None = None,
        model_variant: str = "vit_h",
        target_size: int = 1024,
        points_per_side: int = 32,
        points_per_batch: int = 64,
        pred_iou_thresh: float = 0.88,
        stability_score_thresh: float = 0.95,
        box_nms_thresh: float = 0.7,
        providers: list[str] | str | None = None,
    ) -> None:
        """Initialize SAM segmentor.

        Args:
            encoder_onnx_path:
                Optional path to image encoder model. Resolved automatically if None.
            decoder_onnx_path:
                Optional path to mask decoder model. Resolved automatically if None.
            model_variant:
                SAM backbone variant identifier ('vit_h', 'vit_l', 'vit_b', default: 'vit_h').
            target_size:
                Target spatial resolution for image encoder input (default: 1024).
            points_per_side:
                Grid sampling density along each axis for AMG (default: 32).
            points_per_batch:
                Batch size chunking for decoder inference passes (default: 64).
            pred_iou_thresh:
                Minimum predicted IoU threshold for valid mask candidates.
            stability_score_thresh:
                Minimum stability score threshold across binarization levels.
            box_nms_thresh:
                IoU cutoff threshold for duplicate mask removal via NMS.
            providers:
                Execution providers.

        Raises:
            ValueError:
                If either model path is None and model_variant is not recognized.
        """
        variant = str(model_variant).lower()
        if (encoder_onnx_path is None or decoder_onnx_path is None) and variant not in MODEL_REGISTRY:
            raise ValueError(
                f"Unsupported model_variant '{model_variant}'. Supported variants: {sorted(MODEL_REGISTRY.keys())}"
            )

        reg_entry = MODEL_REGISTRY.get(variant, MODEL_REGISTRY["vit_h"])
        repo_id = reg_entry["repo_id"]

        resolved_enc = resolve_model_path(
            model_path=encoder_onnx_path,
            repo_id=repo_id,
            filename=reg_entry["encoder_filename"] if encoder_onnx_path is None else None,
        )
        resolved_dec = resolve_model_path(
            model_path=decoder_onnx_path,
            repo_id=repo_id,
            filename=reg_entry["decoder_filename"] if decoder_onnx_path is None else None,
        )

        self.encoder_session = create_ort_session(
            model_path=resolved_enc,
            providers=providers,
        )
        self.decoder_session = create_ort_session(
            model_path=resolved_dec,
            providers=providers,
        )

        self.target_size = target_size
        self.points_per_side = points_per_side
        self.points_per_batch = points_per_batch
        self.default_pred_iou_thresh = pred_iou_thresh
        self.default_stability_score_thresh = stability_score_thresh
        self.default_box_nms_thresh = box_nms_thresh

        self.pixel_mean = SAM_PIXEL_MEAN
        self.pixel_std = SAM_PIXEL_STD

        # Precompute uniform point grid in relative [0, 1] coordinates
        self.points_rel = self._generate_point_grid(self.points_per_side)

        # Preallocate constant decoder input buffers for full batches
        self._cached_mask_input = np.zeros((self.points_per_batch, 1, 256, 256), dtype=np.float32)
        self._cached_has_mask_input = np.zeros((self.points_per_batch,), dtype=np.float32)
        self._cached_point_labels = np.ones((self.points_per_batch, 1), dtype=np.float32)

    def _preprocess_image(self, image: np.ndarray) -> tuple[np.ndarray, float, tuple[int, int]]:
        """Preprocess RGB image to padded network input tensor.

        Args:
            image: Input RGB image array of shape (H, W, 3).

        Returns:
            tuple containing:
                - input_tensor: Normalized and padded array of shape (1, 3, target_size, target_size).
                - scale: Resize scaling factor applied to longest side.
                - orig_hw: Original spatial dimensions as (orig_h, orig_w).
        """
        orig_h, orig_w = image.shape[:2]

        scale = float(self.target_size) / max(orig_h, orig_w)
        new_h, new_w = int(orig_h * scale + 0.5), int(orig_w * scale + 0.5)
        sam_img = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        sam_img = normalize_image(sam_img, mean=self.pixel_mean, std=self.pixel_std, to_chw=False)

        pad_h, pad_w = self.target_size - new_h, self.target_size - new_w
        padded_img = np.pad(sam_img, ((0, pad_h), (0, pad_w), (0, 0)), mode="constant")

        input_tensor = padded_img.transpose((2, 0, 1))[None, ...].astype(np.float32)
        return input_tensor, scale, (orig_h, orig_w)

    def _encode_image(self, input_tensor: np.ndarray) -> np.ndarray:
        """Extract image feature embeddings via image encoder session.

        Args:
            input_tensor: Normalized and padded input tensor of shape (1, 3, target_size, target_size).

        Returns:
            np.ndarray: Image embeddings array of shape (1, 256, 64, 64).
        """
        return self.encoder_session.run(None, {"image": input_tensor})[0]

    def _generate_point_grid(self, points_per_side: int | None = None) -> np.ndarray:
        """Generate uniform point grid in relative [0, 1] coordinates.

        Args:
            points_per_side: Grid density along each axis (defaults to self.points_per_side).

        Returns:
            np.ndarray: Array of shape (points_per_side^2, 2) in normalized [0, 1] range.
        """
        n_points = points_per_side if points_per_side is not None else self.points_per_side
        offset = 1.0 / (2 * n_points)
        points_one_side = np.linspace(offset, 1.0 - offset, n_points)

        pts_x = np.tile(points_one_side[None, :], (n_points, 1)).flatten()
        pts_y = np.tile(points_one_side[:, None], (1, n_points)).flatten()

        return np.stack([pts_x, pts_y], axis=1)

    def _run_decoder_batches(
        self,
        image_embedding: np.ndarray,
        points_resized: np.ndarray,
        orig_h: int,
        orig_w: int,
    ) -> tuple[list[np.ndarray], list[np.ndarray]]:
        """Run batched point prompt inference through mask decoder session.

        Args:
            image_embedding: Image features of shape (1, 256, 64, 64).
            points_resized: Point coordinates scaled to encoder input space of shape (N, 2).
            orig_h: Original image height.
            orig_w: Original image width.

        Returns:
            tuple containing:
                - all_raw_masks: List of mask logits arrays of shape (B, H, W).
                - all_iou_preds: List of predicted IoU score arrays of shape (B,).
        """
        all_raw_masks: list[np.ndarray] = []
        all_iou_preds: list[np.ndarray] = []
        orig_im_size = np.array([orig_h, orig_w], dtype=np.float32)

        for i in range(0, len(points_resized), self.points_per_batch):
            batch_pts = points_resized[i : i + self.points_per_batch]
            batch_size = len(batch_pts)

            mask_input = self._cached_mask_input if batch_size == self.points_per_batch else self._cached_mask_input[:batch_size]
            has_mask_input = self._cached_has_mask_input if batch_size == self.points_per_batch else self._cached_has_mask_input[:batch_size]
            point_labels = self._cached_point_labels if batch_size == self.points_per_batch else self._cached_point_labels[:batch_size]

            ort_inputs = {
                "image_embeddings": image_embedding,
                "point_coords": batch_pts[:, None, :].astype(np.float32),
                "point_labels": point_labels,
                "mask_input": mask_input,
                "has_mask_input": has_mask_input,
                "orig_im_size": orig_im_size,
            }

            masks, iou_preds, _ = self.decoder_session.run(None, ort_inputs)
            all_raw_masks.append(masks[:, 0, :, :])
            all_iou_preds.append(iou_preds[:, 0])

        return all_raw_masks, all_iou_preds

    def _postprocess(
        self,
        image: np.ndarray,
        all_raw_masks: list[np.ndarray],
        all_iou_preds: list[np.ndarray],
        orig_h: int,
        orig_w: int,
        pred_iou_thresh: float,
        stability_score_thresh: float,
        box_nms_thresh: float,
    ) -> SegmentationResult:
        """Filter decoder candidate masks and apply non-maximum suppression.

        Args:
            image: Source RGB image array of shape (H, W, 3).
            all_raw_masks: List of mask logits arrays from decoder passes.
            all_iou_preds: List of predicted IoU score arrays from decoder passes.
            orig_h: Original image height.
            orig_w: Original image width.
            pred_iou_thresh: Minimum predicted IoU score threshold.
            stability_score_thresh: Minimum stability score threshold.
            box_nms_thresh: Box IoU cutoff threshold for NMS deduplication.

        Returns:
            SegmentationResult: Filtered boxes, masks, and confidence scores.
        """
        all_masks, all_scores, all_boxes = [], [], []

        for masks, iou_preds in zip(all_raw_masks, all_iou_preds):
            keep = iou_preds > pred_iou_thresh
            masks, iou_preds = masks[keep], iou_preds[keep]
            if len(masks) == 0:
                continue

            intersections = (masks > 1.0).sum(axis=(-1, -2))
            unions = (masks > -1.0).sum(axis=(-1, -2))
            stability_scores = intersections / (unions + 1e-6)

            keep = stability_scores > stability_score_thresh
            masks, iou_preds = masks[keep], iou_preds[keep]
            if len(masks) == 0:
                continue

            masks_binary = masks > 0.0
            boxes = masks_to_boxes(masks_binary)

            all_masks.append(masks_binary)
            all_scores.append(iou_preds)
            all_boxes.append(boxes)

        if len(all_boxes) == 0:
            return SegmentationResult(
                image=image,
                boxes=np.empty((0, 4), dtype=np.float32),
                masks=np.empty((0, orig_h, orig_w), dtype=bool),
                scores=np.empty((0,), dtype=np.float32),
            )

        final_boxes = np.concatenate(all_boxes, axis=0)
        final_scores = np.concatenate(all_scores, axis=0)
        final_masks = np.concatenate(all_masks, axis=0)

        keep_idx = non_max_suppression(final_boxes, final_scores, box_nms_thresh)

        return SegmentationResult(
            image=image,
            boxes=final_boxes[keep_idx].astype(np.float32),
            masks=final_masks[keep_idx],
            scores=final_scores[keep_idx].astype(np.float32),
        )

    def generate_masks(self, image: str | Path | np.ndarray, **kwargs) -> SegmentationResult:
        """Execute Automatic Mask Generation (AMG) over input image.

        Args:
            image: Input image (file path or RGB NumPy array).
            kwargs: Optional threshold overrides ('pred_iou_thresh', 'stability_score_thresh', 'iou_threshold', 'points_per_side').

        Returns:
            SegmentationResult: Dataclass containing bounding boxes, masks, and scores.
        """
        pred_iou_thresh = kwargs.get("pred_iou_thresh", self.default_pred_iou_thresh)
        stability_score_thresh = kwargs.get("stability_score_thresh", self.default_stability_score_thresh)
        box_nms_thresh = kwargs.get("iou_threshold", self.default_box_nms_thresh)

        if isinstance(image, (str, Path)):
            image = load_image(image, color_mode="RGB")

        # Preprocess
        input_tensor, scale, (orig_h, orig_w) = self._preprocess_image(image)

        # Encode
        image_embedding = self._encode_image(input_tensor)

        # Point Grid Coordinate Scaling
        if "points_per_side" in kwargs and kwargs["points_per_side"] != self.points_per_side:
            points_rel = self._generate_point_grid(kwargs["points_per_side"])
        else:
            points_rel = self.points_rel

        points_orig = points_rel * np.array([orig_w, orig_h])
        points_resized = points_orig * scale

        # Batched Mask Decoder Pass
        all_raw_masks, all_iou_preds = self._run_decoder_batches(
            image_embedding=image_embedding,
            points_resized=points_resized,
            orig_h=orig_h,
            orig_w=orig_w,
        )

        # Postprocess
        return self._postprocess(
            image=image,
            all_raw_masks=all_raw_masks,
            all_iou_preds=all_iou_preds,
            orig_h=orig_h,
            orig_w=orig_w,
            pred_iou_thresh=pred_iou_thresh,
            stability_score_thresh=stability_score_thresh,
            box_nms_thresh=box_nms_thresh,
        )

    def __enter__(self) -> SAMAdapter:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    def close(self) -> None:
        """Release underlying ONNX Runtime inference sessions and associated resources."""
        if getattr(self, "encoder_session", None) is not None:
            self.encoder_session = None
        if getattr(self, "decoder_session", None) is not None:
            self.decoder_session = None

