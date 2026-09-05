import logging
from pathlib import Path

import cv2
import numpy as np

from spatialhub.core.runtime import create_ort_session, resolve_model_path
from spatialhub.structures import SegmentationResult
from spatialhub.utils import load_image, non_max_suppression, normalize_image

logger = logging.getLogger(__name__)


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


class SAMAdapter:
    """Automatic Mask Generation (AMG) pipeline using Segment Anything Model (SAM).

    Executes image encoder and mask decoder models independently, grid-sampling
    prompt coordinates across input images to generate fine-grained segmentation
    proposals.
    """

    def __init__(
        self,
        encoder_onnx_path: str | Path | None = None,
        decoder_onnx_path: str | Path | None = None,
        model_variant: str | None = "vit_h",
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
                SAM backbone variant identifier ('sam_vit_h', 'sam_vit_l', 'sam_vit_b', or short names 'vit_h', 'vit_l', 'vit_b').
            points_per_side:
                Grid sampling density along each axis for AMG.
            points_per_batch:
                Batch size chunking for decoder inference passes.
            pred_iou_thresh:
                Minimum predicted IoU threshold for valid mask candidates.
            stability_score_thresh:
                Minimum stability score threshold across binarization levels.
            box_nms_thresh:
                IoU cutoff threshold for duplicate mask removal via NMS.
            providers:
                Execution providers.
        """
        variant = str(model_variant or "sam_vit_h")
        if not variant.startswith("sam_") and not variant.endswith(".onnx"):
            variant = f"sam_{variant}"

        enc_filename = f"{variant}_encoder.onnx" if not (encoder_onnx_path and str(encoder_onnx_path).endswith(".onnx")) else Path(encoder_onnx_path).name
        dec_filename = f"{variant}_decoder.onnx" if not (decoder_onnx_path and str(decoder_onnx_path).endswith(".onnx")) else Path(decoder_onnx_path).name

        resolved_enc = resolve_model_path(
            model_path=encoder_onnx_path,
            repo_id="SpatialHub/sam-onnx",
            filename=enc_filename if encoder_onnx_path is None else None,
            download_sidecar_data=True if "vit_h" in variant else False,
        )
        resolved_dec = resolve_model_path(
            model_path=decoder_onnx_path,
            repo_id="SpatialHub/sam-onnx",
            filename=dec_filename if decoder_onnx_path is None else None,
        )

        self.encoder_session = create_ort_session(
            model_path=resolved_enc,
            providers=providers,
        )
        self.decoder_session = create_ort_session(
            model_path=resolved_dec,
            providers=providers,
        )

        self.points_per_side = points_per_side
        self.points_per_batch = points_per_batch
        self.default_pred_iou_thresh = pred_iou_thresh
        self.default_stability_score_thresh = stability_score_thresh
        self.default_box_nms_thresh = box_nms_thresh

        self.target_size = 1024
        self.pixel_mean = np.array([123.675, 116.28, 103.53], dtype=np.float32).reshape(1, 1, 3)
        self.pixel_std = np.array([58.395, 57.12, 57.375], dtype=np.float32).reshape(1, 1, 3)

    def _encode_image(self, image: np.ndarray) -> tuple[np.ndarray, float]:
        """Preprocess RGB image to 1024x1024 input tensor and extract image embeddings."""
        orig_h, orig_w = image.shape[:2]

        scale = float(self.target_size) / max(orig_h, orig_w)
        new_h, new_w = int(orig_h * scale + 0.5), int(orig_w * scale + 0.5)
        sam_img = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        sam_img = normalize_image(sam_img, mean=self.pixel_mean, std=self.pixel_std, to_chw=False)

        pad_h, pad_w = self.target_size - new_h, self.target_size - new_w
        padded_img = np.pad(sam_img, ((0, pad_h), (0, pad_w), (0, 0)), mode="constant")

        input_tensor = padded_img.transpose((2, 0, 1))[None, ...].astype(np.float32)
        image_embedding = self.encoder_session.run(None, {"image": input_tensor})[0]

        return image_embedding, scale

    def _generate_point_grid(self) -> np.ndarray:
        """Generate uniform point grid in relative [0, 1] coordinates."""
        offset = 1.0 / (2 * self.points_per_side)
        points_one_side = np.linspace(offset, 1.0 - offset, self.points_per_side)

        pts_x = np.tile(points_one_side[None, :], (self.points_per_side, 1)).flatten()
        pts_y = np.tile(points_one_side[:, None], (1, self.points_per_side)).flatten()

        return np.stack([pts_x, pts_y], axis=1)

    def generate_masks(self, image: str | Path | np.ndarray, **kwargs) -> SegmentationResult:
        """Execute Automatic Mask Generation (AMG) over input image.

        Args:
            image: Input image (file path or RGB NumPy array).
            kwargs: Optional threshold overrides ('pred_iou_thresh', 'stability_score_thresh', 'iou_threshold').

        Returns:
            SegmentationResult: Dataclass containing bounding boxes, masks, and scores.
        """
        pred_iou_thresh = kwargs.get("pred_iou_thresh", self.default_pred_iou_thresh)
        stability_score_thresh = kwargs.get("stability_score_thresh", self.default_stability_score_thresh)
        box_nms_thresh = kwargs.get("iou_threshold", self.default_box_nms_thresh)

        if isinstance(image, (str, Path)):
            image = load_image(image, color_mode="RGB")

        orig_h, orig_w = image.shape[:2]

        # Encode Image
        image_embedding, scale = self._encode_image(image)

        # Point Grid Generation & Coordinate Scaling
        points_rel = self._generate_point_grid()
        points_orig = points_rel * np.array([orig_w, orig_h])
        points_resized = points_orig * scale

        all_masks, all_scores, all_boxes = [], [], []

        # Batched Mask Decoder Pass
        for i in range(0, len(points_resized), self.points_per_batch):
            batch_pts = points_resized[i : i + self.points_per_batch]
            batch_size = len(batch_pts)

            ort_inputs = {
                "image_embeddings": image_embedding,
                "point_coords": batch_pts[:, None, :].astype(np.float32),
                "point_labels": np.ones((batch_size, 1), dtype=np.float32),
                "mask_input": np.zeros((batch_size, 1, 256, 256), dtype=np.float32),
                "has_mask_input": np.zeros((batch_size,), dtype=np.float32),
                "orig_im_size": np.array([orig_h, orig_w], dtype=np.float32),
            }

            masks, iou_preds, _ = self.decoder_session.run(None, ort_inputs)
            masks = masks[:, 0, :, :]
            iou_preds = iou_preds[:, 0]

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

        # Safe handling of empty detection batches
        if len(all_boxes) == 0:
            return SegmentationResult(
                image=image,
                boxes=np.empty((0, 4), dtype=np.float32),
                masks=np.empty((0, orig_h, orig_w), dtype=bool),
                scores=np.empty((0,), dtype=np.float32),
            )

        # Consolidate and Apply NMS
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


