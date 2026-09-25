from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from spatialhub.core.runtime import create_ort_session, resolve_model_path
from spatialhub.structures import SegmentationResult
from spatialhub.utils import load_image, non_max_suppression

logger = logging.getLogger(__name__)


def crop_mask(masks: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    """Zeros out mask activations outside proposal bounding boxes.

    Operates in-place on prototype coordinate spatial grids.

    Args:
        masks: Spatial mask array of shape (N, H, W) or (H, W).
        boxes: Bounding box array of shape (N, 4) in [x1, y1, x2, y2] format.

    Returns:
        np.ndarray: Cropped mask array with outside activations set to zero.
    """
    boxes_int = boxes.astype(int)
    x1, y1, x2, y2 = boxes_int.T

    # Spatial dimensions H (y) and W (x)
    h, w = masks.shape[-2:]
    y, x = np.ogrid[:h, :w]

    # Broadcast (H, 1) and (1, W) against box coordinates (N, 1, 1)
    masks *= (
        (x >= x1[:, None, None]) &
        (x < x2[:, None, None]) &
        (y >= y1[:, None, None]) &
        (y < y2[:, None, None])
    )

    return masks


MODEL_REGISTRY: dict[str, dict[str, Any]] = {
    "s": {
        "filename": "FastSAM-s.onnx",
        "repo_id": "SpatialHub/fastsam-onnx",
    },
    "x": {
        "filename": "FastSAM-x.onnx",
        "repo_id": "SpatialHub/fastsam-onnx",
    },
}


class FastSAMAdapter:
    """Instance segmentation and proposal generation pipeline using FastSAM.

    Performs proposal generation, bounding box decoding, prototype mask matrix
    combination, and non-maximum suppression (NMS) over input images.
    """

    def __init__(
        self,
        model_path: str | Path | None = None,
        model_variant: str = "x",
        imgsz: int = 640,
        conf_threshold: float = 0.05,
        iou_threshold: float = 0.7,
        max_det: int = 50,
        providers: list[str] | str | None = None,
    ) -> None:
        """Initialize FastSAM segmentor.

        Args:
            model_path:
                Optional explicit path to local model binary. If None, resolves
                automatically from Hugging Face Hub using model_variant.
            model_variant:
                FastSAM model variant ('s' or 'x', default: 'x').
            imgsz:
                Target spatial resolution for input tensor.
            conf_threshold:
                Default confidence score threshold for proposals.
            iou_threshold:
                Default IoU threshold for Non-Maximum Suppression.
            max_det:
                Maximum number of output detections to retain after NMS (default: 50).
            providers:
                Execution providers.

        Raises:
            ValueError:
                If model_path is None and model_variant is not recognized.
        """
        if model_path is None:
            variant = str(model_variant).lower()
            if variant not in MODEL_REGISTRY:
                raise ValueError(
                    f"Unsupported model_variant '{model_variant}'. Supported variants: {sorted(MODEL_REGISTRY.keys())}"
                )
            filename = MODEL_REGISTRY[variant]["filename"]
            repo_id = MODEL_REGISTRY[variant]["repo_id"]
        else:
            filename = None
            repo_id = "SpatialHub/fastsam-onnx"

        resolved_path = resolve_model_path(
            model_path=model_path,
            repo_id=repo_id,
            filename=filename,
        )

        self.session = create_ort_session(
            model_path=resolved_path,
            providers=providers,
        )
        self.input_name = self.session.get_inputs()[0].name

        self.imgsz = imgsz
        self.default_conf_threshold = conf_threshold
        self.default_iou_threshold = iou_threshold
        self.default_max_det = max_det

    def _preprocess(self, image: np.ndarray, orig_h: int, orig_w: int) -> tuple[np.ndarray, tuple[float, int, int, int, int]]:
        """Scale and letterbox pad input image to network input resolution.

        Transforms image from (orig_h, orig_w, 3) to network tensor (1, 3, imgsz, imgsz).

        Args:
            image: Source RGB array of shape (orig_h, orig_w, 3).
            orig_h: Original spatial height.
            orig_w: Original spatial width.

        Returns:
            tuple containing:
                - input_tensor: Normalized NCHW tensor of shape (1, 3, imgsz, imgsz).
                - letterbox: Tuple of (scale, pad_top, pad_bottom, pad_left, pad_right).
        """
        # Isotropic scale factor mapping original image to unpadded canvas region
        scale = min(self.imgsz / orig_h, self.imgsz / orig_w)
        new_w, new_h = int(orig_w * scale), int(orig_h * scale)
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        # Symmetric border padding to reach (imgsz, imgsz)
        dw, dh = (self.imgsz - new_w) / 2, (self.imgsz - new_h) / 2
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        padded = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))

        # HWC uint8 -> CHW float32 normalized to [0.0, 1.0] -> NCHW
        input_tensor = padded.transpose((2, 0, 1)).astype(np.float32) / 255.0
        input_tensor = np.expand_dims(input_tensor, axis=0)

        letterbox = (scale, top, bottom, left, right)
        return input_tensor, letterbox

    def _decode_predictions(
        self,
        output: np.ndarray,
        conf_threshold: float,
        iou_threshold: float,
        max_det: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Decode box predictions and apply confidence filtering and NMS.

        Parses raw output tensor (1, 37, 8400) where 37 channels contain:
        - Indices 0..3: Bounding box coordinates [cx, cy, w, h] in network canvas space.
        - Index 4: Proposal confidence score.
        - Indices 5..36: 32 prototype mask coefficients.

        Args:
            output: Raw model prediction tensor of shape (1, 37, 8400).
            conf_threshold: Confidence score cutoff threshold.
            iou_threshold: Intersection over Union threshold for NMS.
            max_det: Maximum detections to retain after NMS (default: default_max_det).

        Returns:
            tuple containing:
                - boxes: Bounding boxes [x1, y1, x2, y2] in [0, imgsz] space of shape (N, 4).
                - scores: Confidence scores of shape (N,).
                - mask_coeffs: Prototype mask weight vectors of shape (N, 32).
        """
        max_det = max_det if max_det is not None else self.default_max_det

        # Transpose (1, 37, 8400) -> (8400, 37)
        preds = output[0].T
        scores = preds[:, 4]
        keep = scores > conf_threshold

        preds = preds[keep]
        scores = scores[keep]

        if len(preds) == 0:
            return (
                np.empty((0, 4), dtype=np.float32),
                np.empty((0,), dtype=np.float32),
                np.empty((0, 32), dtype=np.float32),
            )

        # Convert center [cx, cy, w, h] to corner [x1, y1, x2, y2]
        boxes = preds[:, :4]
        cx, cy, w, h = boxes.T
        boxes_xyxy = np.column_stack((cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2))

        # Clamp box coordinates to [0, imgsz] canvas boundary and filter degenerate boxes
        boxes_xyxy[:, [0, 2]] = np.clip(boxes_xyxy[:, [0, 2]], 0.0, float(self.imgsz))
        boxes_xyxy[:, [1, 3]] = np.clip(boxes_xyxy[:, [1, 3]], 0.0, float(self.imgsz))

        valid = (boxes_xyxy[:, 2] > boxes_xyxy[:, 0]) & (boxes_xyxy[:, 3] > boxes_xyxy[:, 1])
        if not np.any(valid):
            return (
                np.empty((0, 4), dtype=np.float32),
                np.empty((0,), dtype=np.float32),
                np.empty((0, 32), dtype=np.float32),
            )

        boxes_xyxy = boxes_xyxy[valid]
        scores = scores[valid]
        preds = preds[valid]

        keep = non_max_suppression(boxes_xyxy, scores, iou_threshold)
        if max_det > 0 and len(keep) > max_det:
            keep = keep[:max_det]

        return (
            boxes_xyxy[keep].astype(np.float32),
            scores[keep].astype(np.float32),
            preds[keep, 5:].astype(np.float32),
        )

    def _scale_boxes(
        self,
        boxes: np.ndarray,
        image_height: int,
        image_width: int,
        letterbox: tuple[float, int, int, int, int],
    ) -> np.ndarray:
        """Map letterboxed box coordinates back to original image space.

        Converts [x1, y1, x2, y2] from (imgsz, imgsz) canvas space back to
        (image_height, image_width) by removing padding and dividing by scale factor.

        Args:
            boxes: Bounding boxes of shape (N, 4) in canvas coordinates.
            image_height: Target original image height.
            image_width: Target original image width.
            letterbox: Tuple of (scale, pad_top, pad_bottom, pad_left, pad_right).

        Returns:
            np.ndarray: Rescaled bounding box array of shape (N, 4).
        """
        scale, pad_top, _, pad_left, _ = letterbox

        scaled_boxes = boxes.copy()
        scaled_boxes[:, [0, 2]] = (scaled_boxes[:, [0, 2]] - pad_left) / scale
        scaled_boxes[:, [1, 3]] = (scaled_boxes[:, [1, 3]] - pad_top) / scale
        scaled_boxes[:, [0, 2]] = np.clip(scaled_boxes[:, [0, 2]], 0, image_width)
        scaled_boxes[:, [1, 3]] = np.clip(scaled_boxes[:, [1, 3]], 0, image_height)

        return scaled_boxes.astype(np.float32)

    def _decode_masks(
        self,
        mask_coeffs: np.ndarray,
        prototypes: np.ndarray,
        boxes: np.ndarray,
        image_height: int,
        image_width: int,
        letterbox: tuple[float, int, int, int, int],
    ) -> np.ndarray:
        """Decode mask coefficients and map prototype masks to original image resolution.

        Pipeline stages:
        1. Prototype synthesis: Linear combination of 32 prototype masks with mask coefficients:
           (N, 32) @ (32, 160*160) -> (N, 160, 160) raw logit activations.
        2. Prototype cropping: Maps boxes from canvas space (640, 640) to prototype space
           (160, 160) using ratio = imgsz / mask_width = 4.0, zeroing out outer regions.
        3. Letterbox unpadding: Slices out padding margins in prototype space.
        4. Spatial interpolation: Bilinearly resizes unpadded prototype masks directly to
           (image_width, image_height) and thresholds at > 0.0 logit (sigmoid > 0.5).

        Args:
            mask_coeffs: Coefficient matrix of shape (N, 32).
            prototypes: Prototype mask tensor of shape (32, mask_height, mask_width), e.g. (32, 160, 160).
            boxes: Bounding boxes in network canvas space of shape (N, 4).
            image_height: Target output image height.
            image_width: Target output image width.
            letterbox: Tuple of (scale, pad_top, pad_bottom, pad_left, pad_right).

        Returns:
            np.ndarray: Binary boolean segmentation mask array of shape (N, image_height, image_width).
        """
        _, pad_top, pad_bottom, pad_left, pad_right = letterbox

        n_masks = len(mask_coeffs)
        if n_masks == 0:
            return np.empty((0, image_height, image_width), dtype=bool)

        num_channels, mask_height, mask_width = prototypes.shape

        # Linear combination: (N, 32) @ (32, mask_height * mask_width) -> (N, mask_height, mask_width)
        masks = mask_coeffs @ prototypes.reshape(num_channels, -1)
        masks = masks.reshape(-1, mask_height, mask_width)

        # Scale canvas boxes (640, 640) to prototype coordinates (160, 160) via stride ratio
        ratio = self.imgsz / mask_width
        boxes_proto = boxes / ratio
        masks = crop_mask(masks, boxes_proto)

        # Compute and crop letterbox padding margins in prototype space
        p_top = int(pad_top / ratio)
        p_bottom = int(pad_bottom / ratio)
        p_left = int(pad_left / ratio)
        p_right = int(pad_right / ratio)

        masks = masks[:, p_top : mask_height - p_bottom, p_left : mask_width - p_right]

        # Multi-channel batched bilinear interpolation chunked by 64 channels
        chunk_size = 64
        out_masks_list: list[np.ndarray] = []

        for i in range(0, n_masks, chunk_size):
            chunk = masks[i : i + chunk_size]
            chunk_hwc = chunk.transpose(1, 2, 0)
            final_resized = cv2.resize(chunk_hwc, (image_width, image_height), interpolation=cv2.INTER_LINEAR)

            if final_resized.ndim == 2:
                final_resized = final_resized[:, :, None]

            # Threshold raw logits > 0.0 (equivalent to sigmoid(logit) > 0.5)
            out_masks_list.append((final_resized.transpose(2, 0, 1) > 0.0))

        return np.concatenate(out_masks_list, axis=0) if len(out_masks_list) > 1 else out_masks_list[0]

    def _postprocess(
        self,
        image: np.ndarray,
        outputs: list[np.ndarray],
        orig_h: int,
        orig_w: int,
        letterbox: tuple[float, int, int, int, int],
        conf_threshold: float,
        iou_threshold: float,
        max_det: int | None = None,
    ) -> SegmentationResult:
        """Decode detections, evaluate mask prototypes, and map back to input coordinates.

        Args:
            image: Source RGB image array.
            outputs: Raw ONNX model inference outputs (detections, prototypes).
            orig_h: Original image height.
            orig_w: Original image width.
            letterbox: Tuple of (scale, pad_top, pad_bottom, pad_left, pad_right).
            conf_threshold: Confidence score threshold for proposals.
            iou_threshold: IoU threshold for Non-Maximum Suppression.
            max_det: Maximum number of detections to retain.

        Returns:
            SegmentationResult: Filtered boxes, masks, and confidence scores.
        """
        max_det = max_det if max_det is not None else self.default_max_det
        boxes, scores, mask_coeffs = self._decode_predictions(
            output=outputs[0],
            conf_threshold=conf_threshold,
            iou_threshold=iou_threshold,
            max_det=max_det,
        )

        if len(boxes) == 0:
            return SegmentationResult(
                image=image,
                boxes=np.empty((0, 4), dtype=np.float32),
                masks=np.empty((0, orig_h, orig_w), dtype=bool),
                scores=np.empty((0,), dtype=np.float32),
            )

        prototypes = outputs[1][0]
        masks = self._decode_masks(
            mask_coeffs,
            prototypes,
            boxes=boxes,
            image_height=orig_h,
            image_width=orig_w,
            letterbox=letterbox,
        )
        scaled_boxes = self._scale_boxes(
            boxes,
            image_height=orig_h,
            image_width=orig_w,
            letterbox=letterbox,
        )

        return SegmentationResult(
            image=image,
            boxes=scaled_boxes.astype(np.float32),
            masks=masks,
            scores=scores.astype(np.float32),
        )

    def generate_masks(
        self,
        image: str | Path | np.ndarray,
        conf_threshold: float | None = None,
        iou_threshold: float | None = None,
        max_det: int | None = None,
    ) -> SegmentationResult:
        """Generate object proposals and instance segmentation masks.

        Args:
            image: Input image (file path or RGB NumPy array).
            conf_threshold: Overrides default confidence score threshold.
            iou_threshold: Overrides default NMS IoU threshold.
            max_det: Overrides default maximum detections threshold.

        Returns:
            SegmentationResult: Dataclass containing image, bounding boxes, binary masks, and scores.
        """
        conf_thresh = conf_threshold if conf_threshold is not None else self.default_conf_threshold
        iou_thresh = iou_threshold if iou_threshold is not None else self.default_iou_threshold
        max_detections = max_det if max_det is not None else self.default_max_det

        if isinstance(image, (str, Path)):
            image = load_image(image, color_mode="RGB")

        orig_h, orig_w = image.shape[:2]

        # Preprocessing
        input_tensor, letterbox = self._preprocess(image, orig_h, orig_w)

        # Model inference
        outputs = self.session.run(None, {self.input_name: input_tensor})

        # Postprocessing
        return self._postprocess(
            image=image,
            outputs=outputs,
            orig_h=orig_h,
            orig_w=orig_w,
            letterbox=letterbox,
            conf_threshold=conf_thresh,
            iou_threshold=iou_thresh,
            max_det=max_detections,
        )

    def __enter__(self) -> FastSAMAdapter:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    def close(self) -> None:
        """Release underlying ONNX Runtime inference session and associated resources."""
        if getattr(self, "session", None) is not None:
            self.session = None

