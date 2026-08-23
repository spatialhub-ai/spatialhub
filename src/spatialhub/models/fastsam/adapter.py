import logging
from pathlib import Path

import cv2
import numpy as np

from spatialhub.core.runtime import create_ort_session, resolve_model_path
from spatialhub.structures import SegmentationResult
from spatialhub.utils import load_image, non_max_suppression

logger = logging.getLogger(__name__)


def crop_mask(masks: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    """Zeros out pixels in masks that fall outside corresponding bounding boxes.

    Args:
        masks: Spatial mask array of shape (N, H, W).
        boxes: Bounding box array of shape (N, 4) in [x1, y1, x2, y2] format.

    Returns:
        np.ndarray: Cropped mask array of shape (N, H, W).
    """
    n, h, w = masks.shape
    x1, y1, x2, y2 = np.split(boxes[:, :, None], 4, axis=1)

    r = np.arange(w, dtype=x1.dtype)[None, None, :]
    c = np.arange(h, dtype=x1.dtype)[None, :, None]

    return masks * ((r >= x1) * (r < x2) * (c >= y1) * (c < y2))


class FastSAMAdapter:
    """ONNX Runtime adapter for FastSAM (YOLOv8-Seg) instance segmentation.

    Performs fast proposal generation, bounding box decoding, prototype mask matrix
    combination, and non-maximum suppression (NMS) over input images.
    """

    def __init__(
        self,
        model_path: str | Path | None = None,
        model_variant: str | None = "FastSAM-x",
        imgsz: int = 640,
        conf_threshold: float = 0.05,
        iou_threshold: float = 0.7,
        providers: list[str] | str | None = None,
    ) -> None:
        """Initialize FastSAM ONNX Runtime adapter.

        Args:
            model_path:
                Optional explicit path to FastSAM ONNX model. If None, resolves
                automatically from Hugging Face Hub.
            model_variant:
                FastSAM model variant ('FastSAM-x' or 'FastSAM-s', or short names 'x' or 's').
            imgsz:
                Target spatial resolution for input tensor.
            conf_threshold:
                Default confidence score threshold for proposals.
            iou_threshold:
                Default IoU threshold for Non-Maximum Suppression.
            providers:
                ONNX Runtime execution providers.
        """
        variant = str(model_variant or "FastSAM-x")
        if not variant.startswith("FastSAM-") and not variant.endswith(".onnx"):
            variant = f"FastSAM-{variant}"
        filename = f"{variant}.onnx" if not variant.endswith(".onnx") else variant

        resolved_path = resolve_model_path(
            model_path=model_path,
            repo_id="SpatialHub/fastsam-onnx",
            filename=filename if model_path is None else None,
        )

        self.session = create_ort_session(
            model_path=resolved_path,
            providers=providers,
        )
        self.input_name = self.session.get_inputs()[0].name

        self.imgsz = imgsz
        self.default_conf_threshold = conf_threshold
        self.default_iou_threshold = iou_threshold

    def _preprocess(self, image: np.ndarray, orig_h: int, orig_w: int) -> tuple[np.ndarray, tuple[float, int, int, int, int]]:
        """Scale and letterbox pad input image to network input resolution."""
        scale = min(self.imgsz / orig_h, self.imgsz / orig_w)
        new_w, new_h = int(orig_w * scale), int(orig_h * scale)
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        dw, dh = (self.imgsz - new_w) / 2, (self.imgsz - new_h) / 2
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        padded = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))

        input_tensor = padded.transpose((2, 0, 1)).astype(np.float32) / 255.0
        input_tensor = np.expand_dims(input_tensor, axis=0)

        letterbox = (scale, top, bottom, left, right)
        return input_tensor, letterbox

    def _decode_predictions(
        self,
        output: np.ndarray,
        conf_threshold: float,
        iou_threshold: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Decode box predictions and apply confidence filtering and NMS."""
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

        boxes = preds[:, :4]
        cx, cy, w, h = boxes.T
        boxes_xyxy = np.column_stack((cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2))

        keep = non_max_suppression(boxes_xyxy, scores, iou_threshold)
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
        """Map letterboxed box coordinates back to original image space."""
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
        """Decode mask coefficients and map prototype masks to original image resolution."""
        _, pad_top, pad_bottom, pad_left, pad_right = letterbox

        if len(mask_coeffs) == 0:
            return np.empty((0, image_height, image_width), dtype=bool)

        num_channels, mask_height, mask_width = prototypes.shape

        masks = mask_coeffs @ prototypes.reshape(num_channels, -1)
        masks = 1.0 / (1.0 + np.exp(-masks))
        masks = masks.reshape(-1, mask_height, mask_width)

        boxes_proto = boxes * (mask_width / self.imgsz)
        masks = crop_mask(masks, boxes_proto)

        masks = np.stack([cv2.resize(mask, (self.imgsz, self.imgsz), interpolation=cv2.INTER_LINEAR) for mask in masks])
        masks = masks[:, pad_top : self.imgsz - pad_bottom, pad_left : self.imgsz - pad_right]
        masks = np.stack([cv2.resize(mask, (image_width, image_height), interpolation=cv2.INTER_LINEAR) for mask in masks])

        return masks > 0.5

    def generate_masks(
        self,
        image: str | Path | np.ndarray,
        conf_threshold: float | None = None,
        iou_threshold: float | None = None,
    ) -> SegmentationResult:
        """Generate object proposals and instance segmentation masks.

        Args:
            image: Input image (file path or RGB NumPy array).
            conf_threshold: Overrides default confidence score threshold.
            iou_threshold: Overrides default NMS IoU threshold.

        Returns:
            SegmentationResult:
                Dataclass containing image, bounding boxes, binary masks, and scores.
        """
        conf_thresh = conf_threshold if conf_threshold is not None else self.default_conf_threshold
        iou_thresh = iou_threshold if iou_threshold is not None else self.default_iou_threshold

        if isinstance(image, (str, Path)):
            image = load_image(image, color_mode="RGB")

        orig_h, orig_w = image.shape[:2]

        # 1. Preprocessing
        input_tensor, letterbox = self._preprocess(image, orig_h, orig_w)

        # 2. ONNX Inference
        outputs = self.session.run(None, {self.input_name: input_tensor})

        # 3. Decode Detections
        boxes, scores, mask_coeffs = self._decode_predictions(
            output=outputs[0],
            conf_threshold=conf_thresh,
            iou_threshold=iou_thresh,
        )

        # Safe return on zero detections
        if len(boxes) == 0:
            return SegmentationResult(
                image=image,
                boxes=np.empty((0, 4), dtype=np.float32),
                masks=np.empty((0, orig_h, orig_w), dtype=bool),
                scores=np.empty((0,), dtype=np.float32),
            )

        # 4. Decode Masks & Scale Boxes
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


