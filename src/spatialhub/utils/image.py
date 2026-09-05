from __future__ import annotations

from pathlib import Path
from typing import Literal

import cv2
import numpy as np


IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32,)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32,)

def load_image(image_input: str | Path | np.ndarray, color_mode: Literal["RGB", "RGBA", "GRAY"] = "RGB") -> np.ndarray:
    """
    Read an image from disk or accept an in-memory array and convert it to the requested color format.

    Args:
        image_input:
            Path to the image file or in-memory NumPy image array.
        color_mode:
            Desired output color format: 'RGB', 'RGBA', or 'GRAY'.

    Returns:
        Image as a NumPy array in the requested color format.

    Raises:
        FileNotFoundError:
            If the image file cannot be read from disk.
        ValueError:
            If the image has an unsupported format or channel layout.
    """
    if color_mode not in {"RGB", "RGBA", "GRAY"}:
        raise ValueError(
            f"Unsupported color_mode '{color_mode}'. "
            "Expected 'RGB', 'RGBA', or 'GRAY'."
        )

    if isinstance(image_input, np.ndarray):
        image = image_input
    else:
        image_path = Path(image_input)
        image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
        if image is None:
            raise FileNotFoundError(f"Could not read image: {image_path}")

    if image.ndim == 2:
        if color_mode == "GRAY":
            return image
        return cv2.cvtColor(
            image,
            cv2.COLOR_GRAY2RGBA if color_mode == "RGBA" else cv2.COLOR_GRAY2RGB,
        )

    elif image.ndim == 3:
        channels = image.shape[2]

        if channels == 1:
            if color_mode == "GRAY":
                return image[:, :, 0]
            return cv2.cvtColor(
                image,
                cv2.COLOR_GRAY2RGBA if color_mode == "RGBA" else cv2.COLOR_GRAY2RGB,
            )

        elif channels == 3:
            if color_mode == "RGB":
                return cv2.cvtColor(image, cv2.COLOR_BGR2RGB) if not isinstance(image_input, np.ndarray) else image
            elif color_mode == "RGBA":
                return cv2.cvtColor(image, cv2.COLOR_BGR2RGBA if not isinstance(image_input, np.ndarray) else cv2.COLOR_RGB2RGBA)
            elif color_mode == "GRAY":
                return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY if not isinstance(image_input, np.ndarray) else cv2.COLOR_RGB2GRAY)

        elif channels == 4:
            if color_mode == "RGB":
                return cv2.cvtColor(image, cv2.COLOR_BGRA2RGB if not isinstance(image_input, np.ndarray) else cv2.COLOR_RGBA2RGB)
            elif color_mode == "RGBA":
                return image
            elif color_mode == "GRAY":
                return cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY if not isinstance(image_input, np.ndarray) else cv2.COLOR_RGBA2GRAY)

        raise ValueError(
            f"Unsupported number of image channels: {channels}"
        )

    raise ValueError(
        f"Unsupported image dimensions: {image.ndim}, shape={image.shape}"
    )


def normalize_image(image: np.ndarray, mean: np.ndarray = IMAGENET_MEAN, std: np.ndarray = IMAGENET_STD, to_chw: bool = True) -> np.ndarray:
    """
    Converts a RGB image to float32, applies channel-wise normalization, and optionally transposes to Channel-First (CHW) format.

    Args:
        image: RGB array of shape (H, W, 3).
        mean: Array of shape (1, 1, 3) representing channel means.
        std: Array of shape (1, 1, 3) representing channel standard deviations.
        to_chw: If True, transposes output from (H, W, C) to (C, H, W).

    Returns:
        Normalized float32 tensor.
    """
    # Convert to float32
    img_float = image.astype(np.float32)

    # Apply Normalization
    normalized = (img_float - mean) / std

    if to_chw:
        # Transpose from (H, W, C) to (C, H, W)
        normalized = normalized.transpose(2, 0, 1)

    return normalized.astype(np.float32)

def extract_foreground_bbox(image: np.ndarray) -> tuple[int, int, int, int]:
    """
    Extracts bounding box [x_min, y_min, x_max, y_max] from non-background pixels.
    Supports RGBA (via alpha channel) or RGB (via thresholding non-black pixels).
    """
    if image.shape[2] == 4:
        alpha = image[:, :, 3]
        mask = alpha > 0
    else:
        mask = np.any(image[:, :, :3] > 0, axis=-1)

    nonzero_coords = np.argwhere(mask)
    if len(nonzero_coords) == 0:
        # Fallback to full frame if no mask found
        h, w = image.shape[:2]
        return 0, 0, w, h

    y_min, x_min = nonzero_coords.min(axis=0)
    y_max, x_max = nonzero_coords.max(axis=0)
    return int(x_min), int(y_min), int(x_max + 1), int(y_max + 1)

def square_crop_and_resize(image: np.ndarray, bbox: tuple[int, int, int, int], target_size: int | None = None) -> np.ndarray:
    """
    Crops an image to the bounding box, pads it to a perfect square to maintain aspect ratio, and optionally resizes it.

    Args:
        image: Array of shape (H, W, C).
        bbox: Tuple of (x_min, y_min, x_max, y_max).
        target_size: Optional integer to resize the final square to (H=W).
        
    Returns:
        Square image array of shape (target_size, target_size, C) or (max_side, max_side, C).
    """
    x_min, y_min, x_max, y_max = bbox
    crop = image[y_min:y_max, x_min:x_max]
    
    h, w = crop.shape[:2]
    max_side = max(h, w)

    # Create square canvas filled with zeros (black)
    square_crop = np.zeros((max_side, max_side, crop.shape[2]), dtype=crop.dtype)
    y_offset = (max_side - h) // 2
    x_offset = (max_side - w) // 2
    square_crop[y_offset : y_offset + h, x_offset : x_offset + w] = crop

    # Resize to network resolution if requested
    if target_size is not None and max_side != target_size:
        square_crop = cv2.resize(square_crop, (target_size, target_size), interpolation=cv2.INTER_LINEAR)

    return square_crop

def non_max_suppression(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float) -> list[int]:
    """Non-Maximum Suppression (NMS) for bounding box filtering.
    
    Args:
        boxes: Array of shape (N, 4) in [x1, y1, x2, y2] format.
        scores: Array of shape (N,) containing confidence scores.
        iou_threshold: Float threshold for overlapping area.
        
    Returns:
        List of indices corresponding to the boxes to keep.
    """
    if len(boxes) == 0:
        return []

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    
    # Sort by descending score
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        
        # Calculate intersection with remaining boxes
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        
        # Calculate Intersection over Union (IoU)
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        # Keep boxes with IoU less than or equal to the threshold
        inds = np.where(ovr <= iou_threshold)[0]
        order = order[inds + 1]

    return keep
