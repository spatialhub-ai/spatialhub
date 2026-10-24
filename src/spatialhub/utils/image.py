"""Image processing utilities, disk I/O, depth scaling, normalization, and geometric transforms."""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path
from typing import Literal, Sequence

import cv2
import numpy as np

from .colors import to_color, validate_color_mode
from .constants import IMAGENET_MEAN, IMAGENET_STD
from .validation import validate_array

_CHANNELS_TO_COLOR_MODE: dict[int, str] = {
    1: "GRAY",
    3: "BGR",
    4: "BGRA",
}

_COLOR_MODE_TO_STORAGE_MODE: dict[str, str] = {
    "GRAY": "GRAY",
    "RGB": "BGR",
    "BGR": "BGR",
    "RGBA": "BGRA",
    "BGRA": "BGRA",
}

_DEPTH_UNIT_SCALES: dict[str, float] = {
    "m": 1.0,
    "cm": 0.01,
    "mm": 0.001,
}


def load_image(
    path: str | Path,
    color_mode: Literal["RGB", "RGBA", "GRAY", "BGR", "BGRA"] | None = "RGB",
) -> np.ndarray:
    """Read an image from disk and convert it to the requested color format.

    Args:
        path: Path to the image file.
        color_mode: Desired output color format ('RGB', 'RGBA', 'GRAY', 'BGR', 'BGRA', or None for raw).
            Defaults to 'RGB'.

    Returns:
        Image array in the requested color format.

    Raises:
        TypeError: If path is not a str or Path.
        FileNotFoundError: If the image file cannot be read from disk.
        ValueError: If color_mode is unsupported or image format is invalid.
    """
    if not isinstance(path, (str, Path)):
        raise TypeError(f"path must be a str or Path, got {type(path).__name__}")

    mode = validate_color_mode(color_mode)

    image_path = Path(path)
    image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    if mode is None:
        return image

    channels = 1 if image.ndim == 2 else image.shape[2]
    src_format = _CHANNELS_TO_COLOR_MODE.get(channels)
    if src_format is None:
        raise ValueError(f"Unsupported number of image channels: {channels}")

    return to_color(image, dst=mode, src=src_format)


def save_image(
    image: np.ndarray,
    path: str | Path,
    color_mode: Literal["RGB", "RGBA", "GRAY", "BGR", "BGRA"] | None = "RGB",
) -> Path:
    """Write an image array to disk.

    Creates parent directories if necessary. Converts color channels for
    standard disk serialization.

    Args:
        image: Image array of shape (H, W), (H, W, 1), (H, W, 3), or (H, W, 4).
        path: Output file path.
        color_mode: Color ordering of input image ('RGB', 'RGBA', 'GRAY', 'BGR', 'BGRA', or None).
            Defaults to 'RGB'.

    Returns:
        Path pointing to the written file.

    Raises:
        ValueError: If color_mode is unsupported or image dimensions are invalid.
        IOError: If writing to disk fails.
    """
    validate_array("image", image, ndim=(2, 3))

    mode = validate_color_mode(color_mode)

    out_arr = image.astype(np.uint8) * 255 if image.dtype == np.bool_ else image
    channels = 1 if out_arr.ndim == 2 else out_arr.shape[2]
    if channels not in _CHANNELS_TO_COLOR_MODE:
        raise ValueError(f"Unsupported number of image channels: {channels}")

    if mode is None:
        encoded = out_arr
    else:
        dst_format = _COLOR_MODE_TO_STORAGE_MODE[mode]
        encoded = to_color(out_arr, dst=dst_format, src=mode)

    save_path = Path(path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    success = cv2.imwrite(str(save_path), encoded)
    if not success:
        raise IOError(f"Failed to write image to: {save_path}")

    return save_path


def save_video(
    frames: np.ndarray | Sequence[np.ndarray],
    path: str | Path,
    fps: float | int = 30,
    color_mode: Literal["RGB", "BGR"] = "RGB",
    fourcc: str | None = None,
) -> Path:
    """Write an image sequence or batched video array to an encoded video file on disk.

    Creates parent directories if necessary. Converts frame color channels
    to standard BGR for video writing.

    Args:
        frames: Array of shape (T, H, W, 3) or sequence of T frame arrays of shape (H, W, 3).
        path: Output file path (e.g. .mp4, .avi).
        fps: Playback frame rate in frames per second (default: 30).
        color_mode: Color ordering of input frames ('RGB' or 'BGR'). Defaults to 'RGB'.
        fourcc: Four-character codec code (e.g. 'mp4v', 'avc1', 'vp09'). If None, derives
            from file extension ('mp4v' for .mp4, 'MJPG' for .avi).

    Returns:
        Path pointing to the written video file.

    Raises:
        ValueError: If frames sequence is empty, frame shapes mismatch, or dimensions are invalid.
        IOError: If video writer initialization or encoding fails.
    """
    if len(frames) == 0:
        raise ValueError("frames sequence cannot be empty")

    first_frame = np.asarray(frames[0])
    validate_array("frame", first_frame, ndim=3, shape_suffix=(3,))
    h, w = first_frame.shape[:2]

    save_path = Path(path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    ext = save_path.suffix.lower()

    if fourcc is None:
        fourcc_code = "mp4v" if ext in (".mp4", ".mov", ".m4v") else "MJPG"
    else:
        fourcc_code = fourcc

    ffmpeg_available = shutil.which("ffmpeg") is not None
    use_ffmpeg_transcode = ffmpeg_available and ext in (".mp4", ".mov", ".m4v")

    write_target_path = (
        save_path.with_name(f".tmp_{save_path.stem}_{os.getpid()}{save_path.suffix}")
        if use_ffmpeg_transcode
        else save_path
    )

    fourcc_val = cv2.VideoWriter_fourcc(*fourcc_code)
    writer = cv2.VideoWriter(str(write_target_path), fourcc_val, float(fps), (w, h))

    if not writer.isOpened():
        raise IOError(f"Failed to initialize VideoWriter for path: {write_target_path} with codec: {fourcc_code}")

    try:
        for idx, frame in enumerate(frames):
            frame_arr = np.asarray(frame)
            if frame_arr.shape[:2] != (h, w) or frame_arr.shape[-1] != 3:
                raise ValueError(
                    f"Frame {idx} shape {frame_arr.shape} does not match initial frame shape {(h, w, 3)}"
                )

            if color_mode.upper() == "RGB":
                bgr_frame = cv2.cvtColor(frame_arr, cv2.COLOR_RGB2BGR)
            elif color_mode.upper() == "BGR":
                bgr_frame = frame_arr
            else:
                raise ValueError(f"Unsupported color_mode: {color_mode}. Expected 'RGB' or 'BGR'.")

            writer.write(bgr_frame)
    finally:
        writer.release()

    if use_ffmpeg_transcode:
        try:
            cmd = [
                "ffmpeg",
                "-y",
                "-i", str(write_target_path),
                "-c:v", "libx264",
                "-pix_fmt", "yuv420p",
                "-movflags", "+faststart",
                str(save_path),
            ]
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            write_target_path.unlink(missing_ok=True)
        except Exception:
            if write_target_path.exists():
                if save_path.exists():
                    save_path.unlink()
                write_target_path.replace(save_path)

    return save_path


def scale_depth(
    depth: np.ndarray,
    depth_unit: Literal["m", "cm", "mm"] | float | int = "m",
) -> np.ndarray:
    """Scale a depth map array to metric meters.

    Args:
        depth: Depth map array of shape (H, W) or (..., H, W).
        depth_unit: Unit system of input depth ('m', 'cm', 'mm', or positive float multiplier to meters).
            Defaults to "m" (meters).

    Returns:
        Depth map array of shape (H, W) or (..., H, W) in float32 metric meters.

    Raises:
        ValueError: If depth array shape is invalid or depth_unit is unsupported.
    """
    d = np.asarray(depth)
    if d.ndim < 2:
        raise ValueError(f"depth array must have at least 2 dimensions, got shape {d.shape}")
    validate_array("depth", d, dtype=(np.floating, np.integer))

    if isinstance(depth_unit, str):
        scale = _DEPTH_UNIT_SCALES.get(depth_unit.lower())
        if scale is None:
            raise ValueError(
                f"Unsupported depth_unit '{depth_unit}'. Expected 'm', 'cm', 'mm', or positive float multiplier."
            )
    elif isinstance(depth_unit, (int, float)):
        if depth_unit <= 0:
            raise ValueError(f"depth_unit multiplier must be positive, got {depth_unit}")
        scale = float(depth_unit)
    else:
        raise ValueError(
            f"Unsupported depth_unit type {type(depth_unit).__name__}. Expected str or float multiplier."
        )

    depth_m = d.astype(np.float32)
    return depth_m if scale == 1.0 else (depth_m * scale)


def normalize_image(
    image: np.ndarray,
    mean: np.ndarray = IMAGENET_MEAN,
    std: np.ndarray = IMAGENET_STD,
    to_chw: bool = True,
) -> np.ndarray:
    """Standardize an RGB image using channel-wise mean and standard deviation.

    Args:
        image: RGB array of shape (H, W, 3) or (..., H, W, 3).
        mean: Array of channel means of shape (3,) or (1, 1, 3). Defaults to ImageNet mean.
        std: Array of channel standard deviations of shape (3,) or (1, 1, 3). Defaults to ImageNet std.
        to_chw: If True, transposes output from (..., H, W, C) to (..., C, H, W). Defaults to True.

    Returns:
        Normalized float32 array.
    """
    validate_array("image", image, shape_suffix=(3,))
    img_float = image.astype(np.float32)

    normalized = (img_float - mean) / std

    if to_chw:
        return np.ascontiguousarray(np.moveaxis(normalized, -1, -3)).astype(np.float32)

    return normalized.astype(np.float32)


def mask_to_bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    """Extract bounding box [x_min, y_min, x_max, y_max] from a binary mask.

    Args:
        mask: Array of shape (H, W) or (H, W, 1). Non-zero pixels are treated as foreground.

    Returns:
        Bounding box tuple of (x_min, y_min, x_max, y_max).
    """
    if mask.ndim == 3 and mask.shape[-1] == 1:
        mask = mask.squeeze(-1)

    validate_array("mask", mask, ndim=2)

    binary_mask = mask > 0
    rows = np.any(binary_mask, axis=1)
    cols = np.any(binary_mask, axis=0)

    if not np.any(rows):
        h, w = mask.shape[:2]
        return 0, 0, w, h

    y_indices = np.where(rows)[0]
    x_indices = np.where(cols)[0]
    return int(x_indices[0]), int(y_indices[0]), int(x_indices[-1] + 1), int(y_indices[-1] + 1)


def pad_to_square(
    image: np.ndarray,
    mode: str = "constant",
    pad_value: float | int = 0,
) -> np.ndarray:
    """Pad image along the shorter dimension to form a square canvas.

    Pads symmetrically so the original content remains centered.

    Args:
        image: Array of shape (H, W), (..., H, W), (H, W, C), or (..., H, W, C).
        mode: Padding mode. Defaults to "constant".
        pad_value: Value used to fill padded regions. Defaults to 0.

    Returns:
        Square array matching input batch and channel dimensions.
    """
    img = np.asarray(image)
    if img.ndim < 2:
        raise ValueError(f"image must have at least 2 dimensions, got {img.ndim}")

    has_channels = img.ndim >= 3 and img.shape[-1] in (1, 2, 3, 4)
    h, w = (img.shape[-3], img.shape[-2]) if has_channels else (img.shape[-2], img.shape[-1])
    if h == w:
        return img.copy()

    max_side = max(h, w)
    y_pad = max_side - h
    x_pad = max_side - w

    top = y_pad // 2
    bottom = y_pad - top
    left = x_pad // 2
    right = x_pad - left

    pad_width = [(0, 0)] * img.ndim
    if has_channels:
        pad_width[-3] = (top, bottom)
        pad_width[-2] = (left, right)
    else:
        pad_width[-2] = (top, bottom)
        pad_width[-1] = (left, right)

    return np.pad(img, pad_width, mode=mode, constant_values=pad_value)


def nms_boxes(
    boxes: np.ndarray,
    scores: np.ndarray,
    iou_threshold: float = 0.5,
) -> list[int]:
    """Non-Maximum Suppression (NMS) for bounding box filtering.

    Args:
        boxes: Array of shape (N, 4) in [x1, y1, x2, y2] format.
        scores: Array of shape (N,) or (N, 1) containing confidence scores.
        iou_threshold: Float threshold for overlapping area (0.0 <= iou_threshold <= 1.0).
            Defaults to 0.5.

    Returns:
        List of integer indices corresponding to the surviving boxes.

    Raises:
        ValueError: If boxes and scores shapes mismatch, if box coordinates are invalid,
            or if iou_threshold is outside [0.0, 1.0].
    """
    b_arr = np.asarray(boxes, dtype=np.float32)
    s_arr = np.asarray(scores, dtype=np.float32)

    if s_arr.ndim == 2 and s_arr.shape[-1] == 1:
        s_arr = s_arr.squeeze(-1)

    validate_array("boxes", b_arr, ndim=2, shape_suffix=(4,), dtype=np.floating)
    validate_array("scores", s_arr, ndim=1, dtype=np.floating)

    if len(b_arr) != len(s_arr):
        raise ValueError(
            f"Mismatch between boxes shape {b_arr.shape} and scores shape {s_arr.shape}"
        )

    if not (0.0 <= iou_threshold <= 1.0):
        raise ValueError(f"iou_threshold must be between 0.0 and 1.0, got {iou_threshold}")

    if len(b_arr) == 0:
        return []

    x1, y1, x2, y2 = b_arr[:, 0], b_arr[:, 1], b_arr[:, 2], b_arr[:, 3]

    if np.any(x1 < 0) or np.any(y1 < 0):
        raise ValueError("Bounding box coordinates must be non-negative")
    if np.any(x1 >= x2) or np.any(y1 >= y2):
        raise ValueError("Invalid bounding boxes: all boxes must satisfy x1 < x2 and y1 < y2")

    boxes_xywh = np.stack([x1, y1, x2 - x1, y2 - y1], axis=1)

    indices = cv2.dnn.NMSBoxes(
        bboxes=boxes_xywh,
        scores=s_arr,
        score_threshold=0.0,
        nms_threshold=float(iou_threshold),
    )

    if len(indices) == 0:
        return []

    return [int(i) for i in np.asarray(indices).flatten()]


__all__ = [
    "load_image",
    "save_image",
    "save_video",
    "scale_depth",
    "normalize_image",
    "mask_to_bbox",
    "pad_to_square",
    "nms_boxes",
]
