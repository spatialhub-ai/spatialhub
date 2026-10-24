"""Shared benchmark utilities and execution provider resolution."""

from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

from spatialhub.utils import load_image

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ONNX_DIR = PROJECT_ROOT / "onnx_weight"
DEFAULT_CACHE_DIR = PROJECT_ROOT / ".cache"
DEFAULT_IMAGES_DIR = DEFAULT_CACHE_DIR / "images"


def get_available_ort_providers(requested: str = "all") -> list[tuple[str | tuple[str, dict[str, str]], str]]:
    """Resolve available execution providers against requested filter.

    Args:
        requested: Provider selector ('cuda', 'cpu', or 'all').

    Returns:
        List of (provider_spec, display_label) tuples.
    """
    available = ort.get_available_providers()
    providers: list[tuple[str | tuple[str, dict[str, str]], str]] = []

    req = requested.lower()
    if req in ["all", "cuda"]:
        if "CUDAExecutionProvider" in available:
            cuda_spec = (
                "CUDAExecutionProvider",
                {
                    "arena_extend_strategy": "kSameAsRequested",
                },
            )
            providers.append((cuda_spec, "CUDA"))
        elif req == "cuda":
            logger.warning("CUDAExecutionProvider requested but not available: %s", available)

    if req in ["all", "cpu"]:
        if "CPUExecutionProvider" in available:
            providers.append(("CPUExecutionProvider", "CPU"))

    return providers


def compute_rotation_geodesic_degrees(r_pt: np.ndarray, r_ort: np.ndarray) -> float:
    """Compute angular geodesic rotation difference in degrees between two rotation matrices.

    Args:
        r_pt: PyTorch rotation matrix of shape (3, 3).
        r_ort: ONNX Runtime rotation matrix of shape (3, 3).

    Returns:
        Geodesic rotation error in degrees.
    """
    r_rel = np.matmul(r_pt, r_ort.T)
    trace = np.trace(r_rel)
    cos_theta = (trace - 1.0) / 2.0

    v = np.array([
        r_rel[2, 1] - r_rel[1, 2],
        r_rel[0, 2] - r_rel[2, 0],
        r_rel[1, 0] - r_rel[0, 1],
    ])
    sin_theta = np.linalg.norm(v) / 2.0

    theta_rad = np.arctan2(sin_theta, cos_theta)
    return float(np.degrees(theta_rad))


def collect_sample_image(data_dir: Path | str, default_shape: tuple[int, int] = (480, 640)) -> np.ndarray:
    """Collect a sample image from directory or generate a synthetic benchmark frame.

    Args:
        data_dir: Path to directory containing image files.
        default_shape: Fallback image dimensions as (height, width).

    Returns:
        RGB image array of shape (H, W, 3).
    """
    img_dir = Path(data_dir)
    valid_extensions = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

    if img_dir.exists() and img_dir.is_dir():
        for file in sorted(img_dir.iterdir()):
            if file.suffix.lower() in valid_extensions and file.is_file():
                return load_image(file, color_mode="RGB")

    h, w = default_shape
    logger.info("No sample image found in %s. Using synthetic %dx%d benchmark frame.", img_dir, w, h)
    img = np.zeros((h, w, 3), dtype=np.uint8)
    xx, yy = np.meshgrid(np.linspace(0, 255, w), np.linspace(0, 255, h))
    img[:, :, 0] = xx % 256
    img[:, :, 1] = yy % 256
    img[:, :, 2] = ((xx + yy) / 2) % 256
    return img


def collect_images(data_dir: Path | str, max_images: int = 20) -> list[Path]:
    """Scan dataset directory for image files, falling back to synthetic patterns if empty.

    Args:
        data_dir: Path to directory containing image files.
        max_images: Maximum number of images to return.

    Returns:
        List of Path objects to valid image files.
    """
    img_dir = Path(data_dir)
    image_paths: list[Path] = []

    valid_extensions = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

    if img_dir.exists() and img_dir.is_dir():
        for file in sorted(img_dir.iterdir()):
            if file.suffix.lower() in valid_extensions and file.is_file():
                image_paths.append(file)
            if len(image_paths) >= max_images:
                break

    if not image_paths:
        logger.warning("No image files found in %s. Generating synthetic test images in %s...", img_dir, img_dir)
        img_dir.mkdir(parents=True, exist_ok=True)

        for idx in range(min(max_images, 10)):
            h, w = 480, 640
            np.random.seed(42 + idx)
            img = np.zeros((h, w, 3), dtype=np.uint8)
            xx, yy = np.meshgrid(np.linspace(0, 255, w), np.linspace(0, 255, h))
            img[:, :, 0] = (xx + idx * 20) % 256
            img[:, :, 1] = (yy + idx * 30) % 256
            img[:, :, 2] = ((xx + yy) / 2 + idx * 10) % 256
            cv2.circle(img, (150 + idx * 30, 200), 50 + idx * 5, (255, 255, 255), -1)
            cv2.rectangle(img, (300, 100 + idx * 10), (500, 300), (50, 180, 240), -1)

            synth_path = img_dir / f"synthetic_sample_{idx:02d}.jpg"
            cv2.imwrite(str(synth_path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            image_paths.append(synth_path)

    return image_paths[:max_images]


