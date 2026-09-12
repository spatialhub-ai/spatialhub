"""Shared pytest fixtures for spatialhub test suite."""

from pathlib import Path
import tempfile
import cv2
import numpy as np
import pytest


@pytest.fixture
def sample_rgb_image() -> np.ndarray:
    """Generate synthetic uint8 RGB image of shape (480, 640, 3)."""
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.circle(image, (320, 240), 100, (255, 0, 0), -1)
    cv2.rectangle(image, (100, 100), (200, 200), (0, 255, 0), -1)
    return image


@pytest.fixture
def sample_grayscale_image() -> np.ndarray:
    """Generate synthetic uint8 Grayscale image of shape (480, 640)."""
    image = np.zeros((480, 640), dtype=np.uint8)
    cv2.circle(image, (320, 240), 100, 255, -1)
    return image


@pytest.fixture
def sample_image_file(sample_rgb_image: np.ndarray) -> str:
    """Save synthetic RGB image to temporary file and return path string."""
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        save_path = tmp.name

    cv2.imwrite(save_path, cv2.cvtColor(sample_rgb_image, cv2.COLOR_RGB2BGR))
    yield save_path

    path_obj = Path(save_path)
    if path_obj.exists():
        path_obj.unlink()


@pytest.fixture
def sample_pair_image_files(sample_rgb_image: np.ndarray) -> tuple[str, str]:
    """Save pair of synthetic RGB images of different resolutions to temporary files."""
    img_a = sample_rgb_image
    img_b = cv2.resize(sample_rgb_image, (800, 600))

    with tempfile.NamedTemporaryFile(suffix="_a.png", delete=False) as tmp_a:
        path_a = tmp_a.name
    with tempfile.NamedTemporaryFile(suffix="_b.png", delete=False) as tmp_b:
        path_b = tmp_b.name

    cv2.imwrite(path_a, cv2.cvtColor(img_a, cv2.COLOR_RGB2BGR))
    cv2.imwrite(path_b, cv2.cvtColor(img_b, cv2.COLOR_RGB2BGR))

    yield path_a, path_b

    for p in [path_a, path_b]:
        path_obj = Path(p)
        if path_obj.exists():
            path_obj.unlink()
