"""Comprehensive unit tests for spatialhub.utils.image module."""

from __future__ import annotations

from pathlib import Path
import cv2
import numpy as np
import pytest

from spatialhub.utils.constants import IMAGENET_MEAN, IMAGENET_STD
from spatialhub.utils.image import (
    load_image,
    mask_to_bbox,
    nms_boxes,
    normalize_image,
    pad_to_square,
    save_image,
    scale_depth,
)


@pytest.fixture
def sample_image_file(tmp_path: Path) -> Path:
    img_path = tmp_path / "test_image.png"
    arr = np.zeros((100, 120, 3), dtype=np.uint8)
    arr[20:80, 30:90] = [255, 128, 64]  # RGB
    # Write as BGR via OpenCV
    cv2.imwrite(str(img_path), cv2.cvtColor(arr, cv2.COLOR_RGB2BGR))
    return img_path


class TestLoadImage:
    """Test suite for load_image across color modes and file formats."""

    def test_load_image_rgb(self, sample_image_file: Path) -> None:
        img = load_image(sample_image_file, color_mode="RGB")
        assert img.shape == (100, 120, 3)
        assert img.dtype == np.uint8
        assert np.array_equal(img[50, 50], [255, 128, 64])

    def test_load_image_rgba(self, sample_image_file: Path) -> None:
        img = load_image(sample_image_file, color_mode="RGBA")
        assert img.shape == (100, 120, 4)
        assert img[50, 50, 3] == 255

    def test_load_image_gray(self, sample_image_file: Path) -> None:
        img = load_image(sample_image_file, color_mode="GRAY")
        assert img.shape == (100, 120)

    def test_load_image_raw_mode(self, sample_image_file: Path) -> None:
        img = load_image(sample_image_file, color_mode=None)
        assert img.ndim == 3

    def test_load_image_case_insensitive(self, sample_image_file: Path) -> None:
        img = load_image(sample_image_file, color_mode="rgb")
        assert img.shape == (100, 120, 3)

    def test_nonexistent_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="Could not read image"):
            load_image(tmp_path / "missing.png")

    def test_invalid_path_type_raises(self) -> None:
        with pytest.raises(TypeError, match="path must be a str or Path"):
            load_image(12345)  # type: ignore[arg-type]


class TestSaveImage:
    """Test suite for save_image."""

    def test_save_rgb_image(self, tmp_path: Path) -> None:
        arr = np.zeros((64, 64, 3), dtype=np.uint8)
        arr[:, :, 0] = 255
        target = tmp_path / "subdir" / "output.png"

        saved_path = save_image(arr, target, color_mode="RGB")
        assert saved_path == target
        assert target.exists()

        loaded = load_image(target, color_mode="RGB")
        assert np.array_equal(loaded, arr)

    def test_save_boolean_mask(self, tmp_path: Path) -> None:
        mask = np.zeros((32, 32), dtype=bool)
        mask[10:20, 10:20] = True
        target = tmp_path / "mask.png"

        save_image(mask, target, color_mode="GRAY")
        assert target.exists()

        loaded = load_image(target, color_mode="GRAY")
        assert loaded[15, 15] == 255
        assert loaded[0, 0] == 0

    def test_invalid_shape_raises(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="Unsupported number of image channels"):
            save_image(np.zeros((32, 32, 5), dtype=np.uint8), tmp_path / "out.png")


class TestScaleDepth:
    """Test suite for scale_depth."""

    def test_metric_units(self) -> None:
        d_mm = np.full((10, 10), 1000.0, dtype=np.float32)
        d_m = scale_depth(d_mm, depth_unit="mm")
        assert np.allclose(d_m, 1.0)

        d_cm = np.full((10, 10), 100.0, dtype=np.float32)
        assert np.allclose(scale_depth(d_cm, depth_unit="cm"), 1.0)

        d_meters = np.full((10, 10), 1.0, dtype=np.float32)
        assert np.allclose(scale_depth(d_meters, depth_unit="m"), 1.0)

    def test_custom_float_multiplier(self) -> None:
        depth = np.ones((5, 5), dtype=np.float32)
        scaled = scale_depth(depth, depth_unit=0.005)
        assert np.allclose(scaled, 0.005)

    def test_invalid_unit_raises(self) -> None:
        depth = np.ones((5, 5), dtype=np.float32)
        with pytest.raises(ValueError, match="Unsupported depth_unit 'km'"):
            scale_depth(depth, depth_unit="km")

        with pytest.raises(ValueError, match="depth_unit multiplier must be positive"):
            scale_depth(depth, depth_unit=-1.0)


class TestNormalizeImage:
    """Test suite for normalize_image."""

    def test_normalize_default_imagenet(self) -> None:
        img = np.ones((10, 10, 3), dtype=np.float32)
        normed = normalize_image(img, to_chw=True)
        assert normed.shape == (3, 10, 10)
        assert normed.dtype == np.float32

        expected_ch0 = (1.0 - IMAGENET_MEAN[0]) / IMAGENET_STD[0]
        assert np.isclose(normed[0, 0, 0], expected_ch0)

    def test_normalize_without_chw(self) -> None:
        img = np.zeros((10, 10, 3), dtype=np.float32)
        normed = normalize_image(img, to_chw=False)
        assert normed.shape == (10, 10, 3)

    def test_normalize_batched(self) -> None:
        batch_img = np.ones((2, 16, 16, 3), dtype=np.float32)
        normed = normalize_image(batch_img, to_chw=True)
        assert normed.shape == (2, 3, 16, 16)


class TestMaskToBbox:
    """Test suite for mask_to_bbox."""

    def test_standard_bbox_extraction(self) -> None:
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[20:60, 30:80] = 1

        bbox = mask_to_bbox(mask)
        assert bbox == (30, 20, 80, 60)

    def test_all_zeros_mask_fallback(self) -> None:
        mask = np.zeros((100, 120), dtype=np.uint8)
        bbox = mask_to_bbox(mask)
        assert bbox == (0, 0, 120, 100)

    def test_single_channel_3d_mask(self) -> None:
        mask = np.zeros((50, 50, 1), dtype=bool)
        mask[10:30, 15:40, 0] = True
        bbox = mask_to_bbox(mask)
        assert bbox == (15, 10, 40, 30)


class TestPadToSquare:
    """Test suite for pad_to_square."""

    def test_already_square_preserved(self) -> None:
        img = np.ones((50, 50, 3), dtype=np.float32)
        padded = pad_to_square(img)
        assert padded.shape == (50, 50, 3)
        assert padded is not img

    def test_landscape_padding(self) -> None:
        # H=40, W=100 -> target (100, 100)
        img = np.ones((40, 100, 3), dtype=np.uint8) * 255
        padded = pad_to_square(img, pad_value=0)
        assert padded.shape == (100, 100, 3)

        # Padding 60 pixels total: 30 top, 30 bottom
        assert np.all(padded[:30, :, :] == 0)
        assert np.all(padded[30:70, :, :] == 255)
        assert np.all(padded[70:, :, :] == 0)

    def test_portrait_padding_2d_grayscale(self) -> None:
        # H=100, W=60 -> target (100, 100)
        img = np.ones((100, 60), dtype=np.uint8) * 128
        padded = pad_to_square(img, pad_value=0)
        assert padded.shape == (100, 100)
        assert np.all(padded[:, :20] == 0)
        assert np.all(padded[:, 20:80] == 128)
        assert np.all(padded[:, 80:] == 0)

    def test_batched_padding(self) -> None:
        batch_img = np.zeros((2, 40, 80, 3), dtype=np.float32)
        padded = pad_to_square(batch_img)
        assert padded.shape == (2, 80, 80, 3)


class TestNmsBoxes:
    """Test suite for nms_boxes."""

    def test_nms_overlapping_boxes(self) -> None:
        # Box 0: (10, 10, 50, 50), score 0.9
        # Box 1: (12, 12, 52, 52), score 0.8 (high overlap with Box 0)
        # Box 2: (100, 100, 150, 150), score 0.7 (distinct)
        boxes = np.array([
            [10.0, 10.0, 50.0, 50.0],
            [12.0, 12.0, 52.0, 52.0],
            [100.0, 100.0, 150.0, 150.0],
        ], dtype=np.float32)
        scores = np.array([0.9, 0.8, 0.7], dtype=np.float32)

        surviving = nms_boxes(boxes, scores, iou_threshold=0.5)
        assert surviving == [0, 2]

    def test_nms_empty_boxes(self) -> None:
        boxes = np.zeros((0, 4), dtype=np.float32)
        scores = np.zeros((0,), dtype=np.float32)
        assert nms_boxes(boxes, scores) == []

    def test_nms_mismatch_raises(self) -> None:
        boxes = np.zeros((3, 4), dtype=np.float32)
        scores = np.zeros((2,), dtype=np.float32)
        with pytest.raises(ValueError, match="Mismatch between boxes shape"):
            nms_boxes(boxes, scores)

    def test_nms_invalid_coords_raises(self) -> None:
        boxes_neg = np.array([[-1.0, 0.0, 10.0, 10.0]], dtype=np.float32)
        with pytest.raises(ValueError, match="Bounding box coordinates must be non-negative"):
            nms_boxes(boxes_neg, np.array([0.5], dtype=np.float32))

        boxes_inverted = np.array([[20.0, 20.0, 10.0, 10.0]], dtype=np.float32)
        with pytest.raises(ValueError, match="all boxes must satisfy x1 < x2 and y1 < y2"):
            nms_boxes(boxes_inverted, np.array([0.5], dtype=np.float32))

    def test_nms_invalid_threshold_raises(self) -> None:
        boxes = np.array([[0.0, 0.0, 10.0, 10.0]], dtype=np.float32)
        scores = np.array([0.5], dtype=np.float32)
        with pytest.raises(ValueError, match=r"iou_threshold must be between 0.0 and 1.0"):
            nms_boxes(boxes, scores, iou_threshold=1.5)
