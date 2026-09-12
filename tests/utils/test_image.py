"""Comprehensive unit tests for spatialhub.utils.image module."""

from pathlib import Path
import tempfile
import cv2
import numpy as np
import pytest

from spatialhub.utils.image import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    extract_foreground_bbox,
    load_image,
    non_max_suppression,
    normalize_image,
    square_crop_and_resize,
)


class TestLoadImage:
    """Test suite for load_image function across diverse formats and inputs."""

    def test_load_image_from_file_rgb(self, sample_image_file: str):
        img = load_image(sample_image_file, color_mode="RGB")
        assert isinstance(img, np.ndarray)
        assert img.ndim == 3
        assert img.shape[2] == 3

    def test_load_image_from_file_rgba(self, sample_image_file: str):
        img = load_image(sample_image_file, color_mode="RGBA")
        assert isinstance(img, np.ndarray)
        assert img.ndim == 3
        assert img.shape[2] == 4

    def test_load_image_from_file_gray(self, sample_image_file: str):
        img = load_image(sample_image_file, color_mode="GRAY")
        assert isinstance(img, np.ndarray)
        assert img.ndim == 2

    @pytest.mark.parametrize("mode", ["rgb", "rgba", "gray", "RGB", "RGBA", "GRAY"])
    def test_load_image_case_insensitive_color_mode(self, sample_image_file: str, mode: str):
        img = load_image(sample_image_file, color_mode=mode)
        assert isinstance(img, np.ndarray)

    def test_load_image_from_array_2d_gray(self):
        arr = np.zeros((100, 120), dtype=np.uint8)
        gray = load_image(arr, color_mode="GRAY")
        assert gray.shape == (100, 120)

        rgb = load_image(arr, color_mode="RGB")
        assert rgb.shape == (100, 120, 3)

        rgba = load_image(arr, color_mode="RGBA")
        assert rgba.shape == (100, 120, 4)

    def test_load_image_from_array_3d_single_channel(self):
        arr = np.zeros((100, 120, 1), dtype=np.uint8)
        gray = load_image(arr, color_mode="GRAY")
        assert gray.shape == (100, 120)

        rgb = load_image(arr, color_mode="RGB")
        assert rgb.shape == (100, 120, 3)

    def test_load_image_from_array_3d_rgb(self):
        arr = np.zeros((100, 120, 3), dtype=np.uint8)
        rgb = load_image(arr, color_mode="RGB")
        assert rgb.shape == (100, 120, 3)

        rgba = load_image(arr, color_mode="RGBA")
        assert rgba.shape == (100, 120, 4)

        gray = load_image(arr, color_mode="GRAY")
        assert gray.shape == (100, 120)

    def test_load_image_from_array_4d_rgba(self):
        arr = np.zeros((100, 120, 4), dtype=np.uint8)
        rgba = load_image(arr, color_mode="RGBA")
        assert rgba.shape == (100, 120, 4)

        rgb = load_image(arr, color_mode="RGB")
        assert rgb.shape == (100, 120, 3)

        gray = load_image(arr, color_mode="GRAY")
        assert gray.shape == (100, 120)

    def test_load_image_missing_file_raises_filenotfound(self):
        with pytest.raises(FileNotFoundError, match="Could not read image"):
            load_image("non_existent_image_file_12345.png")

    def test_load_image_invalid_mode_raises_valueerror(self, sample_image_file: str):
        with pytest.raises(ValueError, match="Unsupported color_mode"):
            load_image(sample_image_file, color_mode="CMYK")

    def test_load_image_invalid_dimensions_raises_valueerror(self):
        arr_5d = np.zeros((1, 2, 3, 4, 5), dtype=np.uint8)
        with pytest.raises(ValueError, match="Unsupported image dimensions"):
            load_image(arr_5d)


class TestNormalizeImage:
    """Test suite for normalize_image function."""

    def test_normalize_image_default_params(self, sample_rgb_image: np.ndarray):
        img_float = sample_rgb_image.astype(np.float32) / 255.0
        normed = normalize_image(img_float, to_chw=True)

        assert normed.dtype == np.float32
        assert normed.shape == (3, sample_rgb_image.shape[0], sample_rgb_image.shape[1])

    def test_normalize_image_to_chw_false(self, sample_rgb_image: np.ndarray):
        img_float = sample_rgb_image.astype(np.float32) / 255.0
        normed = normalize_image(img_float, to_chw=False)

        assert normed.dtype == np.float32
        assert normed.shape == (sample_rgb_image.shape[0], sample_rgb_image.shape[1], 3)

    def test_normalize_image_custom_mean_std(self):
        img = np.ones((50, 50, 3), dtype=np.float32) * 2.0
        custom_mean = np.array([1.0, 1.0, 1.0], dtype=np.float32)
        custom_std = np.array([2.0, 2.0, 2.0], dtype=np.float32)

        normed = normalize_image(img, mean=custom_mean, std=custom_std, to_chw=False)
        expected = (2.0 - 1.0) / 2.0
        assert np.allclose(normed, expected)


class TestExtractForegroundBbox:
    """Test suite for extract_foreground_bbox."""

    def test_extract_foreground_bbox_rgb(self):
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        # Put a 20x20 foreground square at y: 30..50, x: 40..60
        img[30:50, 40:60] = [255, 128, 64]

        x_min, y_min, x_max, y_max = extract_foreground_bbox(img)
        assert x_min == 40
        assert y_min == 30
        assert x_max == 60
        assert y_max == 50

    def test_extract_foreground_bbox_rgba_alpha(self):
        img = np.zeros((100, 100, 4), dtype=np.uint8)
        # Alpha channel > 0 at y: 10..30, x: 20..40
        img[10:30, 20:40, 3] = 255

        x_min, y_min, x_max, y_max = extract_foreground_bbox(img)
        assert x_min == 20
        assert y_min == 10
        assert x_max == 40
        assert y_max == 30

    def test_extract_foreground_bbox_all_black_fallback_full_frame(self):
        img = np.zeros((80, 120, 3), dtype=np.uint8)
        x_min, y_min, x_max, y_max = extract_foreground_bbox(img)
        assert (x_min, y_min, x_max, y_max) == (0, 0, 120, 80)

    def test_extract_foreground_bbox_all_transparent_fallback_full_frame(self):
        img = np.zeros((80, 120, 4), dtype=np.uint8)
        x_min, y_min, x_max, y_max = extract_foreground_bbox(img)
        assert (x_min, y_min, x_max, y_max) == (0, 0, 120, 80)


class TestSquareCropAndResize:
    """Test suite for square_crop_and_resize."""

    def test_square_crop_and_resize_wide_bbox(self):
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        # Wide bbox: width=40, height=20
        bbox = (10, 10, 50, 30)
        cropped = square_crop_and_resize(img, bbox, target_size=None)
        assert cropped.shape == (40, 40, 3)

    def test_square_crop_and_resize_tall_bbox(self):
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        # Tall bbox: width=20, height=60
        bbox = (10, 10, 30, 70)
        cropped = square_crop_and_resize(img, bbox, target_size=None)
        assert cropped.shape == (60, 60, 3)

    def test_square_crop_and_resize_with_target_size(self):
        img = np.zeros((200, 200, 3), dtype=np.uint8)
        bbox = (20, 20, 80, 80)
        cropped = square_crop_and_resize(img, bbox, target_size=224)
        assert cropped.shape == (224, 224, 3)

    def test_square_crop_and_resize_2d_grayscale(self):
        img = np.zeros((100, 100), dtype=np.uint8)
        bbox = (10, 10, 50, 30)
        cropped = square_crop_and_resize(img, bbox, target_size=64)
        assert cropped.shape == (64, 64)

    def test_square_crop_and_resize_negative_coords_raises(self):
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        with pytest.raises(ValueError, match="non-negative"):
            square_crop_and_resize(img, bbox=(-5, 10, 50, 50))

    def test_square_crop_and_resize_inverted_coords_raises(self):
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        # x_min >= x_max
        with pytest.raises(ValueError, match="Invalid bounding box dimensions"):
            square_crop_and_resize(img, bbox=(50, 10, 20, 50))
        # y_min >= y_max
        with pytest.raises(ValueError, match="Invalid bounding box dimensions"):
            square_crop_and_resize(img, bbox=(10, 50, 50, 20))

    def test_square_crop_and_resize_out_of_bounds_raises(self):
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        # x_max exceeds width 100
        with pytest.raises(ValueError, match="exceeds image bounds"):
            square_crop_and_resize(img, bbox=(10, 10, 120, 50))
        # y_max exceeds height 100
        with pytest.raises(ValueError, match="exceeds image bounds"):
            square_crop_and_resize(img, bbox=(10, 10, 50, 120))

    def test_square_crop_and_resize_invalid_length_raises(self):
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        with pytest.raises(ValueError, match="Expected bbox with 4 elements"):
            square_crop_and_resize(img, bbox=(10, 10, 50))

    def test_square_crop_and_resize_invalid_target_size_raises(self):
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        with pytest.raises(ValueError, match="positive integer"):
            square_crop_and_resize(img, bbox=(10, 10, 50, 50), target_size=0)


class TestNonMaxSuppression:
    """Test suite for non_max_suppression."""

    def test_non_max_suppression_empty(self):
        boxes = np.empty((0, 4), dtype=np.float32)
        scores = np.empty((0,), dtype=np.float32)
        keep = non_max_suppression(boxes, scores, iou_threshold=0.5)
        assert keep == []

    def test_non_max_suppression_single_box(self):
        boxes = np.array([[10.0, 10.0, 50.0, 50.0]], dtype=np.float32)
        scores = np.array([0.9], dtype=np.float32)
        keep = non_max_suppression(boxes, scores, iou_threshold=0.5)
        assert keep == [0]

    def test_non_max_suppression_overlapping_suppression(self):
        boxes = np.array([
            [10.0, 10.0, 50.0, 50.0],
            [12.0, 12.0, 48.0, 48.0],  # High overlap with box 0
            [100.0, 100.0, 200.0, 200.0],  # Distinct box
        ], dtype=np.float32)
        scores = np.array([0.9, 0.8, 0.95], dtype=np.float32)

        keep = non_max_suppression(boxes, scores, iou_threshold=0.5)
        assert 0 in keep
        assert 2 in keep
        assert 1 not in keep

    def test_non_max_suppression_threshold_extremes(self):
        boxes = np.array([
            [10.0, 10.0, 50.0, 50.0],
            [15.0, 15.0, 50.0, 50.0],
        ], dtype=np.float32)
        scores = np.array([0.9, 0.8], dtype=np.float32)

        # Threshold 1.0 keeps all boxes
        keep_all = non_max_suppression(boxes, scores, iou_threshold=1.0)
        assert len(keep_all) == 2

        # Threshold 0.0 suppresses any overlapping box
        keep_strict = non_max_suppression(boxes, scores, iou_threshold=0.0)
        assert keep_strict == [0]

    def test_non_max_suppression_negative_coords_raises(self):
        boxes = np.array([[-10.0, 10.0, 50.0, 50.0]], dtype=np.float32)
        scores = np.array([0.9], dtype=np.float32)
        with pytest.raises(ValueError, match="non-negative"):
            non_max_suppression(boxes, scores, iou_threshold=0.5)

    def test_non_max_suppression_inverted_coords_raises(self):
        boxes = np.array([[50.0, 10.0, 20.0, 50.0]], dtype=np.float32)
        scores = np.array([0.9], dtype=np.float32)
        with pytest.raises(ValueError, match="satisfy x1 < x2"):
            non_max_suppression(boxes, scores, iou_threshold=0.5)

    def test_non_max_suppression_invalid_iou_threshold_raises(self):
        boxes = np.array([[10.0, 10.0, 50.0, 50.0]], dtype=np.float32)
        scores = np.array([0.9], dtype=np.float32)
        with pytest.raises(ValueError, match="between 0.0 and 1.0"):
            non_max_suppression(boxes, scores, iou_threshold=-0.1)
        with pytest.raises(ValueError, match="between 0.0 and 1.0"):
            non_max_suppression(boxes, scores, iou_threshold=1.5)

    def test_non_max_suppression_mismatched_lengths_raises(self):
        boxes = np.array([[10.0, 10.0, 50.0, 50.0], [20.0, 20.0, 60.0, 60.0]], dtype=np.float32)
        scores = np.array([0.9], dtype=np.float32)  # length 1 vs 2
        with pytest.raises(ValueError, match="Mismatch between boxes shape"):
            non_max_suppression(boxes, scores, iou_threshold=0.5)

    def test_non_max_suppression_invalid_shape_raises(self):
        boxes = np.zeros((3, 3), dtype=np.float32)
        scores = np.array([0.9, 0.8, 0.7], dtype=np.float32)
        with pytest.raises(ValueError, match="Expected boxes of shape"):
            non_max_suppression(boxes, scores, iou_threshold=0.5)

