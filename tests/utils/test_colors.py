"""Comprehensive unit tests for spatialhub.utils.colors module."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from spatialhub.utils.colors import (
    resolve_colormap,
    to_color,
    validate_color,
    validate_color_mode,
)


class TestValidateColorMode:
    """Test suite for validate_color_mode."""

    def test_none_mode(self) -> None:
        assert validate_color_mode(None) is None

    @pytest.mark.parametrize("mode", ["RGB", "RGBA", "GRAY", "BGR", "BGRA"])
    def test_valid_uppercase_modes(self, mode: str) -> None:
        assert validate_color_mode(mode) == mode

    @pytest.mark.parametrize("mode", ["rgb", "rgba", "gray", "bgr", "bgra"])
    def test_case_insensitive_normalization(self, mode: str) -> None:
        assert validate_color_mode(mode) == mode.upper()

    def test_invalid_mode_raises(self) -> None:
        with pytest.raises(ValueError, match="Unsupported color_mode 'XYZ'"):
            validate_color_mode("XYZ")


class TestValidateColor:
    """Test suite for validate_color."""

    def test_valid_rgb_tuple(self) -> None:
        assert validate_color((255, 0, 128)) == (255, 0, 128)

    def test_valid_list_and_array(self) -> None:
        assert validate_color([0, 255, 0]) == (0, 255, 0)
        assert validate_color(np.array([10, 20, 30])) == (10, 20, 30)

    def test_invalid_length_raises(self) -> None:
        with pytest.raises(ValueError, match="must be a 3-element RGB sequence"):
            validate_color((255, 0))  # type: ignore[arg-type]

        with pytest.raises(ValueError, match="must be a 3-element RGB sequence"):
            validate_color((255, 0, 128, 255))  # type: ignore[arg-type]

    def test_out_of_range_raises(self) -> None:
        with pytest.raises(ValueError, match="RGB color channel values must be in"):
            validate_color((-1, 0, 255))

        with pytest.raises(ValueError, match="RGB color channel values must be in"):
            validate_color((256, 0, 100))


class TestResolveColormap:
    """Test suite for resolve_colormap."""

    def test_named_colormaps(self) -> None:
        assert resolve_colormap("inferno") == cv2.COLORMAP_INFERNO
        assert resolve_colormap("turbo") == cv2.COLORMAP_TURBO
        assert resolve_colormap("viridis") == cv2.COLORMAP_VIRIDIS
        assert resolve_colormap("gray") is None

    def test_int_passthrough(self) -> None:
        assert resolve_colormap(cv2.COLORMAP_HOT) == cv2.COLORMAP_HOT

    def test_none_passthrough(self) -> None:
        assert resolve_colormap(None) is None

    def test_unknown_string_fallback(self) -> None:
        assert resolve_colormap("unknown_map", default=cv2.COLORMAP_JET) == cv2.COLORMAP_JET


class TestToColor:
    """Test suite for to_color conversion and normalization utility."""

    def test_identity_conversion(self) -> None:
        img = np.zeros((32, 32, 3), dtype=np.uint8)
        out = to_color(img, "RGB", "RGB")
        assert np.array_equal(out, img)
        assert out is not img  # Must return copy

    def test_explicit_2d_gray_to_rgb_and_bgr(self) -> None:
        gray = np.full((16, 16), 128, dtype=np.uint8)
        rgb = to_color(gray, "RGB", "GRAY")
        assert rgb.shape == (16, 16, 3)
        assert np.all(rgb == 128)

        rgba = to_color(gray, "RGBA", "GRAY")
        assert rgba.shape == (16, 16, 4)

    def test_explicit_rgb_to_bgr_roundtrip(self) -> None:
        rgb = np.zeros((10, 10, 3), dtype=np.uint8)
        rgb[:, :, 0] = 255  # Pure red
        bgr = to_color(rgb, "BGR", "RGB")
        assert bgr[0, 0, 0] == 0    # Blue channel is 0
        assert bgr[0, 0, 2] == 255  # Red channel is 255

        restored = to_color(bgr, "RGB", "BGR")
        assert np.array_equal(restored, rgb)

    def test_explicit_rgb_to_rgba(self) -> None:
        rgb = np.ones((8, 8, 3), dtype=np.uint8) * 100
        rgba = to_color(rgb, "RGBA", "RGB")
        assert rgba.shape == (8, 8, 4)
        assert rgba[0, 0, 3] == 255  # Alpha channel is 255

    def test_explicit_rgba_to_bgra(self) -> None:
        rgba = np.zeros((8, 8, 4), dtype=np.uint8)
        rgba[:, :, 0] = 200  # Red
        rgba[:, :, 3] = 128  # Alpha
        bgra = to_color(rgba, "BGRA", "RGBA")
        assert bgra.shape == (8, 8, 4)
        assert bgra[0, 0, 2] == 200  # Red is in channel 2
        assert bgra[0, 0, 3] == 128  # Alpha preserved

    def test_explicit_single_channel_3d_gray(self) -> None:
        gray_3d = np.full((10, 10, 1), 75, dtype=np.uint8)
        rgb = to_color(gray_3d, "RGB", "GRAY")
        assert rgb.shape == (10, 10, 3)
        assert np.all(rgb == 75)

    def test_explicit_batched_conversion(self) -> None:
        batch_rgb = np.zeros((4, 16, 16, 3), dtype=np.uint8)
        batch_rgb[:, :, :, 0] = 255
        batch_bgr = to_color(batch_rgb, "BGR", "RGB")
        assert batch_bgr.shape == (4, 16, 16, 3)
        assert batch_bgr[0, 0, 0, 2] == 255

        batch_gray = to_color(batch_rgb, "GRAY", "RGB")
        assert batch_gray.shape == (4, 16, 16)

    def test_explicit_unsupported_conversion_raises(self) -> None:
        img = np.zeros((10, 10, 3), dtype=np.uint8)
        with pytest.raises(ValueError, match="Unsupported color_mode 'HSV'"):
            to_color(img, "HSV", "RGB")

    def test_inferred_default_target_rgb(self) -> None:
        gray = np.full((20, 30), 100, dtype=np.uint8)
        rgb = to_color(gray)
        assert rgb.shape == (20, 30, 3)
        assert np.all(rgb == 100)

    def test_inferred_2d_gray_to_rgb(self) -> None:
        gray = np.full((20, 30), 100, dtype=np.uint8)
        rgb = to_color(gray, "RGB")
        assert rgb.shape == (20, 30, 3)
        assert np.all(rgb == 100)

    def test_inferred_2d_gray_to_gray(self) -> None:
        gray = np.full((20, 30), 100, dtype=np.uint8)
        out = to_color(gray, "GRAY")
        assert out.shape == (20, 30)
        assert np.array_equal(out, gray)

    def test_inferred_3d_single_channel_to_rgb(self) -> None:
        gray_1ch = np.full((20, 30, 1), 150, dtype=np.uint8)
        rgb = to_color(gray_1ch, "RGB")
        assert rgb.shape == (20, 30, 3)
        assert np.all(rgb == 150)

    def test_inferred_3d_single_channel_to_gray(self) -> None:
        gray_1ch = np.full((20, 30, 1), 150, dtype=np.uint8)
        out = to_color(gray_1ch, "GRAY")
        assert out.shape == (20, 30)
        assert np.all(out == 150)

    def test_inferred_3d_rgb_to_rgb(self) -> None:
        rgb = np.full((20, 30, 3), 200, dtype=np.uint8)
        out = to_color(rgb, "RGB")
        assert out.shape == (20, 30, 3)
        assert np.array_equal(out, rgb)

    def test_inferred_3d_rgb_to_gray(self) -> None:
        rgb = np.zeros((20, 30, 3), dtype=np.uint8)
        rgb[:, :, 0] = 255
        gray = to_color(rgb, "GRAY")
        assert gray.shape == (20, 30)
        assert gray.dtype == np.uint8

    def test_inferred_3d_rgba_to_rgb(self) -> None:
        rgba = np.zeros((20, 30, 4), dtype=np.uint8)
        rgba[:, :, 0] = 255
        rgba[:, :, 3] = 255
        rgb = to_color(rgba, "RGB")
        assert rgb.shape == (20, 30, 3)
        assert rgb[0, 0, 0] == 255

    def test_inferred_3d_rgba_to_gray(self) -> None:
        rgba = np.full((20, 30, 4), 180, dtype=np.uint8)
        gray = to_color(rgba, "GRAY")
        assert gray.shape == (20, 30)

    def test_non_numpy_input_raises(self) -> None:
        with pytest.raises(TypeError, match="Expected np.ndarray"):
            to_color("not_an_array")  # type: ignore[arg-type]

    def test_invalid_ndim_raises(self) -> None:
        with pytest.raises(ValueError, match="Expected 2D or 3D image array"):
            to_color(np.zeros((10,), dtype=np.uint8))

        with pytest.raises(ValueError, match="Expected 2D or 3D image array"):
            to_color(np.zeros((1, 2, 3, 4), dtype=np.uint8))

    def test_unsupported_channels_raises(self) -> None:
        with pytest.raises(ValueError, match="Unsupported number of image channels: 5"):
            to_color(np.zeros((20, 30, 5), dtype=np.uint8))

