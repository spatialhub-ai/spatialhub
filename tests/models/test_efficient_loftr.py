"""Integration and unit tests for EfficientLoFTRAdapter."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from spatialhub import EfficientLoFTR
from spatialhub.models.efficient_loftr.adapter import EfficientLoFTRAdapter
from spatialhub.structures import MatchResult


class TestEfficientLoFTRInit:
    """Test suite for EfficientLoFTR adapter initialization and lifecycle management."""

    def test_init_invalid_variant_raises(self):
        """Test exception when unsupported model_type variant is passed."""
        with pytest.raises(ValueError, match="model_type must be either 'full' or 'opt'"):
            EfficientLoFTR(model_type="invalid_variant")

    def test_init_missing_local_file_raises(self):
        """Test exception handling when explicit model_path does not exist locally."""
        with pytest.raises(FileNotFoundError, match="Model file not found locally"):
            EfficientLoFTR(model_path="non_existent_eloftr.onnx")

    @pytest.mark.parametrize("variant, expected_file", [
        ("full", "eloftr_outdoor_full.onnx"),
        ("opt", "eloftr_outdoor_opt.onnx"),
    ])
    def test_init_resolves_correct_variant_filename(self, variant: str, expected_file: str):
        """Test that model variant resolves the expected ONNX filename and passes providers."""
        with patch("spatialhub.models.efficient_loftr.adapter.resolve_model_path") as mock_resolve, \
             patch("spatialhub.models.efficient_loftr.adapter.create_ort_session") as mock_session:

            fake_path = Path(f"/fake/models/{expected_file}")
            mock_resolve.return_value = fake_path
            mock_session_instance = MagicMock()
            mock_input_a = MagicMock()
            mock_input_a.name = "image0"
            mock_input_b = MagicMock()
            mock_input_b.name = "image1"
            mock_session_instance.get_inputs.return_value = [mock_input_a, mock_input_b]
            mock_session.return_value = mock_session_instance

            matcher = EfficientLoFTR(model_type=variant, providers=["CPUExecutionProvider"])

            mock_resolve.assert_called_once_with(
                model_path=None,
                repo_id="SpatialHub/efficient-loftr-onnx",
                filename=expected_file,
            )
            mock_session.assert_called_once_with(
                model_path=fake_path,
                providers=["CPUExecutionProvider"],
            )
            assert matcher.input_names == ["image0", "image1"]

    def test_context_manager_and_close(self):
        """Test context manager exit and close method release session resource."""
        adapter = EfficientLoFTRAdapter.__new__(EfficientLoFTRAdapter)
        adapter.session = MagicMock()

        with adapter as ctx:
            assert ctx.session is not None

        assert adapter.session is None


class TestEfficientLoFTRPreprocess:
    """Test suite for image preprocessing, dimension alignment, and scaling."""

    @pytest.fixture
    def adapter_instance(self) -> EfficientLoFTRAdapter:
        return EfficientLoFTRAdapter.__new__(EfficientLoFTRAdapter)

    def test_preprocess_rgb_image(self, adapter_instance: EfficientLoFTRAdapter, sample_rgb_image: np.ndarray):
        """Test preprocessing an RGB uint8 array."""
        tensor, scale = adapter_instance._preprocess(sample_rgb_image, max_dim=1024)

        assert isinstance(tensor, np.ndarray)
        assert tensor.ndim == 4
        assert tensor.shape[0] == 1 and tensor.shape[1] == 1
        assert tensor.dtype == np.float32
        assert 0.0 <= tensor.min() and tensor.max() <= 1.0
        assert isinstance(scale, np.ndarray)
        assert scale.shape == (2,)

    def test_preprocess_grayscale_image(self, adapter_instance: EfficientLoFTRAdapter, sample_grayscale_image: np.ndarray):
        """Test preprocessing a 2D grayscale uint8 array."""
        tensor, scale = adapter_instance._preprocess(sample_grayscale_image, max_dim=1024)

        assert tensor.ndim == 4
        assert tensor.shape[0] == 1 and tensor.shape[1] == 1
        assert tensor.shape[2:] == (480, 640)
        assert np.allclose(scale, [1.0, 1.0])

    def test_preprocess_rgba_image(self, adapter_instance: EfficientLoFTRAdapter):
        """Test preprocessing an RGBA 4-channel image array."""
        rgba = np.zeros((256, 320, 4), dtype=np.uint8)
        tensor, scale = adapter_instance._preprocess(rgba, max_dim=1024)

        assert tensor.shape[2:] == (256, 320)
        assert np.allclose(scale, [1.0, 1.0])

    def test_preprocess_path_input(self, adapter_instance: EfficientLoFTRAdapter, sample_image_file: str):
        """Test preprocessing from file path string and Path object."""
        tensor_str, scale_str = adapter_instance._preprocess(sample_image_file, max_dim=1024)
        tensor_path, scale_path = adapter_instance._preprocess(Path(sample_image_file), max_dim=1024)

        assert np.array_equal(tensor_str, tensor_path)
        assert np.array_equal(scale_str, scale_path)

    def test_preprocess_dimensions_divisible_by_32(self, adapter_instance: EfficientLoFTRAdapter):
        """Test spatial dimension alignment to multiples of 32 for arbitrary input shapes."""
        odd_image = np.zeros((477, 631, 3), dtype=np.uint8)
        tensor, scale = adapter_instance._preprocess(odd_image, max_dim=1024)

        h, w = tensor.shape[2:]
        assert h % 32 == 0
        assert w % 32 == 0
        assert h == 448
        assert w == 608
        assert np.isclose(scale[0], 631 / 608)
        assert np.isclose(scale[1], 477 / 448)

    def test_preprocess_small_dimensions_clamped_to_32(self, adapter_instance: EfficientLoFTRAdapter):
        """Test that inputs smaller than 32px are clamped to minimum 32x32."""
        tiny_image = np.zeros((15, 20, 3), dtype=np.uint8)
        tensor, scale = adapter_instance._preprocess(tiny_image, max_dim=1024)

        assert tensor.shape[2:] == (32, 32)
        assert np.isclose(scale[0], 20 / 32)
        assert np.isclose(scale[1], 15 / 32)

    def test_preprocess_max_dim_downscaling(self, adapter_instance: EfficientLoFTRAdapter):
        """Test downscaling when image maximum dimension exceeds max_dim."""
        large_image = np.zeros((1200, 1600, 3), dtype=np.uint8)
        tensor, scale = adapter_instance._preprocess(large_image, max_dim=800)

        h, w = tensor.shape[2:]
        assert h % 32 == 0
        assert w % 32 == 0
        assert max(h, w) <= 800
        assert np.isclose(w * scale[0], 1600.0, atol=1.0)
        assert np.isclose(h * scale[1], 1200.0, atol=1.0)

    def test_preprocess_max_dim_none(self, adapter_instance: EfficientLoFTRAdapter):
        """Test preprocessing without max_dim constraint."""
        img = np.zeros((640, 960, 3), dtype=np.uint8)
        tensor, scale = adapter_instance._preprocess(img, max_dim=None)

        assert tensor.shape[2:] == (640, 960)
        assert np.allclose(scale, [1.0, 1.0])


class TestEfficientLoFTRMatching:
    """Test suite for feature matching inference logic, coordinate unscaling, and padding."""

    @pytest.fixture
    def mocked_adapter(self) -> EfficientLoFTRAdapter:
        adapter = EfficientLoFTRAdapter.__new__(EfficientLoFTRAdapter)
        adapter.session = MagicMock()
        adapter.input_names = ["image0", "image1"]
        return adapter

    def test_match_mocked_session_success(self, mocked_adapter: EfficientLoFTRAdapter):
        """Test successful feature matching with coordinate unscaling."""
        img_a = np.zeros((480, 640, 3), dtype=np.uint8)
        img_b = np.zeros((480, 640, 3), dtype=np.uint8)

        mkpts0_raw = np.array([[100.0, 150.0], [200.0, 250.0]], dtype=np.float32)
        mkpts1_raw = np.array([[105.0, 155.0], [205.0, 255.0]], dtype=np.float32)
        mconf = np.array([0.95, 0.88], dtype=np.float32)

        mocked_adapter.session.run.return_value = [mkpts0_raw, mkpts1_raw, mconf]

        result = mocked_adapter.match(img_a, img_b, max_dim=1024)

        assert isinstance(result, MatchResult)
        assert np.array_equal(result.keypoints_a, mkpts0_raw)
        assert np.array_equal(result.keypoints_b, mkpts1_raw)
        assert np.array_equal(result.confidence, mconf)

    def test_match_mocked_zero_matches(self, mocked_adapter: EfficientLoFTRAdapter):
        """Test that zero-detection case safely returns MatchResult with empty float32 arrays."""
        img_a = np.zeros((480, 640, 3), dtype=np.uint8)
        img_b = np.zeros((480, 640, 3), dtype=np.uint8)

        mocked_adapter.session.run.return_value = [
            np.empty((0, 2), dtype=np.float32),
            np.empty((0, 2), dtype=np.float32),
            np.empty((0,), dtype=np.float32),
        ]

        result = mocked_adapter.match(img_a, img_b, max_dim=1024)

        assert isinstance(result, MatchResult)
        assert result.keypoints_a.shape == (0, 2)
        assert result.keypoints_b.shape == (0, 2)
        assert result.confidence.shape == (0,)
        assert result.keypoints_a.dtype == np.float32
        assert result.keypoints_b.dtype == np.float32
        assert result.confidence.dtype == np.float32

    def test_match_mocked_padding_boundary_filtering(self, mocked_adapter: EfficientLoFTRAdapter):
        """Test that keypoints detected inside zero-padded border regions are filtered out."""
        img_a = np.zeros((320, 320, 3), dtype=np.uint8)
        img_b = np.zeros((320, 480, 3), dtype=np.uint8)

        mkpts0_raw = np.array([[100.0, 100.0], [350.0, 100.0]], dtype=np.float32)
        mkpts1_raw = np.array([[150.0, 100.0], [350.0, 100.0]], dtype=np.float32)
        mconf = np.array([0.9, 0.85], dtype=np.float32)

        mocked_adapter.session.run.return_value = [mkpts0_raw, mkpts1_raw, mconf]

        result = mocked_adapter.match(img_a, img_b, max_dim=1024)

        assert len(result.keypoints_a) == 1
        assert np.array_equal(result.keypoints_a[0], [100.0, 100.0])
        assert np.array_equal(result.keypoints_b[0], [150.0, 100.0])
        assert result.confidence[0] == 0.9


class TestEfficientLoFTRIntegration:
    """Integration test suite for EfficientLoFTR end-to-end pipeline."""

    @pytest.mark.integration
    def test_efficient_loftr_match_file_paths(self, sample_pair_image_files: tuple[str, str]):
        """Test feature matching pipeline on image pair from file paths."""
        path_a, path_b = sample_pair_image_files
        try:
            matcher = EfficientLoFTR(model_type="opt")
        except Exception as err:
            pytest.skip(f"Skipping EfficientLoFTR integration test due to missing model weights: {err}")

        result = matcher.match(path_a, path_b, max_dim=1024)

        assert isinstance(result, MatchResult)
        assert isinstance(result.keypoints_a, np.ndarray)
        assert isinstance(result.keypoints_b, np.ndarray)
        assert isinstance(result.confidence, np.ndarray)
        assert result.keypoints_a.ndim == 2 and result.keypoints_a.shape[1] == 2
        assert result.keypoints_b.ndim == 2 and result.keypoints_b.shape[1] == 2
        assert result.confidence.ndim == 1
        assert len(result.keypoints_a) == len(result.keypoints_b) == len(result.confidence)

    @pytest.mark.integration
    def test_efficient_loftr_match_numpy_arrays(self, sample_rgb_image: np.ndarray):
        """Test feature matching pipeline directly with NumPy array inputs."""
        img_a = sample_rgb_image
        img_b = np.rot90(sample_rgb_image, 1).copy()
        try:
            matcher = EfficientLoFTR(model_type="opt")
        except Exception as err:
            pytest.skip(f"Skipping EfficientLoFTR integration test due to missing model weights: {err}")

        result = matcher.match(img_a, img_b, max_dim=640)

        assert isinstance(result, MatchResult)
        assert isinstance(result.keypoints_a, np.ndarray)
        assert isinstance(result.keypoints_b, np.ndarray)
        assert isinstance(result.confidence, np.ndarray)
        assert len(result.keypoints_a) == len(result.keypoints_b) == len(result.confidence)
