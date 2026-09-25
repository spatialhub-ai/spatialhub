"""Integration and unit tests for DINOv2Adapter."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from spatialhub import DINOv2
from spatialhub.models.dinov2.adapter import DINOv2Adapter
from spatialhub.structures import FeatureExtractionResult


class TestDINOv2Init:
    """Test suite for DINOv2 adapter initialization, variant resolution, and parameter validation."""

    @pytest.mark.parametrize("invalid_size", [-1, 0, 100, 225, 500])
    def test_init_invalid_target_size_raises(self, invalid_size: int):
        """Test exception when target_size is non-positive or not a multiple of patch size 14."""
        with pytest.raises(ValueError, match="target_size must be a positive multiple of patch size 14"):
            DINOv2Adapter(target_size=invalid_size)

    def test_init_invalid_variant_raises(self):
        """Test exception when unsupported model_variant is provided."""
        with pytest.raises(ValueError, match="Unsupported model_variant 'invalid_variant'"):
            DINOv2Adapter(model_variant="invalid_variant")

    def test_init_missing_local_file_raises(self):
        """Test exception handling when explicit model_path does not exist locally."""
        with pytest.raises(FileNotFoundError, match="Model file not found locally"):
            DINOv2Adapter(model_path="non_existent_dinov2.onnx")

    @pytest.mark.parametrize(
        ("variant", "expected_file", "expected_dim"),
        [
            ("vits14", "dinov2_vits14.onnx", 384),
            ("vitb14", "dinov2_vitb14.onnx", 768),
            ("vitl14", "dinov2_vitl14.onnx", 1024),
            ("vitg14", "dinov2_vitg14.onnx", 1536),
        ],
    )
    def test_init_resolves_correct_variant_and_outputs(
        self, variant: str, expected_file: str, expected_dim: int
    ):
        """Test that model variant resolves expected ONNX filename and detects graph output metadata."""
        with patch("spatialhub.models.dinov2.adapter.resolve_model_path") as mock_resolve, \
             patch("spatialhub.models.dinov2.adapter.create_ort_session") as mock_session:

            fake_path = Path(f"/fake/models/{expected_file}")
            mock_resolve.return_value = fake_path

            mock_session_instance = MagicMock()
            mock_input = MagicMock()
            mock_input.name = "image"
            mock_session_instance.get_inputs.return_value = [mock_input]

            mock_output = MagicMock()
            mock_output.name = "output"
            mock_output.shape = [1, expected_dim]
            mock_session_instance.get_outputs.return_value = [mock_output]

            mock_session.return_value = mock_session_instance

            adapter = DINOv2(model_variant=variant, target_size=224, providers=["CPUExecutionProvider"])

            mock_resolve.assert_called_once_with(
                model_path=None,
                repo_id="SpatialHub/dinov2-onnx",
                filename=expected_file,
            )
            mock_session.assert_called_once_with(
                model_path=fake_path,
                providers=["CPUExecutionProvider"],
            )
            assert adapter.target_size == 224
            assert adapter.input_name == "image"
            assert adapter.output_name == "output"
            assert adapter.output_dim == expected_dim

    def test_context_manager_and_close(self):
        """Test context manager lifecycle and close method releasing session resources."""
        adapter = DINOv2Adapter.__new__(DINOv2Adapter)
        adapter.session = MagicMock()

        with adapter as ctx:
            assert ctx.session is not None

        assert adapter.session is None


class TestDINOv2Preprocess:
    """Test suite for DINOv2 image preprocessing, aspect-ratio padding, resizing, and normalization."""

    @pytest.fixture
    def adapter_instance(self) -> DINOv2Adapter:
        adapter = DINOv2Adapter.__new__(DINOv2Adapter)
        adapter.target_size = 224
        return adapter

    def test_preprocess_single_rgb_image(self, adapter_instance: DINOv2Adapter, sample_rgb_image: np.ndarray):
        """Test preprocessing a single (H, W, 3) uint8 image array."""
        tensor = adapter_instance._preprocess(sample_rgb_image)

        assert isinstance(tensor, np.ndarray)
        assert tensor.ndim == 4
        assert tensor.shape == (1, 3, 224, 224)
        assert tensor.dtype == np.float32

    def test_preprocess_square_padding_preserves_aspect_ratio(self, adapter_instance: DINOv2Adapter):
        """Test that rectangular image (480, 640, 3) is padded into square canvas before resizing."""
        rect_img = np.ones((480, 640, 3), dtype=np.uint8) * 128
        tensor = adapter_instance._preprocess(rect_img)

        assert tensor.shape == (1, 3, 224, 224)

    def test_preprocess_batch_4d_array(self, adapter_instance: DINOv2Adapter):
        """Test preprocessing a 4D batch array of shape (N, H, W, 3)."""
        batch_imgs = np.zeros((3, 200, 200, 3), dtype=np.uint8)
        tensor = adapter_instance._preprocess(batch_imgs)

        assert tensor.shape == (3, 3, 224, 224)
        assert tensor.dtype == np.float32

    def test_preprocess_float_scaled_inputs(self, adapter_instance: DINOv2Adapter):
        """Test preprocessing float arrays in [0.0, 1.0] without double scaling."""
        float_img = np.ones((224, 224, 3), dtype=np.float32) * 0.5
        tensor = adapter_instance._preprocess(float_img)

        assert tensor.shape == (1, 3, 224, 224)
        assert tensor.dtype == np.float32

    def test_preprocess_invalid_image_dimension_raises(self, adapter_instance: DINOv2Adapter):
        """Test exception when passing 2D grayscale or non-3-channel images."""
        gray_img = np.zeros((224, 224), dtype=np.uint8)
        with pytest.raises(ValueError, match="Expected RGB image array with shape"):
            adapter_instance._preprocess(gray_img)

    def test_preprocess_invalid_type_raises(self, adapter_instance: DINOv2Adapter):
        """Test exception when passing unsupported image container types."""
        with pytest.raises(TypeError, match="Unsupported image input type for preprocessing"):
            adapter_instance._preprocess(12345)


class TestDINOv2Inference:
    """Test suite for DINOv2 feature extraction, polymorphic inputs, and L2 normalization."""

    @pytest.fixture
    def mock_adapter(self) -> DINOv2Adapter:
        """Create mock DINOv2 adapter with simulated 1024-dim ViT-Large output."""
        adapter = DINOv2Adapter.__new__(DINOv2Adapter)
        adapter.target_size = 224
        adapter.input_name = "image"
        adapter.output_name = "output"
        adapter.output_dim = 1024

        mock_session = MagicMock()

        def mock_run(output_names, feed_dict):
            batch_size = feed_dict["image"].shape[0]
            rng = np.random.RandomState(42)
            raw_features = rng.randn(batch_size, 1024).astype(np.float32)
            return [raw_features]

        mock_session.run.side_effect = mock_run
        adapter.session = mock_session
        return adapter

    def test_extract_features_single_array(self, mock_adapter: DINOv2Adapter, sample_rgb_image: np.ndarray):
        """Test extracting features from a single RGB NumPy array."""
        result = mock_adapter.extract_features(sample_rgb_image, l2_normalize=False)

        assert isinstance(result, FeatureExtractionResult)
        assert result.features.shape == (1, 1024)
        assert result.embedding_type == "global"
        assert not result.l2_normalized
        assert result.images.shape == sample_rgb_image.shape

    def test_extract_features_single_path_string(self, mock_adapter: DINOv2Adapter, sample_image_file: str):
        """Test extracting features from a single image file path string."""
        result = mock_adapter.extract_features(sample_image_file, l2_normalize=False)

        assert isinstance(result, FeatureExtractionResult)
        assert result.features.shape == (1, 1024)
        assert result.images.ndim == 3

    def test_extract_features_single_path_obj(self, mock_adapter: DINOv2Adapter, sample_image_file: str):
        """Test extracting features from a pathlib.Path object."""
        result = mock_adapter.extract_features(Path(sample_image_file), l2_normalize=False)

        assert isinstance(result, FeatureExtractionResult)
        assert result.features.shape == (1, 1024)

    def test_extract_features_batch_4d_array(self, mock_adapter: DINOv2Adapter):
        """Test extracting features from a 4D batch array of shape (N, H, W, 3)."""
        batch_imgs = np.zeros((4, 224, 224, 3), dtype=np.uint8)
        result = mock_adapter.extract_features(batch_imgs, l2_normalize=False)

        assert result.features.shape == (4, 1024)
        assert result.images.shape == (4, 224, 224, 3)

    def test_extract_features_list_collection(self, mock_adapter: DINOv2Adapter, sample_pair_image_files: tuple[str, str]):
        """Test extracting features from a mixed collection of image paths and arrays."""
        img_a_path, img_b_path = sample_pair_image_files
        raw_img = np.zeros((100, 100, 3), dtype=np.uint8)

        inputs = [img_a_path, Path(img_b_path), raw_img]
        result = mock_adapter.extract_features(inputs, l2_normalize=False)

        assert result.features.shape == (3, 1024)
        assert len(result.images) == 3

    def test_extract_features_l2_normalization(self, mock_adapter: DINOv2Adapter, sample_rgb_image: np.ndarray):
        """Test that l2_normalize=True produces unit-length embedding vectors."""
        result = mock_adapter.extract_features(sample_rgb_image, l2_normalize=True)

        assert result.l2_normalized
        norm = np.linalg.norm(result.features, axis=-1)
        assert np.allclose(norm, 1.0, atol=1e-5)

    def test_extract_features_closed_session_raises(self, mock_adapter: DINOv2Adapter, sample_rgb_image: np.ndarray):
        """Test exception when attempting feature extraction after closing the session."""
        mock_adapter.close()
        with pytest.raises(RuntimeError, match="Inference session has been closed"):
            mock_adapter.extract_features(sample_rgb_image)

    def test_extract_features_empty_list_raises(self, mock_adapter: DINOv2Adapter):
        """Test exception when an empty collection is provided."""
        with pytest.raises(ValueError, match="Input image collection cannot be empty"):
            mock_adapter.extract_features([])
