"""Integration and unit tests for DepthAnything3Adapter."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from spatialhub import DepthAnything3
from spatialhub.models.depth_anything_3.adapter import DepthAnything3Adapter
from spatialhub.structures import DepthPredictionResult


class TestDepthAnything3Init:
    """Test suite for DepthAnything3 adapter initialization and parameter validation."""

    def test_init_invalid_model_name_type_raises(self):
        """Test exception when an invalid model_name type is passed."""
        with pytest.raises(TypeError, match="model_name must be a string preset"):
            DepthAnything3(model_name=12345)

    def test_init_unsupported_model_name_raises(self):
        """Test exception when an unsupported model_name string is passed."""
        with pytest.raises(ValueError, match="Unsupported model_name 'unsupported_model'"):
            DepthAnything3(model_name="unsupported_model")

    @pytest.mark.parametrize("invalid_res", [-1, 0, 500, 505, 100])
    def test_init_invalid_process_res_raises(self, invalid_res: int):
        """Test exception when process_res is non-positive or not divisible by 14."""
        with pytest.raises(ValueError, match="process_res must be a positive multiple of 14"):
            DepthAnything3(process_res=invalid_res)

    def test_init_invalid_model_variant_raises(self):
        """Test exception when unsupported model_variant is provided."""
        with pytest.raises(ValueError, match="model_variant must be 'relative' or 'metric'"):
            DepthAnything3(model_variant="unsupported")

    @pytest.mark.parametrize("model_input, expected_variant, expected_files", [
        ("da3_base", "relative", ["da3_base.onnx"]),
        ("da3_small", "relative", ["da3_small.onnx"]),
        ("da3_large", "relative", ["da3_large.onnx"]),
        ("da3_giant", "relative", ["da3_giant.onnx"]),
        ("da3mono_large", "relative", ["da3mono_large.onnx"]),
        ("da3metric_large", "metric", ["da3metric_large.onnx"]),
        ("da3nested_small_large", "metric", ["da3_small.onnx", "da3metric_large.onnx"]),
        ("da3nested_base_large", "metric", ["da3_base.onnx", "da3metric_large.onnx"]),
        ("da3nested_large_large", "metric", ["da3_large.onnx", "da3metric_large.onnx"]),
        ("da3nested_giant_large", "metric", ["da3_giant.onnx", "da3metric_large.onnx"]),
    ])
    def test_init_resolves_correct_model_and_files(
        self,
        model_input: str,
        expected_variant: str,
        expected_files: list[str],
    ):
        """Test that adapter correctly infers variant and maps model names to ONNX files."""
        with patch("spatialhub.models.depth_anything_3.adapter.resolve_model_path") as mock_resolve, \
             patch("spatialhub.models.depth_anything_3.adapter.create_ort_session") as mock_session:

            mock_resolve.side_effect = lambda **kwargs: Path(f"/fake/cache/{kwargs['filename']}")
            mock_session.return_value = MagicMock()

            adapter = DepthAnything3(
                model_name=model_input,
                process_res=504,
                process_res_method="upper_bound_resize",
            )

            assert adapter.model_variant == expected_variant
            assert adapter.process_res == 504
            assert adapter.process_res_method == "upper_bound_resize"
            assert len(adapter.ort_sessions) == len(expected_files)

            for expected_file in expected_files:
                mock_resolve.assert_any_call(
                    model_path=None,
                    repo_id="SpatialHub/depth-anything-3-onnx",
                    filename=expected_file,
                )

    def test_custom_process_res_and_method(self):
        """Test initializing adapter with custom process_res and process_res_method."""
        with patch("spatialhub.models.depth_anything_3.adapter.resolve_model_path") as mock_resolve, \
             patch("spatialhub.models.depth_anything_3.adapter.create_ort_session") as mock_session:

            mock_resolve.return_value = Path("/fake/cache/da3_base.onnx")
            mock_session.return_value = MagicMock()

            adapter = DepthAnything3(
                model_name="da3_base",
                process_res=392,
                process_res_method="lower_bound_crop",
            )

            assert adapter.process_res == 392
            assert adapter.process_res_method == "lower_bound_crop"


class TestDepthAnything3Inference:
    """Test suite for DepthAnything3 forward inference and result structuring."""

    @patch("spatialhub.models.depth_anything_3.adapter.resolve_model_path")
    @patch("spatialhub.models.depth_anything_3.adapter.create_ort_session")
    def test_estimate_depth_single_image(self, mock_session, mock_resolve):
        """Test end-to-end forward call on dummy image array."""
        mock_resolve.return_value = Path("/fake/cache/da3_base.onnx")

        mock_session_inst = MagicMock()
        mock_output_depth = MagicMock()
        mock_output_depth.name = "depth"
        mock_output_conf = MagicMock()
        mock_output_conf.name = "depth_conf"
        mock_output_sky = MagicMock()
        mock_output_sky.name = "sky"
        mock_output_ext = MagicMock()
        mock_output_ext.name = "extrinsics_out"
        mock_output_int = MagicMock()
        mock_output_int.name = "intrinsics_out"

        mock_session_inst.get_outputs.return_value = [
            mock_output_depth,
            mock_output_conf,
            mock_output_sky,
            mock_output_ext,
            mock_output_int,
        ]

        # Return shapes for batch B=1, N=1
        dummy_depth = np.ones((1, 1, 504, 504), dtype=np.float32)
        dummy_conf = np.ones((1, 1, 504, 504), dtype=np.float32)
        dummy_sky = np.zeros((1, 1, 504, 504), dtype=np.float32)
        dummy_ext = np.eye(4, dtype=np.float32).reshape(1, 1, 4, 4)
        dummy_int = np.eye(3, dtype=np.float32).reshape(1, 1, 3, 3)

        mock_session_inst.run.return_value = [
            dummy_depth,
            dummy_conf,
            dummy_sky,
            dummy_ext,
            dummy_int,
        ]
        mock_session.return_value = mock_session_inst

        adapter = DepthAnything3(model_name="da3_base")
        dummy_img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

        result = adapter.estimate_depth(images=[dummy_img])

        assert isinstance(result, DepthPredictionResult)
        assert result.depth is not None
        assert result.conf is not None
        assert result.depth_type == "relative"

    def test_context_manager_and_close(self):
        """Test context manager lifecycle and close method releasing session resources."""
        adapter = DepthAnything3Adapter.__new__(DepthAnything3Adapter)
        adapter.ort_sessions = [MagicMock()]

        with adapter as ctx:
            assert len(ctx.ort_sessions) == 1

        assert len(adapter.ort_sessions) == 0

    def test_estimate_depth_closed_session_raises(self):
        """Test exception when attempting inference after closing session resources."""
        adapter = DepthAnything3Adapter.__new__(DepthAnything3Adapter)
        adapter.ort_sessions = []
        with pytest.raises(RuntimeError, match="Inference session has been closed"):
            adapter.estimate_depth(images=[np.zeros((480, 640, 3), dtype=np.uint8)])
