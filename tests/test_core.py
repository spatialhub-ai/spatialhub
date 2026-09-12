"""Comprehensive unit tests for spatialhub.core.runtime module."""

from __future__ import annotations

import logging
from pathlib import Path
from unittest.mock import MagicMock, patch
import onnxruntime as ort
import pytest

from spatialhub.core.runtime import create_ort_session, resolve_model_path


# ============================================================================
# 1. Tests for resolve_model_path
# ============================================================================


class TestResolveModelPath:
    """Test suite covering local resolution, HF hub downloads, sidecars, and fallbacks."""

    def test_resolve_existing_local_path_obj(self, tmp_path: Path):
        fake_onnx = tmp_path / "model.onnx"
        fake_onnx.write_bytes(b"dummy onnx content")

        resolved = resolve_model_path(model_path=fake_onnx)
        assert resolved == fake_onnx
        assert resolved.exists()

    def test_resolve_existing_local_str_path(self, tmp_path: Path):
        fake_onnx = tmp_path / "model.onnx"
        fake_onnx.write_bytes(b"dummy onnx content")

        resolved = resolve_model_path(model_path=str(fake_onnx))
        assert resolved == fake_onnx
        assert isinstance(resolved, Path)

    def test_resolve_missing_local_and_no_hf_raises(self):
        with pytest.raises(FileNotFoundError, match="Model file not found locally"):
            resolve_model_path(model_path="non_existent_file.onnx")

    def test_resolve_none_local_and_no_hf_raises(self):
        with pytest.raises(FileNotFoundError, match="Model file not found locally"):
            resolve_model_path(model_path=None)

    @pytest.mark.parametrize(
        ("repo_id", "filename"),
        [
            (None, "model.onnx"),
            ("spatialhub/weights", None),
            ("", "model.onnx"),
            ("spatialhub/weights", ""),
        ],
    )
    def test_resolve_incomplete_hf_coordinates_raises(self, repo_id: str | None, filename: str | None):
        with pytest.raises(FileNotFoundError, match="no Hugging Face repository/filename was provided"):
            resolve_model_path(model_path=None, repo_id=repo_id, filename=filename)

    def test_resolve_hf_download_success(self, tmp_path: Path):
        expected_path = tmp_path / "downloaded_model.onnx"
        expected_path.write_bytes(b"downloaded bytes")

        with patch("spatialhub.core.runtime.hf_hub_download", return_value=str(expected_path)) as mock_download:
            resolved = resolve_model_path(
                model_path=None,
                repo_id="spatialhub/demo-model",
                filename="demo.onnx",
            )
            assert resolved == expected_path
            mock_download.assert_called_once_with(repo_id="spatialhub/demo-model", filename="demo.onnx")

    def test_resolve_missing_local_falls_back_to_hf_download(self, tmp_path: Path, caplog: pytest.LogCaptureFixture):
        expected_path = tmp_path / "downloaded_model.onnx"
        expected_path.write_bytes(b"downloaded bytes")

        with caplog.at_level(logging.INFO), patch(
            "spatialhub.core.runtime.hf_hub_download", return_value=str(expected_path)
        ) as mock_download:
            resolved = resolve_model_path(
                model_path="non_existent_path.onnx",
                repo_id="spatialhub/demo-model",
                filename="demo.onnx",
            )
            assert resolved == expected_path
            mock_download.assert_called_once_with(repo_id="spatialhub/demo-model", filename="demo.onnx")
            assert "Provided model_path 'non_existent_path.onnx' not found locally" in caplog.text

    def test_resolve_hf_download_with_sidecar_data_success(self, tmp_path: Path):
        model_file = tmp_path / "model.onnx"
        model_file.write_bytes(b"model data")
        sidecar_file = tmp_path / "model.onnx.data"
        sidecar_file.write_bytes(b"sidecar weights data")

        def side_effect(repo_id: str, filename: str):
            if filename == "model.onnx":
                return str(model_file)
            elif filename == "model.onnx.data":
                return str(sidecar_file)
            raise ValueError("Unknown filename")

        with patch("spatialhub.core.runtime.hf_hub_download", side_effect=side_effect) as mock_download:
            resolved = resolve_model_path(
                repo_id="spatialhub/large-model",
                filename="model.onnx",
                download_sidecar_data=True,
            )
            assert resolved == model_file
            assert mock_download.call_count == 2
            mock_download.assert_any_call(repo_id="spatialhub/large-model", filename="model.onnx")
            mock_download.assert_any_call(repo_id="spatialhub/large-model", filename="model.onnx.data")

    def test_resolve_hf_download_with_sidecar_data_missing_ignored(self, tmp_path: Path):
        model_file = tmp_path / "model.onnx"
        model_file.write_bytes(b"model data")

        def side_effect(repo_id: str, filename: str):
            if filename == "model.onnx":
                return str(model_file)
            # Sidecar doesn't exist on HF Hub
            raise RuntimeError("404 Not Found")

        with patch("spatialhub.core.runtime.hf_hub_download", side_effect=side_effect) as mock_download:
            resolved = resolve_model_path(
                repo_id="spatialhub/self-contained-model",
                filename="model.onnx",
                download_sidecar_data=True,
            )
            assert resolved == model_file
            assert mock_download.call_count == 2

    def test_resolve_hf_download_failure_raises_runtime_error(self):
        with patch("spatialhub.core.runtime.hf_hub_download", side_effect=ConnectionError("Network unreachable")):
            with pytest.raises(RuntimeError, match="Failed to download 'fail.onnx' from repository 'broken/repo'"):
                resolve_model_path(repo_id="broken/repo", filename="fail.onnx")


# ============================================================================
# 2. Tests for create_ort_session
# ============================================================================


class TestCreateOrtSession:
    """Test suite covering ONNX runtime session initialization, provider normalization, and fallback logging."""

    def test_create_ort_session_missing_file_raises(self):
        with pytest.raises(FileNotFoundError, match="Cannot initialize ONNX session. File does not exist"):
            create_ort_session(model_path="non_existent_model.onnx")

    def test_create_ort_session_default_provider(self, tmp_path: Path):
        fake_model = tmp_path / "test.onnx"
        fake_model.write_bytes(b"dummy")

        mock_session = MagicMock(spec=ort.InferenceSession)
        mock_session.get_providers.return_value = ["CPUExecutionProvider"]

        with patch("onnxruntime.InferenceSession", return_value=mock_session) as mock_ort:
            session = create_ort_session(fake_model)
            assert session is mock_session
            mock_ort.assert_called_once()
            _, kwargs = mock_ort.call_args
            assert kwargs["providers"] == ["CPUExecutionProvider"]
            assert isinstance(kwargs["sess_options"], ort.SessionOptions)

    def test_create_ort_session_single_str_provider_normalized(self, tmp_path: Path):
        fake_model = tmp_path / "test.onnx"
        fake_model.write_bytes(b"dummy")

        mock_session = MagicMock(spec=ort.InferenceSession)
        mock_session.get_providers.return_value = ["CPUExecutionProvider"]

        with patch("onnxruntime.InferenceSession", return_value=mock_session) as mock_ort:
            create_ort_session(fake_model, providers="CPUExecutionProvider")
            _, kwargs = mock_ort.call_args
            assert kwargs["providers"] == ["CPUExecutionProvider"]

    def test_create_ort_session_list_providers(self, tmp_path: Path):
        fake_model = tmp_path / "test.onnx"
        fake_model.write_bytes(b"dummy")

        mock_session = MagicMock(spec=ort.InferenceSession)
        mock_session.get_providers.return_value = ["CUDAExecutionProvider", "CPUExecutionProvider"]

        with patch("onnxruntime.InferenceSession", return_value=mock_session) as mock_ort:
            create_ort_session(fake_model, providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
            _, kwargs = mock_ort.call_args
            assert kwargs["providers"] == ["CUDAExecutionProvider", "CPUExecutionProvider"]

    def test_create_ort_session_with_log_severity_level(self, tmp_path: Path):
        fake_model = tmp_path / "test.onnx"
        fake_model.write_bytes(b"dummy")

        mock_session = MagicMock(spec=ort.InferenceSession)
        mock_session.get_providers.return_value = ["CPUExecutionProvider"]

        with patch("onnxruntime.InferenceSession", return_value=mock_session) as mock_ort:
            create_ort_session(fake_model, log_severity_level=3)
            _, kwargs = mock_ort.call_args
            assert kwargs["sess_options"].log_severity_level == 3

    def test_create_ort_session_custom_session_options(self, tmp_path: Path):
        fake_model = tmp_path / "test.onnx"
        fake_model.write_bytes(b"dummy")

        custom_options = ort.SessionOptions()
        custom_options.intra_op_num_threads = 4

        mock_session = MagicMock(spec=ort.InferenceSession)
        mock_session.get_providers.return_value = ["CPUExecutionProvider"]

        with patch("onnxruntime.InferenceSession", return_value=mock_session) as mock_ort:
            create_ort_session(fake_model, session_options=custom_options)
            _, kwargs = mock_ort.call_args
            assert kwargs["sess_options"] is custom_options
            assert kwargs["sess_options"].intra_op_num_threads == 4

    def test_create_ort_session_provider_fallback_warning(self, tmp_path: Path, caplog: pytest.LogCaptureFixture):
        fake_model = tmp_path / "test.onnx"
        fake_model.write_bytes(b"dummy")

        # Requested CUDA, but ORT fell back to CPU
        mock_session = MagicMock(spec=ort.InferenceSession)
        mock_session.get_providers.return_value = ["CPUExecutionProvider"]

        with caplog.at_level(logging.WARNING), patch("onnxruntime.InferenceSession", return_value=mock_session):
            create_ort_session(fake_model, providers="CUDAExecutionProvider")
            assert "Requested provider 'CUDAExecutionProvider', but ONNX Runtime fell back to 'CPUExecutionProvider'" in caplog.text

    def test_create_ort_session_matching_provider_debug(self, tmp_path: Path, caplog: pytest.LogCaptureFixture):
        fake_model = tmp_path / "test.onnx"
        fake_model.write_bytes(b"dummy")

        # Requested CUDA and got CUDA
        mock_session = MagicMock(spec=ort.InferenceSession)
        mock_session.get_providers.return_value = ["CUDAExecutionProvider", "CPUExecutionProvider"]

        with caplog.at_level(logging.DEBUG), patch("onnxruntime.InferenceSession", return_value=mock_session):
            create_ort_session(fake_model, providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
            assert "ONNX session initialized on provider: CUDAExecutionProvider" in caplog.text

