"""Comprehensive unit tests for spatialhub.core.runtime module."""

from __future__ import annotations

import logging
from pathlib import Path
from unittest.mock import MagicMock, patch
import onnxruntime as ort
import pytest

from spatialhub.core.runtime import (
    IOBindingRunner,
    create_ort_session,
    resolve_execution_providers,
    resolve_model_path,
)


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
        with pytest.raises(FileNotFoundError, match="repository parameters were not provided"):
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

    def test_resolve_hf_download_with_external_data_success(self, tmp_path: Path):
        model_file = tmp_path / "model.onnx"
        model_file.write_bytes(b"model data")
        sidecar_file = tmp_path / "model.onnx.data"
        sidecar_file.write_bytes(b"sidecar weights data")

        def side_effect(repo_id: str, filename: str):
            if filename == "model.onnx":
                return str(model_file)
            elif filename == "model.onnx.data":
                return str(sidecar_file)
            raise ValueError(f"Unknown filename: {filename}")

        with patch(
            "spatialhub.core.runtime._get_external_data_filenames", return_value=["model.onnx.data"]
        ), patch("spatialhub.core.runtime.hf_hub_download", side_effect=side_effect) as mock_download:
            resolved = resolve_model_path(
                repo_id="spatialhub/large-model",
                filename="model.onnx",
            )
            assert resolved == model_file
            assert mock_download.call_count == 2
            mock_download.assert_any_call(repo_id="spatialhub/large-model", filename="model.onnx")
            mock_download.assert_any_call(repo_id="spatialhub/large-model", filename="model.onnx.data")

    def test_resolve_hf_download_self_contained_model_no_extra_calls(self, tmp_path: Path):
        model_file = tmp_path / "model.onnx"
        model_file.write_bytes(b"model data")

        with patch(
            "spatialhub.core.runtime._get_external_data_filenames", return_value=[]
        ), patch("spatialhub.core.runtime.hf_hub_download", return_value=str(model_file)) as mock_download:
            resolved = resolve_model_path(
                repo_id="spatialhub/self-contained-model",
                filename="model.onnx",
            )
            assert resolved == model_file
            mock_download.assert_called_once_with(repo_id="spatialhub/self-contained-model", filename="model.onnx")

    def test_resolve_hf_download_failure_raises_runtime_error(self):
        with patch("spatialhub.core.runtime.hf_hub_download", side_effect=ConnectionError("Network unreachable")):
            with pytest.raises(RuntimeError, match="Failed to download 'fail.onnx' from repository 'broken/repo'"):
                resolve_model_path(repo_id="broken/repo", filename="fail.onnx")


class TestCreateOrtSession:
    """Test suite covering ONNX runtime session initialization, provider normalization, and fallback logging."""

    def test_create_ort_session_missing_file_raises(self):
        with pytest.raises(FileNotFoundError, match="Cannot initialize session. File does not exist"):
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

        # Requested CUDA, but session fell back to CPU
        mock_session = MagicMock(spec=ort.InferenceSession)
        mock_session.get_providers.return_value = ["CPUExecutionProvider"]

        with caplog.at_level(logging.WARNING), patch("onnxruntime.InferenceSession", return_value=mock_session):
            create_ort_session(fake_model, providers="CUDAExecutionProvider")
            assert "Requested providers ['CUDAExecutionProvider'], but session initialized on 'CPUExecutionProvider'" in caplog.text

    def test_create_ort_session_matching_provider_debug(self, tmp_path: Path, caplog: pytest.LogCaptureFixture):
        fake_model = tmp_path / "test.onnx"
        fake_model.write_bytes(b"dummy")

        # Requested CUDA and got CUDA
        mock_session = MagicMock(spec=ort.InferenceSession)
        mock_session.get_providers.return_value = ["CUDAExecutionProvider", "CPUExecutionProvider"]

        with caplog.at_level(logging.DEBUG), patch("onnxruntime.InferenceSession", return_value=mock_session):
            create_ort_session(fake_model, providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
            assert "Session initialized on provider: CUDAExecutionProvider" in caplog.text


class TestLazyModuleImports:
    """Test PEP 562 dynamic lazy loading of models and utilities."""

    def test_lazy_load_known_models(self):
        import spatialhub
        from spatialhub.models.efficient_loftr.adapter import EfficientLoFTRAdapter

        assert spatialhub.EfficientLoFTR is EfficientLoFTRAdapter
        assert "EfficientLoFTR" in dir(spatialhub)

    def test_lazy_load_invalid_attribute_raises(self):
        import spatialhub

        with pytest.raises(AttributeError, match="has no attribute 'NonExistentModel'"):
            _ = spatialhub.NonExistentModel

    def test_lazy_load_models_package(self):
        import spatialhub.models as models
        from spatialhub.models.dinov2.adapter import DINOv2Adapter

        assert models.DINOv2 is DINOv2Adapter
        assert "DINOv2" in dir(models)

    def test_lazy_load_utils_package(self):
        import spatialhub.utils as utils

        assert hasattr(utils, "load_image")
        assert "AtlasRenderer" in dir(utils)


class TestResolveExecutionProviders:
    """Test suite covering hardware target resolution and provider list normalization."""

    def test_resolve_cpu_target(self):
        provs = resolve_execution_providers(providers="cpu")
        assert provs == ["CPUExecutionProvider"]

    def test_resolve_cuda_target(self):
        with patch("onnxruntime.get_available_providers", return_value=["CUDAExecutionProvider", "CPUExecutionProvider"]):
            provs = resolve_execution_providers(providers="cuda")
            assert provs == ["CUDAExecutionProvider", "CPUExecutionProvider"]

    def test_resolve_auto_target_with_cuda_available(self):
        with patch("onnxruntime.get_available_providers", return_value=["CUDAExecutionProvider", "CPUExecutionProvider"]):
            provs = resolve_execution_providers(providers="auto")
            assert provs == ["CUDAExecutionProvider", "CPUExecutionProvider"]

    def test_resolve_auto_target_without_cuda(self):
        with patch("onnxruntime.get_available_providers", return_value=["CPUExecutionProvider"]):
            provs = resolve_execution_providers(providers="auto")
            assert provs == ["CPUExecutionProvider"]

    def test_resolve_gpu_target_with_cuda_available(self):
        with patch("onnxruntime.get_available_providers", return_value=["CUDAExecutionProvider", "CPUExecutionProvider"]):
            provs = resolve_execution_providers(providers="gpu")
            assert provs == ["CUDAExecutionProvider", "CPUExecutionProvider"]

    def test_resolve_cuda_missing_raises_runtime_error(self):
        with patch("onnxruntime.get_available_providers", return_value=["CPUExecutionProvider"]):
            with pytest.raises(RuntimeError, match="CUDAExecutionProvider' is not available"):
                resolve_execution_providers(providers="cuda")

    def test_resolve_explicit_providers_override(self):
        explicit = ["CPUExecutionProvider"]
        provs = resolve_execution_providers(providers=explicit)
        assert provs == ["CPUExecutionProvider"]

    def test_resolve_invalid_provider_string_raises(self):
        with pytest.raises(ValueError, match="Unrecognized execution provider 'invalid_provider'"):
            resolve_execution_providers(providers="invalid_provider")

    def test_resolve_known_ort_provider_name_case_insensitive(self):
        provs = resolve_execution_providers(providers="cpuexecutionprovider")
        assert provs == ["CPUExecutionProvider"]

    def test_resolve_multi_accelerated_providers_priority(self):
        mock_avail = ["DmlExecutionProvider", "CUDAExecutionProvider", "TensorrtExecutionProvider", "CPUExecutionProvider"]
        with patch("onnxruntime.get_available_providers", return_value=mock_avail):
            providers = resolve_execution_providers(providers="auto")
            assert providers == [
                "TensorrtExecutionProvider",
                "CUDAExecutionProvider",
                "DmlExecutionProvider",
                "CPUExecutionProvider",
            ]

    def test_resolve_explicit_custom_vendor_providers(self):
        rocm_chain = ["ROCMExecutionProvider", "CPUExecutionProvider"]
        providers = resolve_execution_providers(providers=rocm_chain)
        assert providers == rocm_chain


def _create_mock_session(input_names: list[str], output_names: list[str], run_side_effect) -> MagicMock:
    session = MagicMock()
    session.get_providers.return_value = ["CPUExecutionProvider"]
    inputs = []
    for name in input_names:
        inp = MagicMock()
        inp.name = name
        inputs.append(inp)
    session.get_inputs.return_value = inputs

    outputs = []
    for name in output_names:
        out = MagicMock()
        out.name = name
        outputs.append(out)
    session.get_outputs.return_value = outputs

    session.run.side_effect = run_side_effect

    mock_binding = MagicMock()
    bound_inputs = {}

    def bind_cpu_input(name, arr):
        bound_inputs[name] = arr

    def bind_ortvalue_input(name, val):
        bound_inputs[name] = val

    mock_binding.bind_cpu_input.side_effect = bind_cpu_input
    mock_binding.bind_ortvalue_input.side_effect = bind_ortvalue_input

    def run_with_iobinding(binding):
        results = run_side_effect(output_names, bound_inputs)
        out_mocks = []
        for res in results:
            m = MagicMock()
            m.numpy.return_value = res
            out_mocks.append(m)
        mock_binding.get_outputs.return_value = out_mocks

    session.io_binding.return_value = mock_binding
    session.run_with_iobinding.side_effect = run_with_iobinding
    return session


class TestIOBindingMockSession:
    """Tests IOBindingRunner and IOBindingRunner binding methods with mock session."""

    def test_invalid_device_raises(self):
        mock_sess = _create_mock_session(["in_a"], ["out_a"], lambda outs, feed: [np.zeros(1)])
        with pytest.raises(ValueError, match="Invalid device 'tpu'"):
            IOBindingRunner(mock_sess, device="tpu")

    def test_device_id_resolution_from_active_provider(self):
        mock_sess = _create_mock_session(["in_a"], ["out_a"], lambda outs, feed: [np.zeros(1)])
        mock_sess.get_providers.return_value = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        mock_sess.get_provider_options.return_value = {
            "CUDAExecutionProvider": {"device_id": "3"}
        }
        runner = IOBindingRunner(mock_sess)
        assert runner.device == "cuda"
        assert runner.device_id == 3

    def test_run_with_mock_session(self):
        import numpy as np

        def mock_run(out_names, feed):
            return [feed["in_a"] + feed["in_b"]]

        mock_sess = _create_mock_session(["in_a", "in_b"], ["out_sum"], mock_run)
        wrapped = IOBindingRunner(mock_sess)

        feed = {
            "in_a": np.array([1.0, 2.0], dtype=np.float32),
            "in_b": np.array([3.0, 4.0], dtype=np.float32),
        }
        res = wrapped.run(feed)
        assert "out_sum" in res
        np.testing.assert_array_equal(res["out_sum"], np.array([4.0, 6.0], dtype=np.float32))

    def test_bind_input_and_clear_binding(self):
        import numpy as np

        def mock_run(out_names, feed):
            return [feed["in_a"] * 2.0]

        mock_sess = _create_mock_session(["in_a"], ["out_mul"], mock_run)
        wrapped = IOBindingRunner(mock_sess)

        wrapped.bind_input("in_a", np.array([5.0, 10.0], dtype=np.float32))
        wrapped.bind_output("out_mul", device="cpu")
        res = wrapped.run()
        assert "out_mul" in res
        np.testing.assert_array_equal(res["out_mul"], np.array([10.0, 20.0], dtype=np.float32))

        wrapped.clear_binding()

    def test_invalid_input_type_raises(self):
        import numpy as np

        mock_sess = _create_mock_session(["in_a"], ["out_a"], lambda outs, feed: [np.zeros(1)])
        wrapped = IOBindingRunner(mock_sess)

        with pytest.raises(TypeError, match="Unsupported input tensor type"):
            wrapped.bind_input("in_a", "invalid_string_input")

        with pytest.raises(ValueError, match="shape and dtype must be specified"):
            wrapped.bind_input("in_a", 12345678)  # missing shape/dtype for raw pointer

    def test_bind_output_buffer_validation(self):
        import numpy as np

        mock_sess = _create_mock_session(["in_a"], ["out_a"], lambda outs, feed: [np.zeros(1)])
        wrapped = IOBindingRunner(mock_sess)

        with pytest.raises(ValueError, match="shape and dtype must be specified"):
            wrapped.bind_output("out_a", buffer=12345678)
        assert len(wrapped._bound_output_names) == 0

        with pytest.raises(TypeError, match="Unsupported output buffer type"):
            wrapped.bind_output("out_a", buffer="invalid_buffer_type")
        assert len(wrapped._bound_output_names) == 0

    def test_device_validation_on_bind_methods(self):
        import numpy as np

        mock_sess = _create_mock_session(["in_a"], ["out_a"], lambda outs, feed: [np.zeros(1)])
        wrapped = IOBindingRunner(mock_sess)

        with pytest.raises(ValueError, match="Invalid device 'vulkan'"):
            wrapped.bind_input("in_a", np.zeros(1), device="vulkan")

        with pytest.raises(ValueError, match="Invalid device ''"):
            wrapped.bind_input("in_a", np.zeros(1), device="")

        with pytest.raises(ValueError, match="Invalid device 'metal'"):
            wrapped.bind_output("out_a", device="metal")

        with pytest.raises(ValueError, match="Invalid device ''"):
            wrapped.bind_output("out_a", device="")

        with pytest.raises(ValueError, match="Invalid device 'directml'"):
            wrapped.run({"in_a": np.zeros(1)}, output_device="directml")

        with pytest.raises(ValueError, match="Invalid device ''"):
            wrapped.run({"in_a": np.zeros(1)}, output_device="")

    def test_bool_type_rejection_for_memory_pointer(self):
        import numpy as np

        mock_sess = _create_mock_session(["in_a"], ["out_a"], lambda outs, feed: [np.zeros(1)])
        wrapped = IOBindingRunner(mock_sess)

        with pytest.raises(TypeError, match="Unsupported input tensor type: bool"):
            wrapped.bind_input("in_a", True, shape=(1,), dtype=np.float32)

        with pytest.raises(TypeError, match="Unsupported output buffer type: bool"):
            wrapped.bind_output("out_a", buffer=False, shape=(1,), dtype=np.float32)

    def test_sync_flag_invokes_synchronize_outputs(self):
        import numpy as np

        mock_sess = _create_mock_session(["in_a"], ["out_a"], lambda outs, feed: [np.zeros(1)])
        wrapped = IOBindingRunner(mock_sess)
        wrapped.synchronize_outputs = MagicMock()

        wrapped.run({"in_a": np.zeros(1)}, sync=True)
        wrapped.synchronize_outputs.assert_called_once()

    def test_unknown_node_name_validation(self):
        import numpy as np

        mock_sess = _create_mock_session(["in_a"], ["out_a"], lambda outs, feed: [np.zeros(1)])
        wrapped = IOBindingRunner(mock_sess)

        with pytest.raises(ValueError, match="Unknown input node 'bad_in'"):
            wrapped.bind_input("bad_in", np.zeros(1))

        with pytest.raises(ValueError, match="Unknown output node 'bad_out'"):
            wrapped.bind_output("bad_out")

    def test_arbitrary_output_binding_order_parity(self):
        import numpy as np

        def mock_run(out_names, feed):
            return [feed["in_b"], feed["in_a"]]

        mock_sess = _create_mock_session(["in_a", "in_b"], ["out_a", "out_b"], mock_run)
        wrapped = IOBindingRunner(mock_sess)

        wrapped.bind_output("out_b", device="cpu")
        wrapped.bind_output("out_a", device="cpu")
        wrapped.bind_input("in_a", np.array([1.0], dtype=np.float32))
        wrapped.bind_input("in_b", np.array([2.0], dtype=np.float32))

        out = wrapped.run()
        np.testing.assert_array_equal(out["out_b"], np.array([2.0], dtype=np.float32))
        np.testing.assert_array_equal(out["out_a"], np.array([1.0], dtype=np.float32))


class TestIOBindingRealSession:
    """Tests IOBindingRunner against real ONNX models if available."""

    @pytest.fixture
    def refine_session(self):
        from pathlib import Path
        model_path = Path("onnx_weight/refine_net.onnx")
        if not model_path.exists():
            pytest.skip("refine_net.onnx not found in onnx_weight/")

        avail = ort.get_available_providers()
        provs = ["CUDAExecutionProvider"] if "CUDAExecutionProvider" in avail else ["CPUExecutionProvider"]
        return ort.InferenceSession(str(model_path), providers=provs)

    def test_cpu_io_binding_execution(self, refine_session):
        import numpy as np

        wrapped = IOBindingRunner(refine_session)

        a = np.zeros((1, 6, 160, 160), dtype=np.float32)
        b = np.zeros((1, 6, 160, 160), dtype=np.float32)

        feed = {"input_render_A": a, "input_real_B": b}
        out = wrapped.run(feed, return_numpy=True)

        assert "trans" in out
        assert "rot" in out
        assert out["trans"].shape == (1, 3)
        assert out["rot"].shape == (1, 3)

    def test_cuda_io_binding_execution(self, refine_session):
        import numpy as np

        if "CUDAExecutionProvider" not in refine_session.get_providers():
            pytest.skip("CUDAExecutionProvider is not active for this session")

        wrapped = IOBindingRunner(refine_session, device="cuda")
        a = np.zeros((1, 6, 160, 160), dtype=np.float32)
        b = np.zeros((1, 6, 160, 160), dtype=np.float32)

        va = ort.OrtValue.ortvalue_from_numpy(a, "cuda", 0)
        vb = ort.OrtValue.ortvalue_from_numpy(b, "cuda", 0)

        wrapped.bind_input("input_render_A", va)
        wrapped.bind_input("input_real_B", vb)
        wrapped.bind_output("trans", device="cpu")
        wrapped.bind_output("rot", device="cpu")

        out = wrapped.run(return_numpy=True)
        assert "trans" in out
        assert "rot" in out
        assert out["trans"].shape == (1, 3)
        assert out["rot"].shape == (1, 3)
