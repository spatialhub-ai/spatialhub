from collections.abc import Sequence
import logging
import mmap
from pathlib import Path
from typing import Any

from huggingface_hub import hf_hub_download
import numpy as np

try:
    import onnxruntime as ort
except ImportError:
    raise ImportError(
        "SpatialHub requires an ONNX Runtime backend for model inference.\n"
        "Install the package for your hardware setup:\n"
        "  - CPU:                       pip install \"spatialhub[cpu]\"\n"
        "  - NVIDIA GPU (CUDA):         pip install \"spatialhub[gpu]\"\n"
        "  - CPU with 3D CAD rendering: pip install \"spatialhub[cpu,render]\"\n"
        "  - GPU with 3D CAD rendering: pip install \"spatialhub[gpu,render]\"\n\n"
        "Note: Do not install both 'onnxruntime' and 'onnxruntime-gpu' in the same environment."
    ) from None

logger = logging.getLogger(__name__)

VALID_IO_DEVICES = {"cpu", "cuda"}
SUPPORTED_PROVIDER_SHORTHANDS = {"auto", "cpu", "cuda", "gpu"}
ACCELERATED_PROVIDERS_PRIORITY = (
    "TensorrtExecutionProvider",
    "CUDAExecutionProvider",
    "ROCMExecutionProvider",
    "DmlExecutionProvider",
)


def _get_external_data_filenames(model_path: Path) -> list[str]:
    """Extract referenced external tensor data filenames from a serialized model binary.

    Uses zero-copy memory mapping to locate StringStringEntryProto descriptors where
    key is 'location'.
    """
    external_files: set[str] = set()

    try:
        if not model_path.exists() or model_path.stat().st_size == 0:
            return []

        pattern = b"\n\x08location\x12"
        pattern_len = len(pattern)

        with open(model_path, "rb") as f:
            with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
                mm_len = len(mm)
                idx = 0

                while True:
                    pos = mm.find(pattern, idx)
                    if pos == -1:
                        break

                    ptr = pos + pattern_len
                    val_len = 0
                    shift = 0
                    while ptr < mm_len:
                        byte = mm[ptr]
                        ptr += 1
                        val_len |= (byte & 0x7F) << shift
                        if not (byte & 0x80):
                            break
                        shift += 7
                        if shift > 64:
                            break

                    end_ptr = ptr + val_len
                    if 0 < val_len <= 1024 and end_ptr <= mm_len:
                        try:
                            loc_str = mm[ptr:end_ptr].decode("utf-8")
                            cleaned_path = loc_str.replace("\\", "/").strip()
                            if cleaned_path and "\x00" not in cleaned_path:
                                external_files.add(cleaned_path)
                        except UnicodeDecodeError:
                            pass

                    idx = pos + pattern_len

    except Exception as exc:
        logger.debug("Failed to inspect external data in '%s': %s", model_path, exc)

    return list(external_files)


def resolve_model_path(
                    model_path: str | Path | None = None,
                    repo_id: str | None = None,
                    filename: str | None = None,
                ) -> Path:
    """Resolve model binary path locally or retrieve from remote repository.

    Inspects serialized model files for external tensor data references and
    downloads companion files when required.

    Args:
        model_path: Local path to model binary.
        repo_id: Remote repository identifier.
        filename: Model filename in repository.

    Returns:
        Path: Local path to model binary.

    Raises:
        FileNotFoundError: If model cannot be resolved locally or repository parameters are missing.
        RuntimeError: If remote retrieval fails.
    """
    if model_path is not None:
        local_file = Path(model_path)
        if local_file.exists():
            return local_file
        logger.info("Provided model_path '%s' not found locally.", model_path)

    if not repo_id or not filename:
        raise FileNotFoundError(
            f"Model file not found locally at '{model_path}', and repository parameters were not provided."
        )

    logger.info("Fetching '%s' from repository '%s'...", filename, repo_id)
    try:
        downloaded = hf_hub_download(repo_id=repo_id, filename=filename)
        resolved_path = Path(downloaded)

        companion_files = _get_external_data_filenames(resolved_path)
        for companion in companion_files:
            try:
                hf_hub_download(repo_id=repo_id, filename=companion)
                logger.debug("Fetched companion data file: %s", companion)
            except Exception as sidecar_exc:
                logger.warning(
                    "Failed to download companion file '%s' from repository '%s': %s",
                    companion,
                    repo_id,
                    sidecar_exc,
                )

        return resolved_path

    except Exception as exc:
        raise RuntimeError(
            f"Failed to download '{filename}' from repository '{repo_id}'. Error: {exc}"
        ) from exc


def _extract_provider_name(provider: str | tuple[str, dict[str, Any]]) -> str:
    """Extract string identifier from provider specification."""
    return provider[0] if isinstance(provider, tuple) else provider


def create_ort_session(
    model_path: str | Path,
    providers: list[str | tuple[str, dict[str, Any]]] | str | None = None,
    session_options: ort.SessionOptions | None = None,
    log_severity_level: int | None = None,
) -> ort.InferenceSession:
    """Initialize inference session from local model binary.

    Args:
        model_path: Path to model binary.
        providers: Execution provider names or configurations.
        session_options: Custom session configuration options.
        log_severity_level: Session logging severity level.

    Returns:
        ort.InferenceSession: Initialized inference session.

    Raises:
        FileNotFoundError: If model binary does not exist at specified path.
    """
    path_obj = Path(model_path)
    if not path_obj.exists():
        raise FileNotFoundError(f"Cannot initialize session. File does not exist at: {path_obj}")

    if providers is None:
        execution_providers = ["CPUExecutionProvider"]
    elif isinstance(providers, (str, tuple)):
        execution_providers = [providers]
    else:
        execution_providers = list(providers)

    if session_options is None:
        session_options = ort.SessionOptions()
        if log_severity_level is not None:
            session_options.log_severity_level = log_severity_level

    session = ort.InferenceSession(
        str(path_obj),
        sess_options=session_options,
        providers=execution_providers,
    )

    active_provider = session.get_providers()[0]
    requested_names = [_extract_provider_name(p) for p in execution_providers]

    if active_provider not in requested_names:
        logger.warning("Requested providers %s, but session initialized on '%s'.", requested_names, active_provider)
    else:
        logger.debug("Session initialized on provider: %s", active_provider)

    return session


def resolve_execution_providers(
    providers: list[str | tuple[str, dict[str, Any]]] | tuple[str, dict[str, Any]] | str | None = None,
) -> list[str | tuple[str, dict[str, Any]]]:
    """Resolve ONNX Runtime execution provider configuration list.

    Translates shorthand aliases ('auto', 'cuda', 'gpu', 'cpu'), full provider names,
    or explicit provider specifications into an ordered provider list for session initialization.

    Args:
        providers: Execution provider specification. Accepts:
            - None or 'auto': Automatically select best available hardware providers with CPU fallback.
            - 'cpu': Target host CPU ('CPUExecutionProvider').
            - 'cuda': Target NVIDIA CUDA ('CUDAExecutionProvider', 'CPUExecutionProvider').
            - 'gpu': Target any accelerated GPU provider available on the host.
            - Full provider name string: e.g., 'CUDAExecutionProvider', 'CPUExecutionProvider'.
            - Explicit provider list: Sequence of provider names or (name, options) tuples.

    Returns:
        list[str | tuple[str, dict[str, Any]]]: Ordered provider list for InferenceSession initialization.

    Raises:
        ValueError: If a string provider or shorthand is unrecognized.
        RuntimeError: If 'cuda' or 'gpu' is requested but no suitable provider is available.
    """
    avail = ort.get_available_providers()

    if providers is None:
        active = [p for p in ACCELERATED_PROVIDERS_PRIORITY if p in avail]
        return [*active, "CPUExecutionProvider"] if active else ["CPUExecutionProvider"]

    if isinstance(providers, str):
        target = providers.lower().strip()
        if target in ("auto", "default"):
            active = [p for p in ACCELERATED_PROVIDERS_PRIORITY if p in avail]
            return [*active, "CPUExecutionProvider"] if active else ["CPUExecutionProvider"]
        if target == "cpu":
            return ["CPUExecutionProvider"]
        if target == "cuda":
            if "CUDAExecutionProvider" not in avail:
                raise RuntimeError(
                    f"CUDA execution was requested, but 'CUDAExecutionProvider' is not available. "
                    f"Available providers: {avail}"
                )
            return ["CUDAExecutionProvider", "CPUExecutionProvider"]
        if target == "gpu":
            active = [p for p in ACCELERATED_PROVIDERS_PRIORITY if p in avail]
            if not active:
                raise RuntimeError(
                    f"GPU execution was requested, but no accelerated provider is available. "
                    f"Available providers: {avail}"
                )
            return [*active, "CPUExecutionProvider"]

        # Case-insensitive match against registered ORT provider names
        matched_provider = next((p for p in avail if p.lower() == target), None)
        if matched_provider is not None:
            return [matched_provider]

        raise ValueError(
            f"Unrecognized execution provider '{providers}'. "
            f"Supported shorthands: 'auto', 'cpu', 'cuda', 'gpu'. "
            f"Available providers on this system: {avail}"
        )

    if isinstance(providers, tuple):
        return [providers]

    return list(providers)


class IOBindingRunner:
    """Manages cross-device zero-copy memory binding for ONNX Runtime inference sessions."""

    def __init__(
        self,
        session: ort.InferenceSession,
        device: str | None = None,
        device_id: int | None = None,
    ) -> None:
        """Initialize IOBindingRunner wrapper around an active ONNX Runtime InferenceSession.

        Args:
            session: Active ONNX Runtime InferenceSession instance.
            device: Target execution device ('cuda', 'cpu'). If None, resolved from active provider.
            device_id: Target GPU device ordinal. If None, resolved from active provider options.
        """
        self.session = session

        self.input_names: list[str] = [inp.name for inp in session.get_inputs()]
        self.output_names: list[str] = [out.name for out in session.get_outputs()]

        active_provider = session.get_providers()[0]

        if device is not None:
            dev = device.lower().strip()
            if dev not in VALID_IO_DEVICES:
                raise ValueError(f"Invalid device '{device}'. Supported devices are: {sorted(VALID_IO_DEVICES)}")
            self.device = dev
        elif "cuda" in active_provider.lower() or "tensorrt" in active_provider.lower():
            self.device = "cuda"
        else:
            self.device = "cpu"

        if device_id is not None:
            self.device_id = int(device_id)
        elif self.device == "cuda":
            prov_opts = session.get_provider_options().get(active_provider, {})
            try:
                self.device_id = int(prov_opts.get("device_id", 0))
            except (ValueError, TypeError):
                self.device_id = 0
        else:
            self.device_id = 0

        self.io_binding: ort.IOBinding = self.session.io_binding()
        self._bound_output_names: list[str] = []

    def _resolve_device(self, device: str | None) -> str:
        """Resolve and validate memory target device against supported devices."""
        dev = self.device if device is None else str(device).lower().strip()
        if dev not in VALID_IO_DEVICES:
            raise ValueError(f"Invalid device '{device}'. Supported devices are: {sorted(VALID_IO_DEVICES)}")
        return dev

    def clear_binding(self) -> None:
        """Clear all active input and output bindings on the session."""
        self.io_binding.clear_binding_inputs()
        self.io_binding.clear_binding_outputs()
        self._bound_output_names.clear()

    def synchronize_inputs(self) -> None:
        """Synchronize upstream compute streams before reading bound inputs."""
        self.io_binding.synchronize_inputs()

    def synchronize_outputs(self) -> None:
        """Synchronize execution compute streams before reading bound outputs."""
        self.io_binding.synchronize_outputs()

    def bind_input(
        self,
        name: str,
        tensor: np.ndarray | ort.OrtValue | int,
        device: str | None = None,
        device_id: int | None = None,
        shape: Sequence[int] | None = None,
        dtype: type | np.dtype | None = None,
    ) -> None:
        """Bind an input tensor to device memory or host pinned memory.

        Note:
            device and device_id only apply when binding raw memory pointers (int).
            OrtValue instances retain their intrinsic device placement, and NumPy
            arrays always bind on host CPU memory via ORT's memory arena.

        Args:
            name: Model input node identifier name.
            tensor: Input data as OrtValue, memory pointer address (int), or NumPy array.
            device: Target memory device ('cuda' or 'cpu'). Defaults to session device.
            device_id: GPU device ordinal. Defaults to session device_id.
            shape: Explicit tensor shape sequence required when binding raw memory pointers.
            dtype: Element data type required when binding raw memory pointers.

        Raises:
            ValueError: If input node name is unknown, or if shape/dtype is missing for raw pointers.
            TypeError: If input tensor type is unsupported.
        """
        if name not in self.input_names:
            raise ValueError(f"Unknown input node '{name}'. Valid model inputs are: {self.input_names}")

        target_device = self._resolve_device(device)
        target_device_id = self.device_id if device_id is None else device_id

        # Native OrtValue input
        if isinstance(tensor, ort.OrtValue):
            self.io_binding.bind_ortvalue_input(name, tensor)
            return

        # Raw memory pointer address (e.g., from CUDA / OpenGL interop)
        if isinstance(tensor, int) and not isinstance(tensor, bool):
            if shape is None or dtype is None:
                raise ValueError("shape and dtype must be specified when binding a raw buffer pointer.")
            np_dtype = np.dtype(dtype)
            self.io_binding.bind_input(
                name=name,
                device_type=target_device,
                device_id=target_device_id,
                element_type=np_dtype.type,
                shape=list(shape),
                buffer_ptr=tensor,
            )
            return

        # NumPy array input (originates on host memory)
        if isinstance(tensor, np.ndarray):
            contig_tensor = np.ascontiguousarray(tensor)
            self.io_binding.bind_cpu_input(name, contig_tensor)
            return

        raise TypeError(f"Unsupported input tensor type: {type(tensor).__name__}")

    def bind_output(
        self,
        name: str,
        device: str | None = None,
        device_id: int | None = None,
        shape: Sequence[int] | None = None,
        dtype: type | np.dtype | None = None,
        buffer: ort.OrtValue | int | None = None,
    ) -> None:
        """Pre-allocate or bind model output node in target device memory.

        Args:
            name: Model output node identifier name.
            device: Target memory device ('cuda' for GPU residency, 'cpu' for host memory).
            device_id: GPU device ordinal. Defaults to session device_id.
            shape: Explicit tensor shape sequence required when binding raw memory pointers.
            dtype: Element data type required when binding raw memory pointers.
            buffer: Pre-allocated OrtValue or raw memory pointer address (int).

        Raises:
            ValueError: If output node name is unknown, or if shape/dtype is missing for raw pointers.
            TypeError: If buffer type is unsupported.
        """
        if name not in self.output_names:
            raise ValueError(f"Unknown output node '{name}'. Valid model outputs are: {self.output_names}")

        target_device = self._resolve_device(device)
        target_device_id = self.device_id if device_id is None else device_id

        # Pre-allocated OrtValue output buffer
        if isinstance(buffer, ort.OrtValue):
            self.io_binding.bind_ortvalue_output(name, buffer)
            if name not in self._bound_output_names:
                self._bound_output_names.append(name)
            return

        # Raw memory pointer address (e.g., from CUDA / OpenGL interop)
        if isinstance(buffer, int) and not isinstance(buffer, bool):
            if shape is None or dtype is None:
                raise ValueError("shape and dtype must be specified when binding a raw buffer pointer.")
            np_dtype = np.dtype(dtype)
            self.io_binding.bind_output(
                name=name,
                device_type=target_device,
                device_id=target_device_id,
                element_type=np_dtype.type,
                shape=list(shape),
                buffer_ptr=buffer,
            )
            if name not in self._bound_output_names:
                self._bound_output_names.append(name)
            return

        if buffer is not None:
            raise TypeError(
                f"Unsupported output buffer type: {type(buffer).__name__}. "
                "Expected ort.OrtValue, raw memory pointer address (int), or None."
            )

        # Standard ORT-managed device allocation
        self.io_binding.bind_output(
            name=name,
            device_type=target_device,
            device_id=target_device_id,
        )
        if name not in self._bound_output_names:
            self._bound_output_names.append(name)

    def run(
        self,
        feed_dict: dict[str, np.ndarray | ort.OrtValue] | None = None,
        return_numpy: bool = True,
        output_device: str | None = None,
        sync: bool = False,
    ) -> dict[str, np.ndarray | ort.OrtValue]:
        """Execute forward model inference using bound memory resources.

        Note:
            Passing a non-None feed_dict operates in stateless mode and resets
            active bindings. To execute a pre-bound pipeline with persistent
            GPU buffers, call bind_input() and bind_output() once at startup
            and call run() with feed_dict=None.

        Args:
            feed_dict: Optional mapping of input names to arrays or OrtValues.
            return_numpy: Whether to convert output OrtValues to NumPy arrays (default: True).
            output_device: Target memory device for model outputs ('cuda' or 'cpu').
                If None, defaults to 'cpu' when return_numpy=True, or session device otherwise.
            sync: Whether to synchronize GPU output streams before returning (default: False).
                Recommended when return_numpy=False and outputs are read immediately on another stream.

        Returns:
            Dictionary mapping output node names to NumPy arrays or OrtValues.
        """
        target_out_dev = self._resolve_device(output_device) if output_device is not None else ("cpu" if return_numpy else self.device)

        if feed_dict is not None:
            self.clear_binding()
            for name, val in feed_dict.items():
                self.bind_input(name, val)

            for name in self.output_names:
                self.bind_output(name, device=target_out_dev)
        elif not self._bound_output_names:
            for name in self.output_names:
                self.bind_output(name, device=target_out_dev)

        self.session.run_with_iobinding(self.io_binding)

        if sync:
            self.synchronize_outputs()

        ort_outputs = self.io_binding.get_outputs()
        out_dict: dict[str, np.ndarray | ort.OrtValue] = {}
        for i, name in enumerate(self._bound_output_names):
            val = ort_outputs[i]
            if return_numpy:
                arr = val.numpy()
                out_dict[name] = arr.copy() if val.device_name().lower() == "cpu" else arr
            else:
                out_dict[name] = val
        return out_dict

