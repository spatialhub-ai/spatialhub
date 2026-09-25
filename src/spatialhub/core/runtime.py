import logging
import mmap
from pathlib import Path
from typing import Any

from huggingface_hub import hf_hub_download

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

