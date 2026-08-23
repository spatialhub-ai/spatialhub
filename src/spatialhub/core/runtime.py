import logging
from pathlib import Path

from huggingface_hub import hf_hub_download
import onnxruntime as ort

logger = logging.getLogger(__name__)


def resolve_model_path(
                    model_path: str | Path | None = None,
                    repo_id: str | None = None,
                    filename: str | None = None,
                    download_sidecar_data: bool = False,
                ) -> Path:
    """Resolves a local model path or downloads weights from Hugging Face Hub.

    Args:
        model_path: Explicit local path to the ONNX weight file.
        repo_id: Hugging Face repository ID.
        filename: Target filename in the repository.
        download_sidecar_data: If True, attempts to fetch companion '.data' files for large ONNX models (>2GB) with external weights.

    Returns:
        Path: Resolved absolute path to the local ONNX model file.

    Raises:
        FileNotFoundError: If no local file is found and no HF coordinates are provided.
        RuntimeError: If download from Hugging Face fails.
    """
    if model_path is not None:
        local_file = Path(model_path)
        if local_file.exists():
            return local_file
        logger.info("Provided model_path '%s' not found locally.", model_path)

    if not repo_id or not filename:
        raise FileNotFoundError(f"Model file not found locally at '{model_path}', and no Hugging Face repository/filename was provided for automatic download.")

    logger.info("Fetching '%s' from Hugging Face repository '%s'...", filename, repo_id)
    try:
        downloaded = hf_hub_download(repo_id=repo_id, filename=filename)
        resolved_path = Path(downloaded)

        if download_sidecar_data:
            sidecar_name = f"{filename}.data"
            try:
                hf_hub_download(repo_id=repo_id, filename=sidecar_name)
                logger.debug("Fetched companion data file: %s", sidecar_name)
            except Exception:
                # Self-contained ONNX models do not have .data files
                pass

        return resolved_path

    except Exception as exc:
        raise RuntimeError(
            f"Failed to download '{filename}' from repository '{repo_id}'. Please verify network connection and repository accessibility. Error: {exc}"
        ) from exc


def create_ort_session(
                    model_path: str | Path,
                    providers: list[str] | str | None = None,
                    session_options: ort.SessionOptions | None = None,
                    log_severity_level: int | None = None,
                ) -> ort.InferenceSession:
    """Initializes and verifies an ONNX Runtime InferenceSession from a local file.

    Args:
        model_path: Path to the ONNX model binary on disk.
        providers: Target execution provider(s) (e.g., 'CUDAExecutionProvider', 'CPUExecutionProvider').
        session_options: Custom ONNX session options. If None, default optimized options are used.
        log_severity_level: ONNX Runtime internal logging level (3 = Error only).

    Returns:
        ort.InferenceSession: Initialized and verified session.

    Raises:
        FileNotFoundError: If the resolved path does not exist on disk.
    """
    path_obj = Path(model_path)
    if not path_obj.exists():
        raise FileNotFoundError(f"Cannot initialize ONNX session. File does not exist at: {path_obj}")

    # Normalize providers
    if providers is None:
        execution_providers = ["CPUExecutionProvider"]
    elif isinstance(providers, str):
        execution_providers = [providers]
    else:
        execution_providers = list(providers)

    # Configure session options
    if session_options is None:
        session_options = ort.SessionOptions()
        if log_severity_level is not None:
            session_options.log_severity_level = log_severity_level

    session = ort.InferenceSession(
                                str(path_obj),
                                sess_options=session_options,
                                providers=execution_providers,
                            )

    # Validate active execution provider
    active_provider = session.get_providers()[0]
    requested_provider = execution_providers[0]

    if active_provider != requested_provider:
        logger.warning("Requested provider '%s', but ONNX Runtime fell back to '%s'.", requested_provider, active_provider,)
    else:
        logger.debug("ONNX session initialized on provider: %s", active_provider)

    return session

