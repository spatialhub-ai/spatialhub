"""Shared utilities for model weight downloads, external data management, and ONNX graph validation."""

from __future__ import annotations

import http.cookiejar
import logging
from pathlib import Path
import re
import urllib.parse
import urllib.request

import onnx
from onnx.external_data_helper import convert_model_to_external_data
from tqdm import tqdm

logger = logging.getLogger(__name__)


class DownloadProgressBar(tqdm):
    """Progress bar hook for urllib.request.urlretrieve."""

    def update_to(self, block_num: int = 1, block_size: int = 1, total_size: int | None = None) -> None:
        """Update progress bar based on retrieved chunks.

        Args:
            block_num: Number of blocks transferred so far.
            block_size: Size of each block in bytes.
            total_size: Total size of the file in bytes.
        """
        if total_size is not None and total_size > 0:
            self.total = total_size
        self.update(block_num * block_size - self.n)


def _extract_gdrive_file_id(url_or_id: str) -> str:
    """Extract Google Drive file ID from a URL or raw ID string.

    Args:
        url_or_id: Google Drive URL or direct ID.

    Returns:
        str: Extracted Google Drive file ID.
    """
    if "/d/" in url_or_id:
        return url_or_id.split("/d/")[1].split("/")[0].split("?")[0]
    if "id=" in url_or_id:
        return url_or_id.split("id=")[1].split("&")[0]
    return url_or_id


def _download_gdrive_file(file_id: str, output_path: Path, chunk_size: int = 1024 * 1024) -> Path:
    """Download large files hosted on Google Drive using urllib with confirmation handling.

    Args:
        file_id: Google Drive file identifier.
        output_path: Local target file path.
        chunk_size: Byte size of each streaming chunk.

    Returns:
        Path: Path to the downloaded file.
    """
    cookie_jar = http.cookiejar.CookieJar()
    opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(cookie_jar))

    initial_url = f"https://drive.usercontent.google.com/download?id={file_id}&export=download"
    request = urllib.request.Request(initial_url, headers={"User-Agent": "Mozilla/5.0"})
    response = opener.open(request)

    final_url = initial_url
    content_type = response.info().get_content_type()

    if "text/html" in content_type:
        body = response.read().decode("utf-8", errors="ignore")
        form_match = re.search(r'<form[^>]*id="download-form"[^>]*action="([^"]+)"', body)
        action_url = form_match.group(1) if form_match else "https://drive.usercontent.google.com/download"

        params: dict[str, str] = {}
        for input_match in re.finditer(r'<input[^>]*name="([^"]+)"[^>]*value="([^"]+)"', body):
            params[input_match.group(1)] = input_match.group(2)

        if params:
            final_url = f"{action_url}?{urllib.parse.urlencode(params)}"
        else:
            token_match = re.search(r'confirm=([0-9A-Za-z_]+)', body)
            confirm_token = token_match.group(1) if token_match else "t"
            final_url = f"https://drive.usercontent.google.com/download?id={file_id}&export=download&confirm={confirm_token}"

        request = urllib.request.Request(final_url, headers={"User-Agent": "Mozilla/5.0"})
        response = opener.open(request)

    total_size = int(response.info().get("Content-Length", 0))

    with open(output_path, "wb") as file_handle, tqdm(
        total=total_size if total_size > 0 else None,
        unit="B",
        unit_scale=True,
        unit_divisor=1024,
        desc=output_path.name,
    ) as progress_bar:
        while True:
            chunk = response.read(chunk_size)
            if not chunk:
                break
            file_handle.write(chunk)
            progress_bar.update(len(chunk))

    return output_path


def download_file(url: str, output_path: str | Path, chunk_size: int = 1024 * 1024) -> Path:
    """Download a file from an HTTP/HTTPS URL or Google Drive link to a local destination.

    Args:
        url: Source download URL.
        output_path: Target destination file path or directory.
        chunk_size: Streaming chunk size in bytes for Google Drive downloads.

    Returns:
        Path: Path to verified local file.
    """
    target_path = Path(output_path)
    if target_path.is_dir() or (not target_path.suffix and not target_path.exists()):
        filename = url.split("?")[0].split("/")[-1] or "downloaded_model"
        target_path = target_path / filename

    if target_path.exists():
        logger.info("File already exists at %s. Skipping download.", target_path)
        return target_path

    target_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading from %s to %s...", url, target_path)

    if "drive.google.com" in url or "drive.usercontent.google.com" in url:
        file_id = _extract_gdrive_file_id(url)
        _download_gdrive_file(file_id, target_path, chunk_size=chunk_size)
    else:
        with DownloadProgressBar(unit="B", unit_scale=True, miniters=1, desc=target_path.name) as progress_bar:
            urllib.request.urlretrieve(url, filename=str(target_path), reporthook=progress_bar.update_to)

    logger.info("Download completed successfully: %s", target_path)
    return target_path


def check_onnx(onnx_path: str | Path) -> bool:
    """Validate the integrity and well-formedness of an ONNX model graph.

    Args:
        onnx_path: Path to the .onnx model file.

    Returns:
        bool: True if the model passes integrity checks.
    """
    model_path = Path(onnx_path)
    if not model_path.exists():
        raise FileNotFoundError(f"ONNX file not found at {model_path}")

    logger.info("Validating ONNX graph integrity at %s...", model_path)
    
    try:
        onnx.checker.check_model(model=model_path)
        logger.info("ONNX graph validation passed cleanly.")
        return True
    except onnx.checker.ValidationError as err:
        logger.error("ONNX graph validation failed: %s", err)
        raise RuntimeError(f"ONNX graph validation failed: {err}") from err


def convert_to_external_data(
    onnx_path: str | Path,
    output_path: str | Path | None = None,
    data_filename: str | None = None,
    size_threshold: int = 0,
) -> Path:
    """Consolidate external ONNX tensor data into a single unified data file.

    Args:
        onnx_path: Path to the input ONNX model file.
        output_path: Optional destination path for the saved ONNX file.
        data_filename: Optional filename for the external tensor data.
        size_threshold: Minimum byte size threshold to externalize tensors.

    Returns:
        Path: Path to the saved ONNX model file.
    """
    model_path = Path(onnx_path)
    save_path = Path(output_path) if output_path else model_path
    save_path.parent.mkdir(parents=True, exist_ok=True)

    if data_filename is None:
        data_filename = save_path.with_suffix(".onnx.data").name

    logger.info("Consolidating external data for %s -> %s...", model_path.name, data_filename)
    model = onnx.load(str(model_path), load_external_data=True)

    convert_model_to_external_data(
        model,
        all_tensors_to_one_file=True,
        location=data_filename,
        size_threshold=size_threshold,
    )

    onnx.save_model(
        model,
        str(save_path),
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location=data_filename,
        size_threshold=size_threshold,
    )

    return save_path
