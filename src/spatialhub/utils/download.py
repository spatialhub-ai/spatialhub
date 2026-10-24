"""Network download utilities supporting standard HTTP/HTTPS and Google Drive sources."""

from __future__ import annotations

import http.cookiejar
import logging
from pathlib import Path
import re
import urllib.parse
import urllib.request

logger = logging.getLogger(__name__)


def _extract_gdrive_file_id(url_or_id: str) -> str:
    """Extract Google Drive file ID from a URL or raw ID string.

    Args:
        url_or_id: Google Drive URL or direct ID.

    Returns:
        Extracted Google Drive file ID.
    """
    if "/d/" in url_or_id:
        return url_or_id.split("/d/")[1].split("/")[0].split("?")[0]
    if "id=" in url_or_id:
        return url_or_id.split("id=")[1].split("&")[0]
    return url_or_id


def _download_gdrive_file(
    file_id: str,
    output_path: Path,
    chunk_size: int = 1024 * 1024,
    desc: str | None = None,
) -> Path:
    """Download large files hosted on Google Drive with confirmation token handling.

    Args:
        file_id: Google Drive file identifier.
        output_path: Local target file path.
        chunk_size: Byte size of streaming chunks.
        desc: Description label for progress display.

    Returns:
        Path to the downloaded file.
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
            token_match = re.search(r"confirm=([0-9A-Za-z_]+)", body)
            confirm_token = token_match.group(1) if token_match else "t"
            final_url = f"https://drive.usercontent.google.com/download?id={file_id}&export=download&confirm={confirm_token}"

        request = urllib.request.Request(final_url, headers={"User-Agent": "Mozilla/5.0"})
        response = opener.open(request)

    total_size = int(response.info().get("Content-Length", 0))

    try:
        from tqdm import tqdm

        progress_bar = tqdm(
            total=total_size if total_size > 0 else None,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
            desc=desc or output_path.name,
        )
    except ImportError:
        progress_bar = None

    with open(output_path, "wb") as file_handle:
        while True:
            chunk = response.read(chunk_size)
            if not chunk:
                break
            file_handle.write(chunk)
            if progress_bar is not None:
                progress_bar.update(len(chunk))

    if progress_bar is not None:
        progress_bar.close()

    return output_path


def download_file(url: str, output_path: str | Path, chunk_size: int = 1024 * 1024) -> Path:
    """Download a file from an HTTP/HTTPS URL or Google Drive link to a local destination.

    Args:
        url: Source download URL.
        output_path: Target destination file path or directory.
        chunk_size: Streaming chunk size in bytes for Google Drive downloads.

    Returns:
        Path to the downloaded file.
    """
    target_path = Path(output_path)
    if target_path.is_dir() or (not target_path.suffix and not target_path.exists()):
        filename = url.split("?")[0].split("/")[-1] or "downloaded_file"
        target_path = target_path / filename

    if target_path.exists():
        logger.info("File already exists at %s. Skipping download.", target_path)
        return target_path

    target_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading from %s to %s...", url, target_path)

    temp_path = target_path.with_name(f".{target_path.name}.tmp")

    try:
        if "drive.google.com" in url or "drive.usercontent.google.com" in url:
            file_id = _extract_gdrive_file_id(url)
            _download_gdrive_file(file_id, temp_path, chunk_size=chunk_size, desc=target_path.name)
        else:
            try:
                from tqdm import tqdm

                class _DownloadProgressBar(tqdm):
                    def update_to(self, block_num: int = 1, block_size: int = 1, total_size: int | None = None) -> None:
                        if total_size is not None and total_size > 0:
                            self.total = total_size
                        self.update(block_num * block_size - self.n)

                with _DownloadProgressBar(unit="B", unit_scale=True, miniters=1, desc=target_path.name) as bar:
                    urllib.request.urlretrieve(url, filename=str(temp_path), reporthook=bar.update_to)
            except ImportError:
                urllib.request.urlretrieve(url, filename=str(temp_path))

        temp_path.replace(target_path)
    except Exception:
        if temp_path.exists():
            temp_path.unlink()
        raise

    logger.info("Download completed successfully: %s", target_path)
    return target_path


__all__ = ["download_file"]
