"""Comprehensive unit tests for spatialhub.utils.download module."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest

from spatialhub.utils.download import (
    _extract_gdrive_file_id,
    download_file,
)


class TestExtractGdriveFileId:
    """Test suite for _extract_gdrive_file_id helper."""

    def test_url_with_d_path(self) -> None:
        url = "https://drive.google.com/file/d/1A2B3C4D5E6F/view?usp=sharing"
        assert _extract_gdrive_file_id(url) == "1A2B3C4D5E6F"

    def test_url_with_query_param(self) -> None:
        url = "https://drive.google.com/uc?id=XYZ12345&export=download"
        assert _extract_gdrive_file_id(url) == "XYZ12345"

    def test_raw_id(self) -> None:
        raw_id = "sample_file_id_999"
        assert _extract_gdrive_file_id(raw_id) == "sample_file_id_999"


class TestDownloadFile:
    """Test suite for download_file."""

    def test_existing_file_skips_download(self, tmp_path: Path) -> None:
        target = tmp_path / "model.onnx"
        target.write_bytes(b"existing content")

        with patch("urllib.request.urlretrieve") as mock_retrieve:
            result = download_file("https://example.com/model.onnx", target)
            assert result == target
            mock_retrieve.assert_not_called()

    def test_http_download_to_explicit_file(self, tmp_path: Path) -> None:
        target = tmp_path / "model.onnx"

        def fake_urlretrieve(url: str, filename: str, **kwargs) -> tuple[str, None]:
            Path(filename).write_bytes(b"downloaded content")
            return filename, None

        with patch("urllib.request.urlretrieve", side_effect=fake_urlretrieve):
            result = download_file("https://example.com/model.onnx", target)
            assert result == target
            assert target.exists()
            assert target.read_bytes() == b"downloaded content"

    def test_http_download_to_directory(self, tmp_path: Path) -> None:
        dest_dir = tmp_path / "weights"
        dest_dir.mkdir()

        def fake_urlretrieve(url: str, filename: str, **kwargs) -> tuple[str, None]:
            Path(filename).write_bytes(b"weight content")
            return filename, None

        with patch("urllib.request.urlretrieve", side_effect=fake_urlretrieve):
            result = download_file("https://example.com/weights/best.pt", dest_dir)
            expected_file = dest_dir / "best.pt"
            assert result == expected_file
            assert expected_file.exists()

    def test_gdrive_download_dispatch(self, tmp_path: Path) -> None:
        target = tmp_path / "gdrive_model.onnx"
        gdrive_url = "https://drive.google.com/file/d/12345ABCDE/view"

        def fake_gdrive_download(file_id: str, out_path: Path, **kwargs) -> Path:
            assert file_id == "12345ABCDE"
            out_path.write_bytes(b"gdrive payload")
            return out_path

        with patch("spatialhub.utils.download._download_gdrive_file", side_effect=fake_gdrive_download) as mock_gdrive:
            result = download_file(gdrive_url, target)
            assert result == target
            assert target.exists()
            assert target.read_bytes() == b"gdrive payload"
            mock_gdrive.assert_called_once()

    def test_download_failure_cleans_up_temp_file(self, tmp_path: Path) -> None:
        target = tmp_path / "failing_model.onnx"

        def fake_failing_urlretrieve(url: str, filename: str, **kwargs) -> None:
            Path(filename).write_bytes(b"partial corrupted data")
            raise ConnectionError("Network dropped")

        with patch("urllib.request.urlretrieve", side_effect=fake_failing_urlretrieve):
            with pytest.raises(ConnectionError, match="Network dropped"):
                download_file("https://example.com/bad_link", target)

        assert not target.exists()
        temp_file = target.with_name(f".{target.name}.tmp")
        assert not temp_file.exists()
