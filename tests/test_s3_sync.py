"""Tests for scripts/s3_sync.py (S3_Sync_Tool)."""

import hashlib
import subprocess
import sys
from pathlib import Path

import pytest

from scripts.s3_sync import compute_etag_match


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _md5_hex(data: bytes) -> str:
    """Return the hex MD5 digest of *data*."""
    return hashlib.md5(data).hexdigest()


# ---------------------------------------------------------------------------
# compute_etag_match — unit tests
# ---------------------------------------------------------------------------


class TestComputeEtagMatch:
    """Unit tests for compute_etag_match()."""

    def test_returns_false_when_file_does_not_exist(self, tmp_path: Path) -> None:
        missing = tmp_path / "nonexistent.jpg"
        assert compute_etag_match(missing, '"abc123"') is False

    def test_returns_true_when_md5_matches(self, tmp_path: Path) -> None:
        data = b"hello world"
        f = tmp_path / "file.bin"
        f.write_bytes(data)
        etag = f'"{_md5_hex(data)}"'
        assert compute_etag_match(f, etag) is True

    def test_returns_false_when_md5_does_not_match(self, tmp_path: Path) -> None:
        data = b"hello world"
        f = tmp_path / "file.bin"
        f.write_bytes(data)
        wrong_etag = '"deadbeefdeadbeefdeadbeefdeadbeef"'
        assert compute_etag_match(f, wrong_etag) is False

    def test_strips_surrounding_double_quotes(self, tmp_path: Path) -> None:
        data = b"strip me"
        f = tmp_path / "file.bin"
        f.write_bytes(data)
        raw_hex = _md5_hex(data)
        # Pass with quotes — must still match
        assert compute_etag_match(f, f'"{raw_hex}"') is True
        # Pass without quotes — must still match
        assert compute_etag_match(f, raw_hex) is True

    def test_returns_false_for_multipart_etag(self, tmp_path: Path) -> None:
        """ETags containing '-' indicate multipart uploads; always re-download."""
        data = b"multipart content"
        f = tmp_path / "file.bin"
        f.write_bytes(data)
        multipart_etag = '"abc123def456abc123def456abc123de-3"'
        assert compute_etag_match(f, multipart_etag) is False

    def test_returns_false_for_multipart_etag_without_quotes(
        self, tmp_path: Path
    ) -> None:
        data = b"content"
        f = tmp_path / "file.bin"
        f.write_bytes(data)
        assert compute_etag_match(f, "abc123-2") is False

    def test_handles_empty_file(self, tmp_path: Path) -> None:
        f = tmp_path / "empty.bin"
        f.write_bytes(b"")
        etag = f'"{_md5_hex(b"")}"'
        assert compute_etag_match(f, etag) is True

    def test_handles_large_file_in_chunks(self, tmp_path: Path) -> None:
        """File larger than one 8 192-byte chunk should still be hashed correctly."""
        data = b"x" * 100_000  # ~100 KB
        f = tmp_path / "large.bin"
        f.write_bytes(data)
        etag = f'"{_md5_hex(data)}"'
        assert compute_etag_match(f, etag) is True


# ---------------------------------------------------------------------------
# CLI argument validation — unit tests (Req 1.2, 1.9)
# ---------------------------------------------------------------------------


def _run_s3_sync(*args: str) -> subprocess.CompletedProcess:
    """Run s3_sync as a module in a subprocess and return the result."""
    return subprocess.run(
        [sys.executable, "-m", "scripts.s3_sync", *args],
        capture_output=True,
        text=True,
        cwd=Path(__file__).parent.parent,
    )


class TestMaxFilesValidation:
    """Tests for --max-files argument validation (Req 1.2)."""

    def test_zero_max_files_exits_nonzero(self) -> None:
        result = _run_s3_sync("--max-files", "0")
        assert result.returncode != 0
        assert "error" in result.stdout.lower() or "error" in result.stderr.lower()

    def test_negative_max_files_exits_nonzero(self) -> None:
        result = _run_s3_sync("--max-files", "-5")
        assert result.returncode != 0

    def test_positive_max_files_does_not_exit_on_validation(self) -> None:
        # A positive value passes validation; the script will fail later at
        # the AWS credential step, but should NOT fail at the --max-files check.
        result = _run_s3_sync("--max-files", "10")
        # Must NOT exit with the --max-files error message
        combined = result.stdout + result.stderr
        assert "--max-files must be a positive integer" not in combined


class TestSourceValidation:
    """Tests for --source argument validation (Req 1.9)."""

    def test_invalid_source_exits_nonzero(self) -> None:
        result = _run_s3_sync("--source", "invalid_value")
        assert result.returncode != 0

    def test_invalid_source_lists_valid_values(self) -> None:
        result = _run_s3_sync("--source", "bad_source")
        combined = result.stdout + result.stderr
        assert "images" in combined
        assert "videos" in combined

    def test_images_source_passes_validation(self) -> None:
        result = _run_s3_sync("--source", "images")
        combined = result.stdout + result.stderr
        assert "--source must be one of" not in combined

    def test_videos_source_passes_validation(self) -> None:
        result = _run_s3_sync("--source", "videos")
        combined = result.stdout + result.stderr
        assert "--source must be one of" not in combined
