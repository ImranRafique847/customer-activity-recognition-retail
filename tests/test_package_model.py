"""Tests for scripts/package_model.py (Model_Packager).

Covers:
  - create_model_archive  (task 7.2)
  - verify_archive_integrity  (task 7.3)
  - upload_archive  (task 7.4)
  - main() CLI wiring  (tasks 7.2–7.4)
"""

import json
import os
import sys
import tarfile
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from scripts.package_model import (
    create_model_archive,
    upload_archive,
    validate_training_summary,
    verify_archive_integrity,
)


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

VALID_SUMMARY = {
    "model_architecture": "yolov8n",
    "input_size": 416,
    "num_classes": 2,
    "class_names": ["Picking", "Returning"],
    "dataset_version": "v1.0",
    "training_device": "cuda:0",
}


def _make_temp_files(tmp_path: Path, model_size: int = 2048):
    """Create minimal fake source files and return their paths."""
    model_pt = tmp_path / "best.pt"
    model_pt.write_bytes(b"x" * model_size)  # 2 KB fake weights

    inference_py = tmp_path / "inference.py"
    inference_py.write_text("# inference handler\n", encoding="utf-8")

    requirements_txt = tmp_path / "requirements.txt"
    requirements_txt.write_text("ultralytics>=8.0\n", encoding="utf-8")

    return model_pt, inference_py, requirements_txt


# ---------------------------------------------------------------------------
# validate_training_summary
# ---------------------------------------------------------------------------


class TestValidateTrainingSummary:
    """Unit tests for validate_training_summary (task 7.1, Requirement 9.2)."""

    def _write_summary(self, tmp_path: Path, data: dict) -> Path:
        p = tmp_path / "training_summary.json"
        p.write_text(json.dumps(data), encoding="utf-8")
        return p

    # --- happy path ---

    def test_valid_summary_returns_dict(self, tmp_path):
        """A fully valid summary is returned as a dict."""
        path = self._write_summary(tmp_path, VALID_SUMMARY)
        result = validate_training_summary(path)
        assert result == VALID_SUMMARY

    # --- file-level errors ---

    def test_missing_file_raises_file_not_found(self, tmp_path):
        """FileNotFoundError raised when the file does not exist."""
        with pytest.raises(FileNotFoundError):
            validate_training_summary(tmp_path / "nonexistent.json")

    def test_invalid_json_raises_value_error(self, tmp_path):
        """ValueError raised for malformed JSON."""
        p = tmp_path / "training_summary.json"
        p.write_text("{not valid json}", encoding="utf-8")
        with pytest.raises(ValueError, match="invalid JSON"):
            validate_training_summary(p)

    # --- model_architecture ---

    def test_missing_model_architecture_raises(self, tmp_path):
        data = {k: v for k, v in VALID_SUMMARY.items() if k != "model_architecture"}
        path = self._write_summary(tmp_path, data)
        with pytest.raises(ValueError, match="model_architecture"):
            validate_training_summary(path)

    def test_wrong_type_model_architecture_raises(self, tmp_path):
        data = {**VALID_SUMMARY, "model_architecture": 42}
        path = self._write_summary(tmp_path, data)
        with pytest.raises(ValueError, match="model_architecture"):
            validate_training_summary(path)

    def test_empty_model_architecture_raises(self, tmp_path):
        data = {**VALID_SUMMARY, "model_architecture": ""}
        path = self._write_summary(tmp_path, data)
        with pytest.raises(ValueError, match="model_architecture"):
            validate_training_summary(path)

    # --- input_size ---

    def test_missing_input_size_raises(self, tmp_path):
        data = {k: v for k, v in VALID_SUMMARY.items() if k != "input_size"}
        path = self._write_summary(tmp_path, data)
        with pytest.raises(ValueError, match="input_size"):
            validate_training_summary(path)

    def test_wrong_type_input_size_raises(self, tmp_path):
        data = {**VALID_SUMMARY, "input_size": "416"}
        path = self._write_summary(tmp_path, data)
        with pytest.raises(ValueError, match="input_size"):
            validate_training_summary(path)

    def test_zero_input_size_raises(self, tmp_path):
        data = {**VALID_SUMMARY, "input_size": 0}
        path = self._write_summary(tmp_path, data)
        with pytest.raises(ValueError, match="input_size"):
            validate_training_summary(path)

    def test_negative_input_size_raises(self, tmp_path):
        data = {**VALID_SUMMARY, "input_size": -1}
        path = self._write_summary(tmp_path, data)
        with pytest.raises(ValueError, match="input_size"):
            validate_training_summary(path)

    def test_bool_input_size_raises(self, tmp_path):
        """Booleans are a subclass of int in Python — must be rejected."""
        data = {**VALID_SUMMARY, "input_size": True}
        path = self._write_summary(tmp_path, data)
        with pytest.raises(ValueError, match="input_size"):
            validate_training_summary(path)

    # --- num_classes ---

    def test_missing_num_classes_raises(self, tmp_path):
        data = {k: v for k, v in VALID_SUMMARY.items() if k != "num_classes"}
        path = self._write_summary(tmp_path, data)
        with pytest.raises(ValueError, match="num_classes"):
            validate_training_summary(path)

    def test_wrong_type_num_classes_raises(self, tmp_path):
        data = {**VALID_SUMMARY, "num_classes": 5.0}
        path = self._write_summary(tmp_path, data)
        with pytest.raises(ValueError, match="num_classes"):
            validate_training_summary(path)

    def test_zero_num_classes_raises(self, tmp_path):
        data = {**VALID_SUMMARY, "num_classes": 0, "class_names": []}
        path = self._write_summary(tmp_path, data)
        with pytest.raises(ValueError, match="num_classes"):
            validate_training_summary(path)

    def test_negative_num_classes_raises(self, tmp_path):
        data = {**VALID_SUMMARY, "num_classes": -3}
        path = self._write_summary(tmp_path, data)
        with pytest.raises(ValueError, match="num_classes"):
            validate_training_summary(path)

    # --- class_names ---

    def test_missing_class_names_raises(self, tmp_path):
        data = {k: v for k, v in VALID_SUMMARY.items() if k != "class_names"}
        path = self._write_summary(tmp_path, data)
        with pytest.raises(ValueError, match="class_names"):
            validate_training_summary(path)

    def test_class_names_not_list_raises(self, tmp_path):
        data = {**VALID_SUMMARY, "class_names": "Picking,Returning"}
        path = self._write_summary(tmp_path, data)
        with pytest.raises(ValueError, match="class_names"):
            validate_training_summary(path)

    def test_class_names_contains_non_str_raises(self, tmp_path):
        data = {**VALID_SUMMARY, "class_names": ["Picking", 2]}
        path = self._write_summary(tmp_path, data)
        with pytest.raises(ValueError, match="class_names"):
            validate_training_summary(path)

    def test_class_names_length_mismatch_raises(self, tmp_path):
        """len(class_names) != num_classes must raise ValueError."""
        data = {**VALID_SUMMARY, "num_classes": 3, "class_names": ["A", "B"]}
        path = self._write_summary(tmp_path, data)
        with pytest.raises(ValueError, match="class_names"):
            validate_training_summary(path)

    # --- dataset_version ---

    def test_missing_dataset_version_raises(self, tmp_path):
        data = {k: v for k, v in VALID_SUMMARY.items() if k != "dataset_version"}
        path = self._write_summary(tmp_path, data)
        with pytest.raises(ValueError, match="dataset_version"):
            validate_training_summary(path)

    def test_wrong_type_dataset_version_raises(self, tmp_path):
        data = {**VALID_SUMMARY, "dataset_version": 3}
        path = self._write_summary(tmp_path, data)
        with pytest.raises(ValueError, match="dataset_version"):
            validate_training_summary(path)

    def test_empty_dataset_version_raises(self, tmp_path):
        data = {**VALID_SUMMARY, "dataset_version": ""}
        path = self._write_summary(tmp_path, data)
        with pytest.raises(ValueError, match="dataset_version"):
            validate_training_summary(path)

    # --- training_device ---

    def test_missing_training_device_raises(self, tmp_path):
        data = {k: v for k, v in VALID_SUMMARY.items() if k != "training_device"}
        path = self._write_summary(tmp_path, data)
        with pytest.raises(ValueError, match="training_device"):
            validate_training_summary(path)

    def test_wrong_type_training_device_raises(self, tmp_path):
        data = {**VALID_SUMMARY, "training_device": ["cuda:0"]}
        path = self._write_summary(tmp_path, data)
        with pytest.raises(ValueError, match="training_device"):
            validate_training_summary(path)

    def test_empty_training_device_raises(self, tmp_path):
        data = {**VALID_SUMMARY, "training_device": ""}
        path = self._write_summary(tmp_path, data)
        with pytest.raises(ValueError, match="training_device"):
            validate_training_summary(path)

    # --- field ordering: first invalid field is reported ---

    def test_first_missing_field_reported(self, tmp_path):
        """With multiple missing fields, the first (model_architecture) is identified."""
        data = {
            "num_classes": 2,
            "class_names": ["A", "B"],
        }
        path = self._write_summary(tmp_path, data)
        with pytest.raises(ValueError, match="model_architecture"):
            validate_training_summary(path)


# ---------------------------------------------------------------------------
# verify_archive_integrity
# ---------------------------------------------------------------------------


class TestVerifyArchiveIntegrity:
    def test_valid_archive_passes(self, tmp_path):
        """A well-formed archive with best.pt ≥1KB and code/inference.py passes."""
        archive = tmp_path / "model.tar.gz"
        with tarfile.open(archive, "w:gz") as tar:
            pt = tmp_path / "best.pt"
            pt.write_bytes(b"w" * 1024)
            tar.add(pt, arcname="best.pt")

            inf = tmp_path / "inference.py"
            inf.write_text("def model_fn(): pass\n")
            tar.add(inf, arcname="code/inference.py")

        # Should not raise
        verify_archive_integrity(archive)

    def test_best_pt_too_small_raises(self, tmp_path):
        """best.pt < 1024 bytes must raise ValueError."""
        archive = tmp_path / "model.tar.gz"
        with tarfile.open(archive, "w:gz") as tar:
            pt = tmp_path / "best.pt"
            pt.write_bytes(b"w" * 100)  # only 100 bytes
            tar.add(pt, arcname="best.pt")

            inf = tmp_path / "inference.py"
            inf.write_text("def model_fn(): pass\n")
            tar.add(inf, arcname="code/inference.py")

        with pytest.raises(ValueError, match="best.pt"):
            verify_archive_integrity(archive)

    def test_inference_py_empty_raises(self, tmp_path):
        """code/inference.py with 0 bytes must raise ValueError."""
        archive = tmp_path / "model.tar.gz"
        with tarfile.open(archive, "w:gz") as tar:
            pt = tmp_path / "best.pt"
            pt.write_bytes(b"w" * 2048)
            tar.add(pt, arcname="best.pt")

            inf = tmp_path / "inference.py"
            inf.write_bytes(b"")  # empty
            tar.add(inf, arcname="code/inference.py")

        with pytest.raises(ValueError, match="code/inference.py"):
            verify_archive_integrity(archive)

    def test_tempdir_cleaned_up_on_success(self, tmp_path, monkeypatch):
        """The temporary extraction directory is removed after a successful check."""
        created_dirs = []
        real_mkdtemp = tempfile.mkdtemp

        def tracking_mkdtemp(*args, **kwargs):
            d = real_mkdtemp(*args, **kwargs)
            created_dirs.append(d)
            return d

        monkeypatch.setattr(tempfile, "mkdtemp", tracking_mkdtemp)

        archive = tmp_path / "model.tar.gz"
        with tarfile.open(archive, "w:gz") as tar:
            pt = tmp_path / "best.pt"
            pt.write_bytes(b"w" * 1024)
            tar.add(pt, arcname="best.pt")
            inf = tmp_path / "inference.py"
            inf.write_text("# ok\n")
            tar.add(inf, arcname="code/inference.py")

        verify_archive_integrity(archive)

        for d in created_dirs:
            assert not Path(d).exists(), f"Temp dir {d} was not cleaned up"

    def test_tempdir_cleaned_up_on_failure(self, tmp_path, monkeypatch):
        """The temporary extraction directory is removed even when verification fails."""
        created_dirs = []
        real_mkdtemp = tempfile.mkdtemp

        def tracking_mkdtemp(*args, **kwargs):
            d = real_mkdtemp(*args, **kwargs)
            created_dirs.append(d)
            return d

        monkeypatch.setattr(tempfile, "mkdtemp", tracking_mkdtemp)

        archive = tmp_path / "model.tar.gz"
        with tarfile.open(archive, "w:gz") as tar:
            pt = tmp_path / "best.pt"
            pt.write_bytes(b"w" * 10)  # too small — triggers failure
            tar.add(pt, arcname="best.pt")
            inf = tmp_path / "inference.py"
            inf.write_text("# ok\n")
            tar.add(inf, arcname="code/inference.py")

        with pytest.raises(ValueError):
            verify_archive_integrity(archive)

        for d in created_dirs:
            assert not Path(d).exists(), f"Temp dir {d} was not cleaned up on failure"


# ---------------------------------------------------------------------------
# create_model_archive
# ---------------------------------------------------------------------------


class TestCreateModelArchive:
    def test_creates_archive_with_correct_layout(self, tmp_path):
        """Archive contains best.pt, code/inference.py, code/requirements.txt,
        and code/model_config.json."""
        model_pt, inference_py, requirements_txt = _make_temp_files(tmp_path)
        output_dir = tmp_path / "out"

        archive = create_model_archive(
            model_path=model_pt,
            inference_py=inference_py,
            requirements_txt=requirements_txt,
            training_summary=VALID_SUMMARY,
            output_dir=output_dir,
        )

        assert archive.exists()
        with tarfile.open(archive) as tar:
            names = tar.getnames()

        assert "best.pt" in names
        assert "code/inference.py" in names
        assert "code/requirements.txt" in names
        assert "code/model_config.json" in names

    def test_model_config_json_content(self, tmp_path):
        """code/model_config.json contains num_classes, class_names, input_size."""
        model_pt, inference_py, requirements_txt = _make_temp_files(tmp_path)
        output_dir = tmp_path / "out"

        archive = create_model_archive(
            model_path=model_pt,
            inference_py=inference_py,
            requirements_txt=requirements_txt,
            training_summary=VALID_SUMMARY,
            output_dir=output_dir,
        )

        with tarfile.open(archive) as tar:
            member = tar.getmember("code/model_config.json")
            content = tar.extractfile(member).read()

        config = json.loads(content)
        assert config["num_classes"] == VALID_SUMMARY["num_classes"]
        assert config["class_names"] == VALID_SUMMARY["class_names"]
        assert config["input_size"] == VALID_SUMMARY["input_size"]
        # Must contain exactly these three keys
        assert set(config.keys()) == {"num_classes", "class_names", "input_size"}

    def test_missing_model_pt_raises_file_not_found(self, tmp_path):
        """FileNotFoundError is raised when best.pt does not exist."""
        _, inference_py, requirements_txt = _make_temp_files(tmp_path)
        missing_pt = tmp_path / "nonexistent.pt"

        with pytest.raises(FileNotFoundError, match="nonexistent.pt"):
            create_model_archive(
                model_path=missing_pt,
                inference_py=inference_py,
                requirements_txt=requirements_txt,
                training_summary=VALID_SUMMARY,
                output_dir=tmp_path / "out",
            )

    def test_missing_inference_py_raises_file_not_found(self, tmp_path):
        """FileNotFoundError is raised when code/inference.py does not exist."""
        model_pt, _, requirements_txt = _make_temp_files(tmp_path)
        missing_inf = tmp_path / "no_inference.py"

        with pytest.raises(FileNotFoundError, match="no_inference.py"):
            create_model_archive(
                model_path=model_pt,
                inference_py=missing_inf,
                requirements_txt=requirements_txt,
                training_summary=VALID_SUMMARY,
                output_dir=tmp_path / "out",
            )

    def test_missing_requirements_txt_raises_file_not_found(self, tmp_path):
        """FileNotFoundError is raised when code/requirements.txt does not exist."""
        model_pt, inference_py, _ = _make_temp_files(tmp_path)
        missing_req = tmp_path / "no_requirements.txt"

        with pytest.raises(FileNotFoundError, match="no_requirements.txt"):
            create_model_archive(
                model_path=model_pt,
                inference_py=inference_py,
                requirements_txt=missing_req,
                training_summary=VALID_SUMMARY,
                output_dir=tmp_path / "out",
            )

    def test_does_not_create_archive_before_checking_files(self, tmp_path):
        """No partial archive is created when a source file is missing."""
        _, inference_py, requirements_txt = _make_temp_files(tmp_path)
        output_dir = tmp_path / "out"

        with pytest.raises(FileNotFoundError):
            create_model_archive(
                model_path=tmp_path / "ghost.pt",
                inference_py=inference_py,
                requirements_txt=requirements_txt,
                training_summary=VALID_SUMMARY,
                output_dir=output_dir,
            )

        # The output dir may or may not exist, but model.tar.gz must not
        assert not (output_dir / "model.tar.gz").exists()

    def test_output_dir_created_if_absent(self, tmp_path):
        """output_dir is created automatically if it does not exist."""
        model_pt, inference_py, requirements_txt = _make_temp_files(tmp_path)
        output_dir = tmp_path / "deep" / "nested" / "out"

        archive = create_model_archive(
            model_path=model_pt,
            inference_py=inference_py,
            requirements_txt=requirements_txt,
            training_summary=VALID_SUMMARY,
            output_dir=output_dir,
        )

        assert archive.exists()

    def test_returns_absolute_path(self, tmp_path):
        """Returned path is absolute."""
        model_pt, inference_py, requirements_txt = _make_temp_files(tmp_path)
        archive = create_model_archive(
            model_path=model_pt,
            inference_py=inference_py,
            requirements_txt=requirements_txt,
            training_summary=VALID_SUMMARY,
            output_dir=tmp_path / "out",
        )
        assert archive.is_absolute()


# ---------------------------------------------------------------------------
# upload_archive
# ---------------------------------------------------------------------------


class TestUploadArchive:
    def _make_archive(self, tmp_path: Path) -> Path:
        model_pt, inference_py, requirements_txt = _make_temp_files(tmp_path)
        return create_model_archive(
            model_path=model_pt,
            inference_py=inference_py,
            requirements_txt=requirements_txt,
            training_summary=VALID_SUMMARY,
            output_dir=tmp_path / "out",
        )

    def test_returns_correct_s3_uri(self, tmp_path):
        """Returns s3://{bucket}/models/{run_name}/model.tar.gz on success."""
        archive = self._make_archive(tmp_path)
        mock_s3 = MagicMock()

        uri = upload_archive(archive, "my-bucket", "run_001", mock_s3)

        assert uri == "s3://my-bucket/models/run_001/model.tar.gz"

    def test_calls_upload_file_with_correct_args(self, tmp_path):
        """upload_file is called with the archive path, bucket, and key."""
        archive = self._make_archive(tmp_path)
        mock_s3 = MagicMock()

        upload_archive(archive, "my-bucket", "run_001", mock_s3)

        mock_s3.upload_file.assert_called_once_with(
            str(archive), "my-bucket", "models/run_001/model.tar.gz"
        )

    def test_raises_runtime_error_on_upload_failure(self, tmp_path):
        """RuntimeError is raised when upload_file raises."""
        archive = self._make_archive(tmp_path)
        mock_s3 = MagicMock()
        mock_s3.upload_file.side_effect = Exception("connection refused")

        with pytest.raises(RuntimeError, match="Failed to upload"):
            upload_archive(archive, "my-bucket", "run_001", mock_s3)

    def test_runtime_error_includes_bucket_and_key(self, tmp_path):
        """RuntimeError message identifies the destination s3 key."""
        archive = self._make_archive(tmp_path)
        mock_s3 = MagicMock()
        mock_s3.upload_file.side_effect = Exception("timeout")

        with pytest.raises(RuntimeError) as exc_info:
            upload_archive(archive, "retaildata-cv-2026", "run_A", mock_s3)

        msg = str(exc_info.value)
        assert "retaildata-cv-2026" in msg
        assert "models/run_A/model.tar.gz" in msg


# ---------------------------------------------------------------------------
# main() CLI
# ---------------------------------------------------------------------------


class TestMainCLI:
    """Smoke tests for the wired-up main() entrypoint."""

    def _call_main(self, args, env=None, mock_s3=None):
        """Invoke main() with patched sys.argv and environment."""
        default_env = {
            "AWS_ACCESS_KEY_ID": "AKIAIOSFODNN7EXAMPLE",
            "AWS_SECRET_ACCESS_KEY": "wJalrXUtnFEMI",
            "AWS_DEFAULT_REGION": "ap-southeast-2",
        }
        env = {**default_env, **(env or {})}

        if mock_s3 is None:
            mock_s3 = MagicMock()

        with (
            patch("sys.argv", ["package_model"] + args),
            patch.dict(os.environ, env, clear=False),
            patch("scripts.package_model.load_env_and_validate"),
            patch("scripts.package_model.boto3.client", return_value=mock_s3),
        ):
            return mock_s3

    def test_invalid_run_name_exits_1(self, tmp_path, capsys):
        """main() exits with code 1 when --run-name contains invalid characters."""
        with pytest.raises(SystemExit) as exc_info:
            with (
                patch("sys.argv", ["package_model",
                                   "--model-path", str(tmp_path / "best.pt"),
                                   "--run-name", "bad name!"]),
                patch("scripts.package_model.load_env_and_validate"),
            ):
                from scripts.package_model import main
                main()

        assert exc_info.value.code == 1
        captured = capsys.readouterr()
        assert "Error" in captured.out

    def test_success_prints_s3_uri_and_exits_0(self, tmp_path, capsys):
        """main() prints the S3 URI and exits 0 on a successful run."""
        model_pt, inference_py, requirements_txt = _make_temp_files(tmp_path)
        summary = tmp_path / "training_summary.json"
        summary.write_text(json.dumps(VALID_SUMMARY), encoding="utf-8")

        mock_s3 = MagicMock()

        # code/inference.py and code/requirements.txt are read from the real
        # project root; skip gracefully if they are absent in this environment.
        project_root = Path(__file__).parent.parent
        real_inf = project_root / "code" / "inference.py"
        real_req = project_root / "code" / "requirements.txt"
        if not real_inf.exists() or not real_req.exists():
            pytest.skip("code/inference.py or code/requirements.txt missing from repo root")

        with (
            patch("sys.argv", ["package_model",
                               "--model-path", str(model_pt),
                               "--run-name", "run_test",
                               "--output-dir", str(tmp_path / "out")]),
            patch("scripts.package_model.load_env_and_validate"),
            patch.dict(os.environ, {
                "AWS_ACCESS_KEY_ID": "key",
                "AWS_SECRET_ACCESS_KEY": "secret",
                "AWS_DEFAULT_REGION": "ap-southeast-2",
            }),
            patch("scripts.package_model.boto3.client", return_value=mock_s3),
            pytest.raises(SystemExit) as exc_info,
        ):
            from scripts.package_model import main
            main()

        assert exc_info.value.code == 0
        out = capsys.readouterr().out
        assert "s3://" in out
        assert "run_test" in out
