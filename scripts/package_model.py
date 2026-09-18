"""
scripts/package_model.py — Model_Packager

Bundles a trained YOLO model, the SageMaker inference handler, and its
dependencies into a SageMaker-compatible ``model.tar.gz`` artifact, verifies
the archive integrity, and uploads the result to S3.

Usage examples:
  # Package and upload a model
  python -m scripts.package_model \\
      --model-path models/run_A/best.pt \\
      --run-name run_A

  # Package with a custom output directory
  python -m scripts.package_model \\
      --model-path models/run_A/best.pt \\
      --run-name run_A \\
      --output-dir /tmp/artifacts
"""

import argparse
import io
import json
import os
import shutil
import sys
import tarfile
import tempfile
from pathlib import Path

import boto3

# validate_run_name is implemented in task 3.1 (scripts/s3_sync.py).
# Imported here for use in the CLI validation.
from scripts.s3_sync import validate_run_name  # noqa: F401

from scripts.common.env_loader import load_env_and_validate
from scripts.common.retry import abort_multipart_on_failure


# ---------------------------------------------------------------------------
# Training summary validation  (Requirement 9.2)
# ---------------------------------------------------------------------------


def validate_training_summary(summary_path: Path) -> dict:
    """Load and validate ``training_summary.json``.

    Parameters
    ----------
    summary_path:
        Path to the ``training_summary.json`` file produced by the local
        training pipeline.

    Returns
    -------
    dict
        The parsed JSON document when all required fields are present and
        have the correct types/values.

    Raises
    ------
    FileNotFoundError
        When ``summary_path`` does not exist on the local filesystem.
    ValueError
        When the file contains invalid JSON, or when any required field is
        absent, has the wrong type, or violates a value constraint.  The
        error message always identifies the specific field.

    Required fields and constraints
    --------------------------------
    - ``model_architecture`` — non-empty ``str``
    - ``input_size``         — positive ``int``
    - ``num_classes``        — ``int`` > 0
    - ``class_names``        — ``list[str]`` with ``len == num_classes``
    - ``dataset_version``    — non-empty ``str``
    - ``training_device``    — non-empty ``str``
    """
    if not summary_path.exists():
        raise FileNotFoundError(summary_path)

    raw = summary_path.read_text(encoding="utf-8")
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"training_summary.json contains invalid JSON: {exc}"
        ) from exc

    # --- model_architecture ---
    if "model_architecture" not in data:
        raise ValueError(
            "training_summary.json: field 'model_architecture' is missing"
        )
    if not isinstance(data["model_architecture"], str):
        raise ValueError(
            "training_summary.json: field 'model_architecture' must be a str"
        )
    if not data["model_architecture"]:
        raise ValueError(
            "training_summary.json: field 'model_architecture' must be a non-empty str"
        )

    # --- input_size ---
    if "input_size" not in data:
        raise ValueError(
            "training_summary.json: field 'input_size' is missing"
        )
    # JSON integers deserialise as int; booleans are a subclass of int in Python,
    # so we reject them explicitly to match the intent of "positive int".
    if not isinstance(data["input_size"], int) or isinstance(data["input_size"], bool):
        raise ValueError(
            "training_summary.json: field 'input_size' must be an int"
        )
    if data["input_size"] <= 0:
        raise ValueError(
            "training_summary.json: field 'input_size' must be a positive int"
        )

    # --- num_classes ---
    if "num_classes" not in data:
        raise ValueError(
            "training_summary.json: field 'num_classes' is missing"
        )
    if not isinstance(data["num_classes"], int) or isinstance(data["num_classes"], bool):
        raise ValueError(
            "training_summary.json: field 'num_classes' must be an int"
        )
    if data["num_classes"] <= 0:
        raise ValueError(
            "training_summary.json: field 'num_classes' must be an int > 0"
        )

    # --- class_names ---
    if "class_names" not in data:
        raise ValueError(
            "training_summary.json: field 'class_names' is missing"
        )
    if not isinstance(data["class_names"], list):
        raise ValueError(
            "training_summary.json: field 'class_names' must be a list of str"
        )
    if not all(isinstance(name, str) for name in data["class_names"]):
        raise ValueError(
            "training_summary.json: field 'class_names' must be a list of str"
        )
    if len(data["class_names"]) != data["num_classes"]:
        raise ValueError(
            f"training_summary.json: field 'class_names' has length "
            f"{len(data['class_names'])} but 'num_classes' is {data['num_classes']}"
        )

    # --- dataset_version ---
    if "dataset_version" not in data:
        raise ValueError(
            "training_summary.json: field 'dataset_version' is missing"
        )
    if not isinstance(data["dataset_version"], str):
        raise ValueError(
            "training_summary.json: field 'dataset_version' must be a str"
        )
    if not data["dataset_version"]:
        raise ValueError(
            "training_summary.json: field 'dataset_version' must be a non-empty str"
        )

    # --- training_device ---
    if "training_device" not in data:
        raise ValueError(
            "training_summary.json: field 'training_device' is missing"
        )
    if not isinstance(data["training_device"], str):
        raise ValueError(
            "training_summary.json: field 'training_device' must be a str"
        )
    if not data["training_device"]:
        raise ValueError(
            "training_summary.json: field 'training_device' must be a non-empty str"
        )

    return data


# ---------------------------------------------------------------------------
# Archive creation  (Requirement 5.1, 5.5, 9.3)
# ---------------------------------------------------------------------------


def create_model_archive(
    model_path: Path,
    inference_py: Path,
    requirements_txt: Path,
    training_summary: dict,
    output_dir: Path,
) -> Path:
    """Bundle model artefacts into a SageMaker-compatible ``model.tar.gz``.

    Parameters
    ----------
    model_path:
        Path to ``best.pt``.
    inference_py:
        Path to ``code/inference.py``.
    requirements_txt:
        Path to ``code/requirements.txt``.
    training_summary:
        Validated training-summary dict (returned by
        :func:`validate_training_summary`).
    output_dir:
        Directory where the resulting ``model.tar.gz`` will be written.

    Returns
    -------
    Path
        Absolute path to the created ``model.tar.gz``.

    Raises
    ------
    FileNotFoundError
        When any required source file does not exist on the local filesystem.
    ValueError
        When archive integrity verification fails after creation.
    """
    # --- Requirement 5.5: check all required source files exist first ---
    for src in (model_path, inference_py, requirements_txt):
        if not Path(src).exists():
            raise FileNotFoundError(
                f"Required file not found: {src}"
            )

    # --- Requirement 9.3: generate model_config.json from training_summary ---
    model_config = {
        "num_classes": training_summary["num_classes"],
        "class_names": training_summary["class_names"],
        "input_size": training_summary["input_size"],
    }
    model_config_json = json.dumps(model_config, indent=2)

    # --- Requirement 5.1: create model.tar.gz with SageMaker layout ---
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    archive_path = output_dir / "model.tar.gz"

    with tarfile.open(archive_path, "w:gz") as tar:
        # best.pt at archive root
        tar.add(model_path, arcname="best.pt")
        # code/inference.py
        tar.add(inference_py, arcname="code/inference.py")
        # code/requirements.txt
        tar.add(requirements_txt, arcname="code/requirements.txt")
        # code/model_config.json — generated from training_summary
        config_bytes = model_config_json.encode("utf-8")
        info = tarfile.TarInfo(name="code/model_config.json")
        info.size = len(config_bytes)
        tar.addfile(info, io.BytesIO(config_bytes))

    # --- Requirement 5.4: verify integrity before returning ---
    verify_archive_integrity(archive_path)

    return archive_path.resolve()


# ---------------------------------------------------------------------------
# Archive integrity verification  (Requirement 5.4)
# ---------------------------------------------------------------------------


def verify_archive_integrity(archive_path: Path) -> None:
    """Verify that a ``model.tar.gz`` archive contains all required files at
    their minimum sizes.

    Parameters
    ----------
    archive_path:
        Path to the ``model.tar.gz`` archive to inspect.

    Raises
    ------
    ValueError
        When a required file is absent or below its minimum size threshold.
    """
    tempdir = tempfile.mkdtemp()
    try:
        with tarfile.open(archive_path) as tar:
            tar.extractall(tempdir)

        best_pt = Path(tempdir) / "best.pt"
        if best_pt.stat().st_size < 1024:
            raise ValueError("best.pt is absent or too small in archive")

        inference_py = Path(tempdir) / "code" / "inference.py"
        if inference_py.stat().st_size < 1:
            raise ValueError("code/inference.py is absent or empty in archive")
    finally:
        shutil.rmtree(tempdir)


# ---------------------------------------------------------------------------
# S3 upload  (Requirement 5.3, 2.7)
# ---------------------------------------------------------------------------


def upload_archive(
    archive_path: Path,
    bucket: str,
    run_name: str,
    s3_client,
) -> str:
    """Upload ``model.tar.gz`` to S3 with partial-upload protection.

    Parameters
    ----------
    archive_path:
        Local path to the ``model.tar.gz`` archive.
    bucket:
        S3 bucket name (e.g. ``retaildata-cv-2026``).
    run_name:
        Run name used to construct the S3 key
        ``models/{run_name}/model.tar.gz``.
    s3_client:
        A boto3 S3 client instance.

    Returns
    -------
    str
        Full S3 URI of the uploaded object
        (``s3://{bucket}/models/{run_name}/model.tar.gz``).

    Raises
    ------
    RuntimeError
        When the S3 upload fails for any reason.
    """
    key = f"models/{run_name}/model.tar.gz"
    try:
        with abort_multipart_on_failure(s3_client, bucket, key):
            s3_client.upload_file(str(archive_path), bucket, key)
    except Exception as exc:
        raise RuntimeError(
            f"Failed to upload {archive_path} to s3://{bucket}/{key}: {exc}"
        ) from exc
    return f"s3://{bucket}/{key}"


# ---------------------------------------------------------------------------
# CLI entrypoint
# ---------------------------------------------------------------------------


def main() -> None:
    """Parse arguments and run the model packaging pipeline."""
    parser = argparse.ArgumentParser(
        prog="package_model",
        description=(
            "Bundle a trained YOLO model into a SageMaker-compatible "
            "model.tar.gz archive and upload it to S3."
        ),
    )
    parser.add_argument(
        "--model-path",
        metavar="PATH",
        required=True,
        help="Path to the local best.pt file.",
    )
    parser.add_argument(
        "--run-name",
        metavar="NAME",
        required=True,
        help=(
            "Run name used to build the S3 key "
            "(alphanumeric, underscores, hyphens; max 128 chars)."
        ),
    )
    parser.add_argument(
        "--output-dir",
        metavar="DIR",
        default=None,
        help="Directory for the generated model.tar.gz (defaults to model directory).",
    )

    args = parser.parse_args()

    # Validate --run-name (Requirement 5.10)
    try:
        validate_run_name(args.run_name)
    except ValueError as exc:
        print(f"Error: {exc}")
        sys.exit(1)

    # Load and validate AWS credentials from the project .env file (Requirement 9.6)
    load_env_and_validate(
        ["AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_DEFAULT_REGION"]
    )

    model_path = Path(args.model_path)
    output_dir = Path(args.output_dir) if args.output_dir else model_path.parent

    # training_summary.json lives next to best.pt (Requirement 9.2)
    summary_path = model_path.parent / "training_summary.json"

    # Inference handler and requirements live under the project root's code/ dir
    project_root = Path(__file__).parent.parent
    inference_py = project_root / "code" / "inference.py"
    requirements_txt = project_root / "code" / "requirements.txt"

    # Validate training_summary.json (Requirement 9.2)
    training_summary = validate_training_summary(summary_path)

    # Create the archive (Requirements 5.1, 5.4, 5.5, 9.3)
    archive_path = create_model_archive(
        model_path=model_path,
        inference_py=inference_py,
        requirements_txt=requirements_txt,
        training_summary=training_summary,
        output_dir=output_dir,
    )

    # Upload to S3 (Requirement 5.3)
    s3_client = boto3.client(
        "s3",
        aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"],
        region_name=os.environ["AWS_DEFAULT_REGION"],
    )
    s3_uri = upload_archive(
        archive_path=archive_path,
        bucket="retaildata-cv-2026",
        run_name=args.run_name,
        s3_client=s3_client,
    )

    print(s3_uri)
    sys.exit(0)


if __name__ == "__main__":
    main()
