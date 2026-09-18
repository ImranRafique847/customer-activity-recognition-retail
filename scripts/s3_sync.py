"""
scripts/s3_sync.py — S3_Sync_Tool

Two operation modes:
  download  Pull labeled image subsets or raw video from S3 to local storage.
  upload    Push a trained .pt model file and run_metadata.json to S3.

Usage:
  # Download mode
  python -m scripts.s3_sync [--source images|videos] [--subset SUBSET]
                             [--max-files N] [--store-ids ID1,ID2]
                             [--local-dest PATH]

  # Upload mode
  python -m scripts.s3_sync --upload-model PATH [--run-name NAME]
                             [--overwrite]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, TypeVar

# ---------------------------------------------------------------------------
# Version
# ---------------------------------------------------------------------------

__version__ = "1.0.0"

# ---------------------------------------------------------------------------
# S3 / bucket constants
# ---------------------------------------------------------------------------

BUCKET = "retaildata-cv-2026"

# ---------------------------------------------------------------------------
# validate_run_name / generate_run_name  (Task 3.1 — Req 2.2, 2.3)
# ---------------------------------------------------------------------------

_RUN_NAME_PATTERN = re.compile(r"^[a-zA-Z0-9_\-]+$")
_RUN_NAME_MAX_LEN = 128


def validate_run_name(name: str) -> None:
    """Validate that *name* is a legal run name.

    Raises:
        ValueError: if *name* contains characters outside ``[a-zA-Z0-9_-]``
                    or is longer than 128 characters.

    Validates: Requirements 2.2, 5.10
    """
    if len(name) > _RUN_NAME_MAX_LEN:
        raise ValueError(
            f"run_name exceeds the maximum length of {_RUN_NAME_MAX_LEN} characters "
            f"(got {len(name)} characters)."
        )
    if not _RUN_NAME_PATTERN.match(name):
        raise ValueError(
            "run_name contains invalid characters. "
            "Only alphanumeric characters, underscores (_), and hyphens (-) are allowed."
        )


def generate_run_name() -> str:
    """Generate a timestamped run name of the form ``run_{YYYYMMDDTHHmmssZ}``.

    Uses ``datetime.utcnow()`` for the timestamp and prints the generated name
    to stdout so operators can record it.

    Returns:
        A string like ``run_20250115T103000Z``.

    Validates: Requirement 2.3
    """
    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    name = f"run_{ts}"
    print(name)
    return name


# ---------------------------------------------------------------------------
# ETag / checksum utility  (Task 2.2 — Req 1.4)
# ---------------------------------------------------------------------------


def compute_etag_match(local_path: Path, s3_etag: str) -> bool:
    """Return True when the local file's MD5 matches *s3_etag*.

    Conservative rules:
    - Returns False if the file does not exist.
    - Returns False for multipart ETags (those containing ``'-'``) — always
      triggers a re-download for safety.
    - Strips surrounding double-quotes from *s3_etag* before comparing.

    Validates: Requirement 1.4
    """
    # Strip surrounding quotes from ETag value
    clean_etag = s3_etag.strip('"')

    # Conservative: multipart uploads use "hash-partcount" format; always re-download
    if "-" in clean_etag:
        return False

    if not local_path.exists():
        return False

    # Compute MD5 of the local file in chunks to support large files
    md5 = hashlib.md5()
    try:
        with local_path.open("rb") as fh:
            for chunk in iter(lambda: fh.read(8192), b""):
                md5.update(chunk)
    except OSError:
        return False

    return md5.hexdigest() == clean_etag


# ---------------------------------------------------------------------------
# Download mode  (Tasks 2.1, 2.3, 2.4 — Req 1.x)
# ---------------------------------------------------------------------------


def download_dataset(
    bucket: str,
    source: str,
    subset: str | None,
    max_files: int | None,
    store_ids: list[str] | None,
    local_dest: Path,
) -> dict[str, Any]:
    """Download image/label pairs or video files from S3 to *local_dest*.

    - Skips files whose S3 ETag matches the local MD5 checksum.
    - Retries each file up to 3 times with exponential backoff (base=1s).
    - Writes ``download_manifest.json`` to *local_dest* on completion.

    Returns the manifest dict.
    Raises SystemExit(1) on invalid params or exhausted retries.

    Validates: Requirements 1.1, 1.3–1.8, 1.10
    """
    import boto3
    from botocore.exceptions import ClientError

    from scripts.common.env_loader import load_env_and_validate
    from scripts.common.retry import exponential_backoff_retry

    load_env_and_validate(
        ["AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_DEFAULT_REGION"]
    )

    s3 = boto3.client("s3")

    # Determine S3 prefix(es) to list
    if source == "images":
        if subset:
            prefix = f"extracted_images_and_labels/data/{subset}/"
        else:
            prefix = "extracted_images_and_labels/data/"
        prefixes = [prefix]
    else:  # videos
        if store_ids:
            prefixes = [f"extracted_videos/{sid}/" for sid in store_ids]
        else:
            prefixes = ["extracted_videos/"]

    local_dest.mkdir(parents=True, exist_ok=True)
    invocation_timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    downloaded_files: list[dict] = []
    skipped_count = 0
    download_count = 0
    limit_reached = False

    for prefix in prefixes:
        # Verify prefix exists in S3
        paginator = s3.get_paginator("list_objects_v2")
        pages = paginator.paginate(Bucket=bucket, Prefix=prefix, MaxKeys=1)
        first_page = next(iter(pages), None)
        if not first_page or first_page.get("KeyCount", 0) == 0:
            print(
                f"Error: S3 path does not exist: s3://{bucket}/{prefix}"
            )
            sys.exit(1)

        # List all objects under prefix
        all_pages = paginator.paginate(Bucket=bucket, Prefix=prefix)
        for page in all_pages:
            if limit_reached:
                break
            for obj in page.get("Contents", []):
                if limit_reached:
                    break
                key = obj["Key"]
                etag = obj.get("ETag", "")
                size_bytes = obj.get("Size", 0)

                # Compute local path (preserve relative key structure)
                rel_path = key.lstrip("/")
                local_file = local_dest / rel_path
                local_file.parent.mkdir(parents=True, exist_ok=True)

                # ETag-based skip check
                if compute_etag_match(local_file, etag):
                    skipped_count += 1
                    continue

                # Apply max_files limit
                if max_files is not None and download_count >= max_files:
                    limit_reached = True
                    break

                # Download with retry
                def _download(k=key, lf=local_file):
                    s3.download_file(bucket, k, str(lf))

                try:
                    exponential_backoff_retry(_download, max_retries=3, base_seconds=1.0)
                except Exception as exc:
                    print(f"Error: Download failed for s3://{bucket}/{key}: {exc}")
                    sys.exit(1)

                downloaded_files.append({"s3_key": key, "size_bytes": size_bytes})
                download_count += 1

    # Write manifest
    manifest = {
        "invocation_timestamp": invocation_timestamp,
        "source": source,
        "subset": subset,
        "total_files_downloaded": download_count,
        "total_files_skipped": skipped_count,
        "files": downloaded_files,
    }
    manifest_path = local_dest / "download_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


# ---------------------------------------------------------------------------
# Upload mode  (Tasks 3.2–3.4 — Req 2.x)
# ---------------------------------------------------------------------------


def upload_model(
    local_pt_path: Path,
    bucket: str,
    run_name: str | None,
    overwrite: bool = False,
) -> str:
    """Upload *local_pt_path* to ``s3://{bucket}/models/{run_name}/best.pt``.

    - Validates .pt extension and file existence.
    - Validates/generates run_name.
    - Checks for existing S3 object; aborts unless overwrite=True.
    - Uploads best.pt, verifies size match, uploads run_metadata.json.

    Returns the S3 URI of the uploaded model.
    Raises SystemExit(1) on any validation or upload failure.

    Validates: Requirements 2.1, 2.4, 2.5, 2.7
    """
    import boto3
    from botocore.exceptions import ClientError

    from scripts.common.env_loader import load_env_and_validate
    from scripts.common.retry import abort_multipart_on_failure

    load_env_and_validate(
        ["AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_DEFAULT_REGION"]
    )

    # Validate local path
    if not local_pt_path.exists():
        print(f"Error: File not found: {local_pt_path}")
        sys.exit(1)
    if local_pt_path.suffix.lower() != ".pt":
        print(
            f"Error: Invalid file extension '{local_pt_path.suffix}'. "
            "Expected a .pt file."
        )
        sys.exit(1)

    # Resolve run name
    if run_name is None:
        run_name = generate_run_name()
    else:
        try:
            validate_run_name(run_name)
        except ValueError as exc:
            print(f"Error: {exc}")
            sys.exit(1)

    s3 = boto3.client("s3")
    model_key = f"models/{run_name}/best.pt"

    # Overwrite guard
    try:
        s3.head_object(Bucket=bucket, Key=model_key)
        # Object exists
        if not overwrite:
            print(
                f"Error: S3 object already exists: s3://{bucket}/{model_key}. "
                "Use --overwrite to replace it."
            )
            sys.exit(1)
    except ClientError as exc:
        if exc.response["Error"]["Code"] not in ("404", "NoSuchKey"):
            print(f"Error checking S3 object: {exc}")
            sys.exit(1)
        # 404 means it doesn't exist — proceed with upload

    # Upload model file
    local_size = local_pt_path.stat().st_size
    try:
        s3.upload_file(str(local_pt_path), bucket, model_key)
    except Exception as exc:
        print(f"Error: Failed to upload model: {exc}")
        sys.exit(1)

    # Verify size
    try:
        head = s3.head_object(Bucket=bucket, Key=model_key)
        s3_size = head["ContentLength"]
    except Exception as exc:
        print(f"Error: Could not verify upload size: {exc}")
        sys.exit(1)

    if s3_size != local_size:
        print(
            f"Error: Upload size mismatch for s3://{bucket}/{model_key}. "
            f"Local: {local_size} bytes, S3: {s3_size} bytes."
        )
        sys.exit(1)

    s3_uri = f"s3://{bucket}/{model_key}"
    print(s3_uri)

    # Upload run_metadata.json
    upload_timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    metadata = {
        "run_name": run_name,
        "upload_timestamp": upload_timestamp,
        "local_file_size_bytes": local_size,
        "python_script_version": __version__,
    }
    metadata_key = f"models/{run_name}/run_metadata.json"
    metadata_body = json.dumps(metadata, indent=2).encode("utf-8")
    try:
        s3.put_object(Bucket=bucket, Key=metadata_key, Body=metadata_body)
    except Exception as exc:
        print(
            f"Error: Failed to upload run_metadata.json to "
            f"s3://{bucket}/{metadata_key}: {exc}"
        )
        sys.exit(1)

    return s3_uri


# ---------------------------------------------------------------------------
# CLI argument parsing and entry point
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="s3_sync",
        description="S3_Sync_Tool: download training data or upload model artifacts.",
    )

    # ---- Download mode flags ----
    parser.add_argument(
        "--source",
        choices=["images", "videos"],
        default="images",
        help="Data source type (default: images).",
    )
    parser.add_argument(
        "--subset",
        default=None,
        help="Activity-category subfolder to download (images mode).",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=None,
        metavar="N",
        help="Maximum number of files to download (must be > 0).",
    )
    parser.add_argument(
        "--store-ids",
        default=None,
        help="Comma-separated store folder names (videos mode).",
    )
    parser.add_argument(
        "--local-dest",
        type=Path,
        default=Path("data"),
        help="Local destination directory (default: data/).",
    )

    # ---- Upload mode flags ----
    parser.add_argument(
        "--upload-model",
        type=Path,
        default=None,
        metavar="PATH",
        help="Local path to a .pt model file to upload.",
    )
    parser.add_argument(
        "--run-name",
        default=None,
        help=(
            "Run name for the upload (alphanumeric, underscores, hyphens; max 128 chars). "
            "Auto-generated when omitted."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        default=False,
        help="Allow overwriting an existing S3 model object.",
    )

    return parser


def main(argv: list[str] | None = None) -> None:
    """CLI entry point for s3_sync."""
    parser = _build_parser()

    # argparse does not natively support the custom --source validation message
    # (it would print its own "invalid choice" message), so we intercept it.
    # Also handle --max-files <= 0 validation here.
    args, _ = parser.parse_known_args(argv)

    # Re-parse fully to catch unknown args (let argparse handle them)
    args = parser.parse_args(argv)

    # ---- Custom validation: --max-files ----
    if args.max_files is not None and args.max_files <= 0:
        print(
            f"Error: --max-files must be a positive integer greater than 0 "
            f"(got {args.max_files})."
        )
        sys.exit(1)

    # ---- Dispatch ----
    if args.upload_model is not None:
        # Upload mode
        if args.run_name is not None:
            try:
                validate_run_name(args.run_name)
            except ValueError as exc:
                print(f"Error: {exc}")
                sys.exit(1)

        upload_model(
            local_pt_path=args.upload_model,
            bucket=BUCKET,
            run_name=args.run_name,
            overwrite=args.overwrite,
        )
    else:
        # Download mode
        store_ids: list[str] | None = None
        if args.store_ids:
            store_ids = [s.strip() for s in args.store_ids.split(",") if s.strip()]

        download_dataset(
            bucket=BUCKET,
            source=args.source,
            subset=args.subset,
            max_files=args.max_files,
            store_ids=store_ids,
            local_dest=args.local_dest,
        )


if __name__ == "__main__":
    main()
