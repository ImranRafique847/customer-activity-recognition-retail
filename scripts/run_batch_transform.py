"""
scripts/run_batch_transform.py — SageMaker Batch Transform runner

Submits a SageMaker batch transform job, polls until completion,
and handles the terminal state.

Usage:
    python -m scripts.run_batch_transform \\
        --input-s3-uri  s3://retaildata-cv-2026/batch-input/run_001/ \\
        --output-s3-uri s3://retaildata-cv-2026/batch-output/run_001/ \\
        --model-name    retail-cv-yolo-run_001

Exit codes:
    0  — job completed successfully
    1  — job failed, stopped, timed out, or CLI/API error
"""

import argparse
import sys
import time
from datetime import datetime, timezone

import boto3

from scripts.common.env_loader import load_env_and_validate


# ---------------------------------------------------------------------------
# Job submission
# ---------------------------------------------------------------------------


def submit_batch_transform(
    sm_client,
    model_name: str,
    input_s3_uri: str,
    output_s3_uri: str,
    instance_type: str = "ml.g4dn.xlarge",
) -> str:
    """Submit a SageMaker batch transform job and return the job name.

    Tags every job with Project=retail-cv-analytics and
    ManagedBy=aws-model-deployment-spec (Req 6.8).
    Exits with code 1 on API failure without entering the polling loop.
    """
    job_name = (
        f"retail-cv-batch-"
        f"{datetime.now(tz=timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    )

    try:
        sm_client.create_transform_job(
            TransformJobName=job_name,
            ModelName=model_name,
            BatchStrategy="SingleRecord",
            TransformInput={
                "DataSource": {
                    "S3DataSource": {
                        "S3DataType": "S3Prefix",
                        "S3Uri": input_s3_uri,
                        "S3DataDistributionType": "FullyReplicated",
                    }
                },
                "ContentType": "image/jpeg",
                "SplitType": "None",
            },
            TransformOutput={
                "S3OutputPath": output_s3_uri,
                "Accept": "application/json",
                "AssembleWith": "None",
            },
            TransformResources={
                "InstanceType": instance_type,
                "InstanceCount": 1,
            },
            Tags=[
                {"Key": "Project", "Value": "retail-cv-analytics"},
                {"Key": "ManagedBy", "Value": "aws-model-deployment-spec"},
            ],
        )
    except Exception as exc:
        print(f"Error: CreateTransformJob failed: {exc}")
        sys.exit(1)

    print(f"Submitted batch transform job: {job_name}")
    return job_name


# ---------------------------------------------------------------------------
# Polling loop
# ---------------------------------------------------------------------------

_TERMINAL_STATES = {"Completed", "Failed", "Stopped"}


def poll_until_terminal(
    sm_client,
    job_name: str,
    poll_interval_seconds: int = 60,
    max_duration_seconds: int = 86400,
) -> str:
    """Poll DescribeTransformJob every poll_interval_seconds until terminal state.

    Prints status on every poll. Retries DescribeTransformJob up to 3 times
    with a fixed 30-second interval on exception (Req 6.9).
    Exits with code 1 if max_duration_seconds is exceeded.
    Returns the terminal status string.
    """
    start_time = time.time()

    while True:
        # Fetch status with fixed-interval retry on exception.
        status = _describe_with_retry(sm_client, job_name)

        ts = datetime.now(tz=timezone.utc).isoformat()
        print(f"[{ts}] Job: {job_name} | Status: {status}")

        if status in _TERMINAL_STATES:
            return status

        elapsed = time.time() - start_time
        if elapsed > max_duration_seconds:
            print(
                f"Error: polling exceeded maximum duration of "
                f"{max_duration_seconds}s for job {job_name}"
            )
            sys.exit(1)

        time.sleep(poll_interval_seconds)


def _describe_with_retry(sm_client, job_name: str) -> str:
    """Call DescribeTransformJob, retrying up to 3 times with 30-second intervals."""
    max_retries = 3
    retry_interval = 30

    for attempt in range(max_retries + 1):
        try:
            resp = sm_client.describe_transform_job(TransformJobName=job_name)
            return resp["TransformJobStatus"]
        except Exception as exc:
            print(f"Error polling DescribeTransformJob (attempt {attempt + 1}): {exc}")
            if attempt < max_retries:
                time.sleep(retry_interval)
            else:
                sys.exit(1)


# ---------------------------------------------------------------------------
# Terminal state handling
# ---------------------------------------------------------------------------


def handle_terminal_state(
    sm_client,
    job_name: str,
    status: str,
    start_time: float,
    output_s3_uri: str,
) -> None:
    """Handle the terminal state of a batch transform job.

    Completed → print output URI and duration, exit 0.
    Failed/Stopped → print failure reason, exit 1.
    """
    if status == "Completed":
        duration = time.time() - start_time
        print(f"Job completed successfully.")
        print(f"Output: {output_s3_uri}")
        print(f"Duration: {duration:.1f}s")
        sys.exit(0)
    else:
        try:
            resp = sm_client.describe_transform_job(TransformJobName=job_name)
            reason = resp.get("FailureReason") or "Failure reason not available"
        except Exception:
            reason = "Failure reason not available"

        print(f"Job {status}. Reason: {reason}")
        sys.exit(1)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="run_batch_transform",
        description="Submit and monitor a SageMaker batch transform job.",
    )
    parser.add_argument("--input-s3-uri", default=None)
    parser.add_argument("--output-s3-uri", default=None)
    parser.add_argument("--model-name", default=None)
    parser.add_argument(
        "--instance-type",
        default="ml.g4dn.xlarge",
        help="SageMaker instance type (default: ml.g4dn.xlarge)",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    # Validate required args manually for clear error messages (Req 6.2).
    missing = [
        name
        for name, val in [
            ("--input-s3-uri", args.input_s3_uri),
            ("--output-s3-uri", args.output_s3_uri),
            ("--model-name", args.model_name),
        ]
        if not val
    ]
    if missing:
        print(f"Error: the following required arguments are missing: {', '.join(missing)}")
        sys.exit(1)

    load_env_and_validate(
        ["AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_DEFAULT_REGION"]
    )

    sm_client = boto3.client("sagemaker")

    start_time = time.time()
    job_name = submit_batch_transform(
        sm_client,
        model_name=args.model_name,
        input_s3_uri=args.input_s3_uri,
        output_s3_uri=args.output_s3_uri,
        instance_type=args.instance_type,
    )

    status = poll_until_terminal(sm_client, job_name)
    handle_terminal_state(sm_client, job_name, status, start_time, args.output_s3_uri)


if __name__ == "__main__":
    main()
