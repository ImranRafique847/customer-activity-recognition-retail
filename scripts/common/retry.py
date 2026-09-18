"""
Shared retry utility for AWS network operations.

Provides exponential backoff retry logic used across all scripts in the
retail-cv-analytics pipeline. The abort_multipart_on_failure context manager
(added in task 3.3) will also live here.
"""

import time
from contextlib import contextmanager
from typing import Callable, TypeVar

from botocore.exceptions import ClientError

T = TypeVar("T")


def exponential_backoff_retry(
    fn: Callable[[], T],
    max_retries: int = 3,
    base_seconds: float = 1.0,
    retryable_exceptions: tuple = (ClientError, ConnectionError, TimeoutError),
) -> T:
    """
    Call fn() with exponential backoff retry on transient failures.

    Attempt 0: call fn() immediately (no initial wait).
    On a retryable exception at attempt N: wait base_seconds * (2 ** N) seconds,
    then retry as attempt N+1.
    After max_retries attempts are exhausted, re-raise the last exception.
    Non-retryable exceptions propagate immediately without any retry.

    Args:
        fn: Zero-argument callable to invoke.
        max_retries: Maximum number of retry attempts after the initial call
                     (total calls = max_retries + 1 if all fail). Defaults to 3.
        base_seconds: Base wait interval in seconds. The wait before retry N is
                      base_seconds * (2 ** attempt), where attempt starts at 0.
                      Defaults to 1.0.
        retryable_exceptions: Tuple of exception types that should trigger a retry.
                              Defaults to (ClientError, ConnectionError, TimeoutError).

    Returns:
        The return value of fn() on success.

    Raises:
        The last caught retryable exception once max_retries are exhausted.
        Any non-retryable exception is re-raised immediately.

    Example:
        result = exponential_backoff_retry(
            lambda: s3_client.download_file(bucket, key, local_path),
            max_retries=3,
            base_seconds=1.0,
        )
        # Waits: 1s, 2s, 4s between attempts 0→1, 1→2, 2→3 if needed.
    """
    last_exception: Exception | None = None

    for attempt in range(max_retries + 1):
        try:
            return fn()
        except retryable_exceptions as exc:
            last_exception = exc
            if attempt < max_retries:
                wait_seconds = base_seconds * (2 ** attempt)
                time.sleep(wait_seconds)
            # On the final attempt fall through to re-raise below
        # Non-retryable exceptions propagate immediately (not caught here)

    # All attempts exhausted — re-raise the last retryable exception
    raise last_exception


from contextlib import contextmanager


@contextmanager
def abort_multipart_on_failure(s3_client, bucket: str, key: str):
    """
    Context manager that tracks a multipart upload and aborts it on failure.

    Usage:
        with abort_multipart_on_failure(s3_client, bucket, key) as upload_id_holder:
            # The caller initiates the multipart upload and stores the upload ID:
            # upload_id_holder["upload_id"] = response["UploadId"]
            # ... perform the upload parts ...

    On any exception inside the block:
    - If upload_id_holder["upload_id"] is set, calls s3_client.abort_multipart_upload(
        Bucket=bucket, Key=key, UploadId=upload_id_holder["upload_id"])
    - Re-raises the original exception after aborting

    On success (no exception): does nothing on exit.

    Note: For small files (<8MB) boto3 uses a simple PUT (atomic), so this
    context manager is only needed for large file uploads.

    Validates: Requirements 2.7, 5.3
    """
    upload_id_holder: dict = {"upload_id": None}
    try:
        yield upload_id_holder
    except Exception:
        upload_id = upload_id_holder.get("upload_id")
        if upload_id is not None:
            try:
                s3_client.abort_multipart_upload(
                    Bucket=bucket,
                    Key=key,
                    UploadId=upload_id,
                )
            except Exception:
                # Suppress secondary abort errors so the original exception propagates
                pass
        raise
