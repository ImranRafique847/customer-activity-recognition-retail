"""
Tests for scripts/run_batch_transform.py (Batch Transform).

Covers tasks 11.1–11.6:
  - submit_batch_transform: required tags, default instance type, API failure
  - poll_until_terminal: status printing, timeout, DescribeTransformJob retry
  - handle_terminal_state: Completed (exit 0), Failed/Stopped (exit 1 + reason)

Requirements: 6.2, 6.3, 6.4, 6.5, 6.6, 6.7, 6.8, 6.9, 6.10
"""

import time
from unittest.mock import MagicMock, call, patch

import pytest
from botocore.exceptions import ClientError

from scripts.run_batch_transform import (
    build_parser,
    handle_terminal_state,
    poll_until_terminal,
    submit_batch_transform,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_FAKE_JOB_NAME = "retail-cv-batch-20250115T103000Z"


def _make_sm_client(statuses=None, failure_reason=None):
    """Return a mock SageMaker client.

    statuses: list of TransformJobStatus strings returned on successive
              DescribeTransformJob calls.
    failure_reason: value for FailureReason in the describe response.
    """
    client = MagicMock()
    client.create_transform_job.return_value = {}

    if statuses is None:
        statuses = ["Completed"]

    responses = [{"TransformJobStatus": s} for s in statuses]
    if failure_reason is not None:
        for r in responses:
            r["FailureReason"] = failure_reason

    client.describe_transform_job.side_effect = responses
    return client


# ---------------------------------------------------------------------------
# submit_batch_transform
# ---------------------------------------------------------------------------


class TestSubmitBatchTransform:
    """Tests for submit_batch_transform (Req 6.2, 6.3, 6.8, 6.10)."""

    def test_returns_job_name_string(self):
        sm = _make_sm_client()
        job_name = submit_batch_transform(
            sm,
            model_name="my-model",
            input_s3_uri="s3://bucket/input/",
            output_s3_uri="s3://bucket/output/",
        )
        assert isinstance(job_name, str)
        assert job_name.startswith("retail-cv-batch-")

    def test_job_name_contains_timestamp(self):
        """Job name follows retail-cv-batch-{timestamp} format."""
        sm = _make_sm_client()
        job_name = submit_batch_transform(
            sm,
            model_name="my-model",
            input_s3_uri="s3://bucket/input/",
            output_s3_uri="s3://bucket/output/",
        )
        # Should be "retail-cv-batch-" followed by a UTC timestamp
        parts = job_name.split("retail-cv-batch-")
        assert len(parts) == 2
        assert len(parts[1]) > 0

    def test_calls_create_transform_job(self):
        sm = _make_sm_client()
        submit_batch_transform(
            sm,
            model_name="my-model",
            input_s3_uri="s3://bucket/input/",
            output_s3_uri="s3://bucket/output/",
        )
        sm.create_transform_job.assert_called_once()

    def test_passes_model_name(self):
        sm = _make_sm_client()
        submit_batch_transform(
            sm,
            model_name="retail-cv-yolo-v1",
            input_s3_uri="s3://bucket/input/",
            output_s3_uri="s3://bucket/output/",
        )
        kwargs = sm.create_transform_job.call_args.kwargs
        assert kwargs["ModelName"] == "retail-cv-yolo-v1"

    def test_passes_input_s3_uri(self):
        sm = _make_sm_client()
        submit_batch_transform(
            sm,
            model_name="m",
            input_s3_uri="s3://my-bucket/input/",
            output_s3_uri="s3://my-bucket/output/",
        )
        kwargs = sm.create_transform_job.call_args.kwargs
        s3_uri = kwargs["TransformInput"]["DataSource"]["S3DataSource"]["S3Uri"]
        assert s3_uri == "s3://my-bucket/input/"

    def test_passes_output_s3_uri(self):
        sm = _make_sm_client()
        submit_batch_transform(
            sm,
            model_name="m",
            input_s3_uri="s3://my-bucket/input/",
            output_s3_uri="s3://my-bucket/output/",
        )
        kwargs = sm.create_transform_job.call_args.kwargs
        assert kwargs["TransformOutput"]["S3OutputPath"] == "s3://my-bucket/output/"

    def test_default_instance_type_is_ml_g4dn_xlarge(self):
        """Req 6.3: default instance type must be ml.g4dn.xlarge."""
        sm = _make_sm_client()
        submit_batch_transform(
            sm,
            model_name="m",
            input_s3_uri="s3://b/in/",
            output_s3_uri="s3://b/out/",
        )
        kwargs = sm.create_transform_job.call_args.kwargs
        assert kwargs["TransformResources"]["InstanceType"] == "ml.g4dn.xlarge"

    def test_custom_instance_type_passed_through(self):
        sm = _make_sm_client()
        submit_batch_transform(
            sm,
            model_name="m",
            input_s3_uri="s3://b/in/",
            output_s3_uri="s3://b/out/",
            instance_type="ml.p3.2xlarge",
        )
        kwargs = sm.create_transform_job.call_args.kwargs
        assert kwargs["TransformResources"]["InstanceType"] == "ml.p3.2xlarge"

    def test_includes_project_tag(self):
        """Req 6.8: Tags must include Project=retail-cv-analytics."""
        sm = _make_sm_client()
        submit_batch_transform(
            sm,
            model_name="m",
            input_s3_uri="s3://b/in/",
            output_s3_uri="s3://b/out/",
        )
        kwargs = sm.create_transform_job.call_args.kwargs
        tags = {t["Key"]: t["Value"] for t in kwargs["Tags"]}
        assert tags.get("Project") == "retail-cv-analytics"

    def test_includes_managed_by_tag(self):
        """Req 6.8: Tags must include ManagedBy=aws-model-deployment-spec."""
        sm = _make_sm_client()
        submit_batch_transform(
            sm,
            model_name="m",
            input_s3_uri="s3://b/in/",
            output_s3_uri="s3://b/out/",
        )
        kwargs = sm.create_transform_job.call_args.kwargs
        tags = {t["Key"]: t["Value"] for t in kwargs["Tags"]}
        assert tags.get("ManagedBy") == "aws-model-deployment-spec"

    def test_batch_strategy_is_single_record(self):
        sm = _make_sm_client()
        submit_batch_transform(
            sm, model_name="m",
            input_s3_uri="s3://b/in/",
            output_s3_uri="s3://b/out/",
        )
        kwargs = sm.create_transform_job.call_args.kwargs
        assert kwargs["BatchStrategy"] == "SingleRecord"

    def test_split_type_is_none(self):
        """TransformInput.SplitType must be 'None' (string)."""
        sm = _make_sm_client()
        submit_batch_transform(
            sm, model_name="m",
            input_s3_uri="s3://b/in/",
            output_s3_uri="s3://b/out/",
        )
        kwargs = sm.create_transform_job.call_args.kwargs
        assert kwargs["TransformInput"]["SplitType"] == "None"

    def test_api_failure_exits_nonzero(self):
        """Req 6.10: CreateTransformJob failure must exit 1 without polling."""
        sm = MagicMock()
        sm.create_transform_job.side_effect = Exception("API error")

        with pytest.raises(SystemExit) as exc_info:
            submit_batch_transform(
                sm, model_name="m",
                input_s3_uri="s3://b/in/",
                output_s3_uri="s3://b/out/",
            )
        assert exc_info.value.code != 0

    def test_api_failure_prints_error(self, capsys):
        """Req 6.10: API error is printed before exit."""
        sm = MagicMock()
        sm.create_transform_job.side_effect = Exception("detailed API error msg")

        with pytest.raises(SystemExit):
            submit_batch_transform(
                sm, model_name="m",
                input_s3_uri="s3://b/in/",
                output_s3_uri="s3://b/out/",
            )
        captured = capsys.readouterr()
        assert "error" in captured.out.lower() or "detailed API error msg" in captured.out


# ---------------------------------------------------------------------------
# poll_until_terminal
# ---------------------------------------------------------------------------


class TestPollUntilTerminal:
    """Tests for poll_until_terminal (Req 6.4, 6.5, 6.9)."""

    def test_returns_completed_status(self):
        sm = _make_sm_client(statuses=["Completed"])
        with patch("time.sleep"):
            status = poll_until_terminal(sm, _FAKE_JOB_NAME, poll_interval_seconds=1)
        assert status == "Completed"

    def test_returns_failed_status(self):
        sm = _make_sm_client(statuses=["Failed"])
        with patch("time.sleep"):
            status = poll_until_terminal(sm, _FAKE_JOB_NAME, poll_interval_seconds=1)
        assert status == "Failed"

    def test_returns_stopped_status(self):
        sm = _make_sm_client(statuses=["Stopped"])
        with patch("time.sleep"):
            status = poll_until_terminal(sm, _FAKE_JOB_NAME, poll_interval_seconds=1)
        assert status == "Stopped"

    def test_polls_until_terminal(self):
        """Polls through InProgress statuses until terminal."""
        sm = _make_sm_client(statuses=["InProgress", "InProgress", "Completed"])
        with patch("time.sleep"):
            status = poll_until_terminal(sm, _FAKE_JOB_NAME, poll_interval_seconds=1)
        assert status == "Completed"
        assert sm.describe_transform_job.call_count == 3

    def test_prints_status_on_each_poll(self, capsys):
        """Req 6.4: current status and job name are printed on each poll."""
        sm = _make_sm_client(statuses=["InProgress", "Completed"])
        with patch("time.sleep"):
            poll_until_terminal(sm, _FAKE_JOB_NAME, poll_interval_seconds=1)
        captured = capsys.readouterr()
        assert _FAKE_JOB_NAME in captured.out
        # Two polls happened — both printed
        assert captured.out.count(_FAKE_JOB_NAME) >= 2

    def test_timeout_exits_nonzero(self):
        """Req 6.4: exits 1 when total polling duration exceeds max_duration_seconds."""
        sm = MagicMock()
        # Always return InProgress so we never reach terminal state
        sm.describe_transform_job.return_value = {"TransformJobStatus": "InProgress"}

        with patch("time.sleep"), pytest.raises(SystemExit) as exc_info:
            poll_until_terminal(
                sm,
                _FAKE_JOB_NAME,
                poll_interval_seconds=1,
                max_duration_seconds=0,  # immediately exceeded
            )
        assert exc_info.value.code != 0

    def test_describe_exception_triggers_retry(self):
        """Req 6.9: DescribeTransformJob exceptions are retried up to 3 times."""
        sm = MagicMock()
        # Fail once then succeed
        sm.describe_transform_job.side_effect = [
            Exception("transient"),
            {"TransformJobStatus": "Completed"},
        ]
        with patch("time.sleep"):
            status = poll_until_terminal(sm, _FAKE_JOB_NAME, poll_interval_seconds=1)
        assert status == "Completed"
        assert sm.describe_transform_job.call_count == 2

    def test_describe_exception_exhausted_retries_exits_nonzero(self):
        """Req 6.9: exits non-zero after all 3 retries fail."""
        sm = MagicMock()
        # Always raise
        sm.describe_transform_job.side_effect = Exception("persistent error")

        with patch("time.sleep"), pytest.raises(SystemExit) as exc_info:
            poll_until_terminal(sm, _FAKE_JOB_NAME, poll_interval_seconds=1)
        assert exc_info.value.code != 0

    def test_describe_exception_retry_uses_fixed_30s_interval(self):
        """Req 6.9: retry interval for describe exceptions is fixed at 30 s."""
        sm = MagicMock()
        sm.describe_transform_job.side_effect = [
            Exception("err1"),
            Exception("err2"),
            {"TransformJobStatus": "Completed"},
        ]

        sleep_calls = []
        with patch("time.sleep", side_effect=lambda s: sleep_calls.append(s)):
            poll_until_terminal(sm, _FAKE_JOB_NAME, poll_interval_seconds=60)

        # The retry sleeps should be exactly 30 s each (not exponential)
        retry_sleeps = [s for s in sleep_calls if s == 30]
        assert len(retry_sleeps) >= 2

    def test_prints_exception_message_on_describe_error(self, capsys):
        """Req 6.9: exception message is printed on retry."""
        sm = MagicMock()
        sm.describe_transform_job.side_effect = [
            Exception("network_failure_sentinel"),
            {"TransformJobStatus": "Completed"},
        ]
        with patch("time.sleep"):
            poll_until_terminal(sm, _FAKE_JOB_NAME, poll_interval_seconds=1)
        captured = capsys.readouterr()
        assert "network_failure_sentinel" in captured.out


# ---------------------------------------------------------------------------
# handle_terminal_state
# ---------------------------------------------------------------------------


class TestHandleTerminalState:
    """Tests for handle_terminal_state (Req 6.6, 6.7)."""

    def test_completed_exits_zero(self):
        """Req 6.6: Completed state exits with code 0."""
        sm = _make_sm_client()
        with pytest.raises(SystemExit) as exc_info:
            handle_terminal_state(
                sm, _FAKE_JOB_NAME, "Completed",
                start_time=time.time() - 10,
                output_s3_uri="s3://bucket/output/",
            )
        assert exc_info.value.code == 0

    def test_completed_prints_output_uri(self, capsys):
        """Req 6.6: output S3 URI is printed on Completed."""
        sm = _make_sm_client()
        with pytest.raises(SystemExit):
            handle_terminal_state(
                sm, _FAKE_JOB_NAME, "Completed",
                start_time=time.time() - 5,
                output_s3_uri="s3://my-bucket/output/",
            )
        captured = capsys.readouterr()
        assert "s3://my-bucket/output/" in captured.out

    def test_completed_prints_duration(self, capsys):
        """Req 6.6: total job duration in seconds is printed on Completed."""
        sm = _make_sm_client()
        start = time.time() - 42.0
        with pytest.raises(SystemExit):
            handle_terminal_state(
                sm, _FAKE_JOB_NAME, "Completed",
                start_time=start,
                output_s3_uri="s3://b/out/",
            )
        captured = capsys.readouterr()
        # Duration should be ~42 seconds; just check a numeric value is present
        assert any(char.isdigit() for char in captured.out)

    def test_failed_exits_nonzero(self):
        """Req 6.7: Failed state exits non-zero."""
        sm = _make_sm_client(
            statuses=["Failed"],
            failure_reason="Out of memory",
        )
        with pytest.raises(SystemExit) as exc_info:
            handle_terminal_state(
                sm, _FAKE_JOB_NAME, "Failed",
                start_time=time.time(),
                output_s3_uri="s3://b/out/",
            )
        assert exc_info.value.code != 0

    def test_stopped_exits_nonzero(self):
        """Req 6.7: Stopped state exits non-zero."""
        sm = _make_sm_client(statuses=["Stopped"])
        with pytest.raises(SystemExit) as exc_info:
            handle_terminal_state(
                sm, _FAKE_JOB_NAME, "Stopped",
                start_time=time.time(),
                output_s3_uri="s3://b/out/",
            )
        assert exc_info.value.code != 0

    def test_failed_prints_failure_reason(self, capsys):
        """Req 6.7: FailureReason from the API is printed."""
        sm = MagicMock()
        sm.describe_transform_job.return_value = {
            "TransformJobStatus": "Failed",
            "FailureReason": "Container ran out of disk space",
        }
        with pytest.raises(SystemExit):
            handle_terminal_state(
                sm, _FAKE_JOB_NAME, "Failed",
                start_time=time.time(),
                output_s3_uri="s3://b/out/",
            )
        captured = capsys.readouterr()
        assert "Container ran out of disk space" in captured.out

    def test_failed_with_null_reason_prints_fallback(self, capsys):
        """Req 6.7: 'Failure reason not available' printed when reason is None."""
        sm = MagicMock()
        sm.describe_transform_job.return_value = {
            "TransformJobStatus": "Failed",
            "FailureReason": None,
        }
        with pytest.raises(SystemExit):
            handle_terminal_state(
                sm, _FAKE_JOB_NAME, "Failed",
                start_time=time.time(),
                output_s3_uri="s3://b/out/",
            )
        captured = capsys.readouterr()
        assert "Failure reason not available" in captured.out

    def test_failed_with_missing_reason_field_prints_fallback(self, capsys):
        """Req 6.7: 'Failure reason not available' when FailureReason absent."""
        sm = MagicMock()
        sm.describe_transform_job.return_value = {"TransformJobStatus": "Failed"}
        with pytest.raises(SystemExit):
            handle_terminal_state(
                sm, _FAKE_JOB_NAME, "Failed",
                start_time=time.time(),
                output_s3_uri="s3://b/out/",
            )
        captured = capsys.readouterr()
        assert "Failure reason not available" in captured.out

    def test_describe_failure_during_reason_retrieval_prints_fallback(self, capsys):
        """If DescribeTransformJob raises during reason retrieval, fallback is used."""
        sm = MagicMock()
        sm.describe_transform_job.side_effect = Exception("describe failed")
        with pytest.raises(SystemExit):
            handle_terminal_state(
                sm, _FAKE_JOB_NAME, "Failed",
                start_time=time.time(),
                output_s3_uri="s3://b/out/",
            )
        captured = capsys.readouterr()
        assert "Failure reason not available" in captured.out


# ---------------------------------------------------------------------------
# CLI argument parsing (build_parser)
# ---------------------------------------------------------------------------


class TestBuildParser:
    """Tests for the argument parser (Req 6.2, 6.3)."""

    def test_default_instance_type(self):
        parser = build_parser()
        args = parser.parse_args([
            "--input-s3-uri", "s3://b/in/",
            "--output-s3-uri", "s3://b/out/",
            "--model-name", "my-model",
        ])
        assert args.instance_type == "ml.g4dn.xlarge"

    def test_custom_instance_type(self):
        parser = build_parser()
        args = parser.parse_args([
            "--input-s3-uri", "s3://b/in/",
            "--output-s3-uri", "s3://b/out/",
            "--model-name", "my-model",
            "--instance-type", "ml.p3.2xlarge",
        ])
        assert args.instance_type == "ml.p3.2xlarge"

    def test_missing_input_s3_uri_defaults_to_none(self):
        parser = build_parser()
        args = parser.parse_args([
            "--output-s3-uri", "s3://b/out/",
            "--model-name", "my-model",
        ])
        assert args.input_s3_uri is None

    def test_all_required_args_parsed_correctly(self):
        parser = build_parser()
        args = parser.parse_args([
            "--input-s3-uri", "s3://b/input/",
            "--output-s3-uri", "s3://b/output/",
            "--model-name", "test-model",
        ])
        assert args.input_s3_uri == "s3://b/input/"
        assert args.output_s3_uri == "s3://b/output/"
        assert args.model_name == "test-model"
