"""
Tests for scripts/setup_cloudwatch.py (CloudWatch_Monitor).

Covers tasks 13.1–13.7:
  - ensure_log_group: create if absent, update retention, idempotent
  - put_latency_alarm: correct parameters, SNS action set/unset
  - put_5xx_alarm: correct parameters
  - put_dashboard: correct widget structure, GPU vs CPU metric
  - run_setup / main: orchestration, idempotency, SNS warning

Requirements: 8.1, 8.3, 8.4, 8.5, 8.6, 8.7
"""

import json
import os
from unittest.mock import MagicMock, call, patch

import pytest
from botocore.exceptions import ClientError

from scripts.setup_cloudwatch import (
    ensure_log_group,
    put_5xx_alarm,
    put_dashboard,
    put_latency_alarm,
    run_setup,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

ENDPOINT_NAME = "retail-cv-inference"


def _resource_exists_error():
    return ClientError(
        {"Error": {"Code": "ResourceAlreadyExistsException", "Message": "exists"}},
        "CreateLogGroup",
    )


def _other_client_error():
    return ClientError(
        {"Error": {"Code": "SomeOtherError", "Message": "unexpected"}},
        "CreateLogGroup",
    )


# ---------------------------------------------------------------------------
# ensure_log_group
# ---------------------------------------------------------------------------


class TestEnsureLogGroup:
    """Tests for ensure_log_group (Req 8.1)."""

    def test_creates_log_group_when_absent(self):
        cw = MagicMock()
        ensure_log_group(cw, "/aws/sagemaker/Endpoints/retail-cv-inference")
        cw.create_log_group.assert_called_once_with(
            logGroupName="/aws/sagemaker/Endpoints/retail-cv-inference"
        )

    def test_does_not_raise_when_log_group_already_exists(self):
        """ResourceAlreadyExistsException must be silently swallowed."""
        cw = MagicMock()
        cw.create_log_group.side_effect = _resource_exists_error()
        # Should not raise
        ensure_log_group(cw, "/aws/sagemaker/Endpoints/retail-cv-inference")

    def test_reraises_other_client_errors(self):
        """Non-ResourceAlreadyExists ClientErrors must propagate."""
        cw = MagicMock()
        cw.create_log_group.side_effect = _other_client_error()
        with pytest.raises(ClientError):
            ensure_log_group(cw, "/aws/sagemaker/Endpoints/retail-cv-inference")

    def test_always_calls_put_retention_policy(self):
        """Retention policy is set regardless of whether group was just created."""
        cw = MagicMock()
        ensure_log_group(cw, "/aws/sagemaker/Endpoints/retail-cv-inference", retention_days=30)
        cw.put_retention_policy.assert_called_once_with(
            logGroupName="/aws/sagemaker/Endpoints/retail-cv-inference",
            retentionInDays=30,
        )

    def test_retention_policy_set_when_group_already_exists(self):
        """Retention is still updated even when the group already exists."""
        cw = MagicMock()
        cw.create_log_group.side_effect = _resource_exists_error()
        ensure_log_group(cw, "/aws/sagemaker/Endpoints/retail-cv-inference", retention_days=30)
        cw.put_retention_policy.assert_called_once_with(
            logGroupName="/aws/sagemaker/Endpoints/retail-cv-inference",
            retentionInDays=30,
        )

    def test_default_retention_is_30_days(self):
        cw = MagicMock()
        ensure_log_group(cw, "/aws/sagemaker/TransformJobs/retail-cv-batch")
        _, kwargs = cw.put_retention_policy.call_args
        assert kwargs["retentionInDays"] == 30

    def test_idempotent_second_call(self):
        """Running twice must not raise and must call put_retention_policy twice."""
        cw = MagicMock()
        cw.create_log_group.side_effect = [None, _resource_exists_error()]
        ensure_log_group(cw, "/aws/sagemaker/Endpoints/retail-cv-inference")
        ensure_log_group(cw, "/aws/sagemaker/Endpoints/retail-cv-inference")
        assert cw.put_retention_policy.call_count == 2


# ---------------------------------------------------------------------------
# put_latency_alarm
# ---------------------------------------------------------------------------


class TestPutLatencyAlarm:
    """Tests for put_latency_alarm (Req 8.3, 8.7)."""

    def test_calls_put_metric_alarm(self):
        cw = MagicMock()
        put_latency_alarm(cw, ENDPOINT_NAME)
        cw.put_metric_alarm.assert_called_once()

    def test_alarm_name(self):
        cw = MagicMock()
        put_latency_alarm(cw, ENDPOINT_NAME)
        kwargs = cw.put_metric_alarm.call_args.kwargs
        assert kwargs["AlarmName"] == "retail-cv-latency-p99"

    def test_metric_name_is_model_latency(self):
        cw = MagicMock()
        put_latency_alarm(cw, ENDPOINT_NAME)
        kwargs = cw.put_metric_alarm.call_args.kwargs
        assert kwargs["MetricName"] == "ModelLatency"

    def test_namespace_is_aws_sagemaker(self):
        cw = MagicMock()
        put_latency_alarm(cw, ENDPOINT_NAME)
        kwargs = cw.put_metric_alarm.call_args.kwargs
        assert kwargs["Namespace"] == "AWS/SageMaker"

    def test_endpoint_dimension(self):
        cw = MagicMock()
        put_latency_alarm(cw, ENDPOINT_NAME)
        kwargs = cw.put_metric_alarm.call_args.kwargs
        dims = kwargs["Dimensions"]
        assert any(d["Name"] == "EndpointName" and d["Value"] == ENDPOINT_NAME for d in dims)

    def test_extended_statistic_p99(self):
        cw = MagicMock()
        put_latency_alarm(cw, ENDPOINT_NAME)
        kwargs = cw.put_metric_alarm.call_args.kwargs
        assert kwargs["ExtendedStatistic"] == "p99"

    def test_threshold_4000ms(self):
        cw = MagicMock()
        put_latency_alarm(cw, ENDPOINT_NAME)
        kwargs = cw.put_metric_alarm.call_args.kwargs
        assert kwargs["Threshold"] == 4000

    def test_period_300s(self):
        cw = MagicMock()
        put_latency_alarm(cw, ENDPOINT_NAME)
        kwargs = cw.put_metric_alarm.call_args.kwargs
        assert kwargs["Period"] == 300

    def test_evaluation_periods_1(self):
        cw = MagicMock()
        put_latency_alarm(cw, ENDPOINT_NAME)
        kwargs = cw.put_metric_alarm.call_args.kwargs
        assert kwargs["EvaluationPeriods"] == 1

    def test_datapoints_to_alarm_1(self):
        cw = MagicMock()
        put_latency_alarm(cw, ENDPOINT_NAME)
        kwargs = cw.put_metric_alarm.call_args.kwargs
        assert kwargs["DatapointsToAlarm"] == 1

    def test_comparison_operator(self):
        cw = MagicMock()
        put_latency_alarm(cw, ENDPOINT_NAME)
        kwargs = cw.put_metric_alarm.call_args.kwargs
        assert kwargs["ComparisonOperator"] == "GreaterThanThreshold"

    def test_treat_missing_data_not_breaching(self):
        cw = MagicMock()
        put_latency_alarm(cw, ENDPOINT_NAME)
        kwargs = cw.put_metric_alarm.call_args.kwargs
        assert kwargs["TreatMissingData"] == "notBreaching"

    def test_sns_arn_added_to_alarm_actions_when_set(self):
        """Req 8.7: SNS action is added when sns_arn is provided."""
        cw = MagicMock()
        put_latency_alarm(cw, ENDPOINT_NAME, sns_arn="arn:aws:sns:ap-southeast-2:123:alerts")
        kwargs = cw.put_metric_alarm.call_args.kwargs
        assert "arn:aws:sns:ap-southeast-2:123:alerts" in kwargs["AlarmActions"]

    def test_no_alarm_actions_when_sns_arn_none(self):
        """Req 8.7: No SNS action when sns_arn is None."""
        cw = MagicMock()
        put_latency_alarm(cw, ENDPOINT_NAME, sns_arn=None)
        kwargs = cw.put_metric_alarm.call_args.kwargs
        assert kwargs["AlarmActions"] == []

    def test_warning_printed_when_sns_arn_none(self, capsys):
        """Req 8.7: A warning is printed to stdout when SNS ARN is unset."""
        cw = MagicMock()
        put_latency_alarm(cw, ENDPOINT_NAME, sns_arn=None)
        captured = capsys.readouterr()
        assert "warning" in captured.out.lower() or "Warning" in captured.out

    def test_no_warning_when_sns_arn_set(self, capsys):
        """No warning is printed when SNS ARN is provided."""
        cw = MagicMock()
        put_latency_alarm(cw, ENDPOINT_NAME, sns_arn="arn:aws:sns:us-east-1:123:topic")
        captured = capsys.readouterr()
        assert "CLOUDWATCH_ALARM_SNS_ARN" not in captured.out

    def test_idempotent_second_call(self):
        """put_metric_alarm is idempotent by CloudWatch API design; calling twice is fine."""
        cw = MagicMock()
        put_latency_alarm(cw, ENDPOINT_NAME)
        put_latency_alarm(cw, ENDPOINT_NAME)
        assert cw.put_metric_alarm.call_count == 2


# ---------------------------------------------------------------------------
# put_5xx_alarm
# ---------------------------------------------------------------------------


class TestPut5xxAlarm:
    """Tests for put_5xx_alarm (Req 8.4)."""

    def test_calls_put_metric_alarm(self):
        cw = MagicMock()
        put_5xx_alarm(cw, ENDPOINT_NAME)
        cw.put_metric_alarm.assert_called_once()

    def test_alarm_name(self):
        cw = MagicMock()
        put_5xx_alarm(cw, ENDPOINT_NAME)
        kwargs = cw.put_metric_alarm.call_args.kwargs
        assert kwargs["AlarmName"] == "retail-cv-5xx-errors"

    def test_metric_name(self):
        cw = MagicMock()
        put_5xx_alarm(cw, ENDPOINT_NAME)
        kwargs = cw.put_metric_alarm.call_args.kwargs
        assert kwargs["MetricName"] == "Invocation5XXErrors"

    def test_statistic_is_sum(self):
        cw = MagicMock()
        put_5xx_alarm(cw, ENDPOINT_NAME)
        kwargs = cw.put_metric_alarm.call_args.kwargs
        assert kwargs["Statistic"] == "Sum"

    def test_threshold_is_5(self):
        cw = MagicMock()
        put_5xx_alarm(cw, ENDPOINT_NAME)
        kwargs = cw.put_metric_alarm.call_args.kwargs
        assert kwargs["Threshold"] == 5

    def test_period_300s(self):
        cw = MagicMock()
        put_5xx_alarm(cw, ENDPOINT_NAME)
        kwargs = cw.put_metric_alarm.call_args.kwargs
        assert kwargs["Period"] == 300

    def test_evaluation_periods_1(self):
        cw = MagicMock()
        put_5xx_alarm(cw, ENDPOINT_NAME)
        kwargs = cw.put_metric_alarm.call_args.kwargs
        assert kwargs["EvaluationPeriods"] == 1

    def test_treat_missing_data_not_breaching(self):
        cw = MagicMock()
        put_5xx_alarm(cw, ENDPOINT_NAME)
        kwargs = cw.put_metric_alarm.call_args.kwargs
        assert kwargs["TreatMissingData"] == "notBreaching"

    def test_namespace_is_aws_sagemaker(self):
        cw = MagicMock()
        put_5xx_alarm(cw, ENDPOINT_NAME)
        kwargs = cw.put_metric_alarm.call_args.kwargs
        assert kwargs["Namespace"] == "AWS/SageMaker"

    def test_endpoint_dimension(self):
        cw = MagicMock()
        put_5xx_alarm(cw, ENDPOINT_NAME)
        kwargs = cw.put_metric_alarm.call_args.kwargs
        dims = kwargs["Dimensions"]
        assert any(d["Name"] == "EndpointName" and d["Value"] == ENDPOINT_NAME for d in dims)

    def test_sns_arn_added_when_set(self):
        cw = MagicMock()
        put_5xx_alarm(cw, ENDPOINT_NAME, sns_arn="arn:aws:sns:ap-southeast-2:123:alerts")
        kwargs = cw.put_metric_alarm.call_args.kwargs
        assert "arn:aws:sns:ap-southeast-2:123:alerts" in kwargs["AlarmActions"]

    def test_no_alarm_actions_when_sns_arn_none(self):
        cw = MagicMock()
        put_5xx_alarm(cw, ENDPOINT_NAME, sns_arn=None)
        kwargs = cw.put_metric_alarm.call_args.kwargs
        assert kwargs["AlarmActions"] == []


# ---------------------------------------------------------------------------
# put_dashboard
# ---------------------------------------------------------------------------


class TestPutDashboard:
    """Tests for put_dashboard (Req 8.5)."""

    def _get_widget_titles(self, cw) -> list[str]:
        kwargs = cw.put_dashboard.call_args.kwargs
        body = json.loads(kwargs["DashboardBody"])
        return [w["properties"]["title"] for w in body["widgets"]]

    def test_calls_put_dashboard(self):
        cw = MagicMock()
        put_dashboard(cw, ENDPOINT_NAME)
        cw.put_dashboard.assert_called_once()

    def test_dashboard_name(self):
        cw = MagicMock()
        put_dashboard(cw, ENDPOINT_NAME)
        kwargs = cw.put_dashboard.call_args.kwargs
        assert kwargs["DashboardName"] == "retail-cv-inference-dashboard"

    def test_dashboard_body_is_valid_json(self):
        cw = MagicMock()
        put_dashboard(cw, ENDPOINT_NAME)
        kwargs = cw.put_dashboard.call_args.kwargs
        body = json.loads(kwargs["DashboardBody"])
        assert "widgets" in body

    def test_model_latency_widget_present(self):
        cw = MagicMock()
        put_dashboard(cw, ENDPOINT_NAME)
        titles = self._get_widget_titles(cw)
        assert "ModelLatency" in titles

    def test_invocations_widget_present(self):
        cw = MagicMock()
        put_dashboard(cw, ENDPOINT_NAME)
        titles = self._get_widget_titles(cw)
        assert "Invocations" in titles

    def test_5xx_errors_widget_present(self):
        cw = MagicMock()
        put_dashboard(cw, ENDPOINT_NAME)
        titles = self._get_widget_titles(cw)
        assert "Invocation5XXErrors" in titles

    def test_cpu_utilization_widget_when_not_gpu(self):
        """Req 8.5: CPUUtilization widget when use_gpu_metrics=False."""
        cw = MagicMock()
        put_dashboard(cw, ENDPOINT_NAME, use_gpu_metrics=False)
        titles = self._get_widget_titles(cw)
        assert "CPUUtilization" in titles
        assert "GPUUtilization" not in titles

    def test_gpu_utilization_widget_when_gpu(self):
        """Req 8.5: GPUUtilization widget when use_gpu_metrics=True."""
        cw = MagicMock()
        put_dashboard(cw, ENDPOINT_NAME, use_gpu_metrics=True)
        titles = self._get_widget_titles(cw)
        assert "GPUUtilization" in titles
        assert "CPUUtilization" not in titles

    def test_model_latency_widget_has_p50_p90_p99(self):
        """ModelLatency widget must include p50, p90, and p99 statistics."""
        cw = MagicMock()
        put_dashboard(cw, ENDPOINT_NAME)
        kwargs = cw.put_dashboard.call_args.kwargs
        body = json.loads(kwargs["DashboardBody"])

        latency_widget = next(
            w for w in body["widgets"]
            if w["properties"]["title"] == "ModelLatency"
        )
        # Collect all stat labels from the metrics list
        body_str = json.dumps(latency_widget)
        assert "p50" in body_str
        assert "p90" in body_str
        assert "p99" in body_str

    def test_four_widgets_total(self):
        cw = MagicMock()
        put_dashboard(cw, ENDPOINT_NAME)
        kwargs = cw.put_dashboard.call_args.kwargs
        body = json.loads(kwargs["DashboardBody"])
        assert len(body["widgets"]) == 4

    def test_idempotent_second_call(self):
        """put_dashboard CloudWatch API is idempotent; calling twice is fine."""
        cw = MagicMock()
        put_dashboard(cw, ENDPOINT_NAME)
        put_dashboard(cw, ENDPOINT_NAME)
        assert cw.put_dashboard.call_count == 2


# ---------------------------------------------------------------------------
# run_setup orchestration
# ---------------------------------------------------------------------------


class TestRunSetup:
    """Tests for run_setup orchestration (Req 8.1, 8.6, 8.7)."""

    def _make_clients(self):
        """Return separate mock clients for logs and cloudwatch metrics."""
        cw_logs = MagicMock()
        return cw_logs

    def test_calls_ensure_log_group(self):
        cw = MagicMock()
        with patch.dict(os.environ, {}, clear=False):
            if "CLOUDWATCH_ALARM_SNS_ARN" in os.environ:
                del os.environ["CLOUDWATCH_ALARM_SNS_ARN"]
            run_setup(cw, ENDPOINT_NAME, "realtime")
        cw.create_log_group.assert_called()

    def test_realtime_uses_correct_log_group(self):
        cw = MagicMock()
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("CLOUDWATCH_ALARM_SNS_ARN", None)
            run_setup(cw, ENDPOINT_NAME, "realtime")
        log_group_name = cw.create_log_group.call_args.kwargs["logGroupName"]
        assert log_group_name == "/aws/sagemaker/Endpoints/retail-cv-inference"

    def test_batch_uses_correct_log_group(self):
        cw = MagicMock()
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("CLOUDWATCH_ALARM_SNS_ARN", None)
            run_setup(cw, ENDPOINT_NAME, "batch")
        log_group_name = cw.create_log_group.call_args.kwargs["logGroupName"]
        assert log_group_name == "/aws/sagemaker/TransformJobs/retail-cv-batch"

    def test_calls_put_metric_alarm_twice(self):
        """Both latency and 5xx alarms are created."""
        cw = MagicMock()
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("CLOUDWATCH_ALARM_SNS_ARN", None)
            run_setup(cw, ENDPOINT_NAME, "realtime")
        assert cw.put_metric_alarm.call_count == 2

    def test_calls_put_dashboard(self):
        cw = MagicMock()
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("CLOUDWATCH_ALARM_SNS_ARN", None)
            run_setup(cw, ENDPOINT_NAME, "realtime")
        cw.put_dashboard.assert_called_once()

    def test_sns_arn_from_env_passed_to_alarms(self):
        """Req 8.7: CLOUDWATCH_ALARM_SNS_ARN env var is used for alarm actions."""
        cw = MagicMock()
        test_arn = "arn:aws:sns:ap-southeast-2:123:retail-cv-alerts"
        with patch.dict(os.environ, {"CLOUDWATCH_ALARM_SNS_ARN": test_arn}):
            run_setup(cw, ENDPOINT_NAME, "realtime")
        for call_kwargs in [c.kwargs for c in cw.put_metric_alarm.call_args_list]:
            assert test_arn in call_kwargs["AlarmActions"]

    def test_warning_printed_when_sns_arn_not_set(self, capsys):
        """Req 8.7: warning is printed to stdout when SNS ARN env var is absent."""
        cw = MagicMock()
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("CLOUDWATCH_ALARM_SNS_ARN", None)
            run_setup(cw, ENDPOINT_NAME, "realtime")
        captured = capsys.readouterr()
        assert "warning" in captured.out.lower() or "Warning" in captured.out

    def test_idempotent_multiple_calls(self):
        """Req 8.6: running setup multiple times must not raise."""
        cw = MagicMock()
        cw.create_log_group.side_effect = [None, _resource_exists_error()]
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("CLOUDWATCH_ALARM_SNS_ARN", None)
            run_setup(cw, ENDPOINT_NAME, "realtime")
            run_setup(cw, ENDPOINT_NAME, "realtime")
        # Both runs completed without raising; assert put_metric_alarm called twice per run
        assert cw.put_metric_alarm.call_count == 4
