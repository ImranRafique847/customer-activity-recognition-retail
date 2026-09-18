"""
scripts/setup_cloudwatch.py — CloudWatch_Monitor setup (idempotent)

Creates or updates CloudWatch log groups, metric alarms, and a dashboard
for the retail-cv inference endpoint. Safe to run multiple times.

Usage:
    python -m scripts.setup_cloudwatch \\
        --endpoint-name retail-cv-inference \\
        --deployment-type realtime

Exit codes:
    0  — setup complete
    1  — missing env vars or AWS error
"""

import argparse
import json
import os
import sys

import boto3
from botocore.exceptions import ClientError

from scripts.common.env_loader import load_env_and_validate


# ---------------------------------------------------------------------------
# Log group (idempotent)
# ---------------------------------------------------------------------------


def ensure_log_group(
    cw_client,
    log_group_name: str,
    retention_days: int = 30,
) -> None:
    """Create log group if absent; always enforce retention policy (Req 8.1)."""
    try:
        cw_client.create_log_group(logGroupName=log_group_name)
    except ClientError as exc:
        if exc.response["Error"]["Code"] != "ResourceAlreadyExistsException":
            raise

    # Always set/update retention — idempotent.
    cw_client.put_retention_policy(
        logGroupName=log_group_name,
        retentionInDays=retention_days,
    )


# ---------------------------------------------------------------------------
# Latency alarm (idempotent via put_metric_alarm)
# ---------------------------------------------------------------------------


def put_latency_alarm(
    cw_client,
    endpoint_name: str,
    sns_arn: str = None,
) -> None:
    """Create or update the p99 latency alarm (Req 8.3, 8.7)."""
    if sns_arn is None:
        print(
            "Warning: CLOUDWATCH_ALARM_SNS_ARN not set; "
            "alarm created without SNS action"
        )

    cw_client.put_metric_alarm(
        AlarmName="retail-cv-latency-p99",
        AlarmDescription="p99 ModelLatency exceeds 4 000 ms",
        MetricName="ModelLatency",
        Namespace="AWS/SageMaker",
        Dimensions=[{"Name": "EndpointName", "Value": endpoint_name}],
        ExtendedStatistic="p99",
        Period=300,
        EvaluationPeriods=1,
        DatapointsToAlarm=1,
        Threshold=4000,
        ComparisonOperator="GreaterThanThreshold",
        TreatMissingData="notBreaching",
        AlarmActions=[sns_arn] if sns_arn else [],
    )


# ---------------------------------------------------------------------------
# 5XX error alarm (idempotent)
# ---------------------------------------------------------------------------


def put_5xx_alarm(
    cw_client,
    endpoint_name: str,
    sns_arn: str = None,
) -> None:
    """Create or update the 5XX error count alarm (Req 8.4)."""
    cw_client.put_metric_alarm(
        AlarmName="retail-cv-5xx-errors",
        AlarmDescription="Invocation5XXErrors exceeds 5 in a 5-minute window",
        MetricName="Invocation5XXErrors",
        Namespace="AWS/SageMaker",
        Dimensions=[{"Name": "EndpointName", "Value": endpoint_name}],
        Statistic="Sum",
        Period=300,
        EvaluationPeriods=1,
        DatapointsToAlarm=1,
        Threshold=5,
        ComparisonOperator="GreaterThanThreshold",
        TreatMissingData="notBreaching",
        AlarmActions=[sns_arn] if sns_arn else [],
    )


# ---------------------------------------------------------------------------
# Dashboard (idempotent via put_dashboard)
# ---------------------------------------------------------------------------


def put_dashboard(
    cw_client,
    endpoint_name: str,
    use_gpu_metrics: bool = False,
) -> None:
    """Create or update the retail-cv-inference-dashboard (Req 8.5)."""
    util_metric = "GPUUtilization" if use_gpu_metrics else "CPUUtilization"

    widgets = [
        # ModelLatency — p50, p90, p99
        {
            "type": "metric",
            "properties": {
                "title": "ModelLatency",
                "metrics": [
                    ["AWS/SageMaker", "ModelLatency", "EndpointName", endpoint_name,
                     {"stat": "p50", "label": "p50"}],
                    ["...", {"stat": "p90", "label": "p90"}],
                    ["...", {"stat": "p99", "label": "p99"}],
                ],
                "period": 300,
                "view": "timeSeries",
            },
        },
        # Invocations
        {
            "type": "metric",
            "properties": {
                "title": "Invocations",
                "metrics": [
                    ["AWS/SageMaker", "Invocations", "EndpointName", endpoint_name]
                ],
                "period": 300,
                "view": "timeSeries",
            },
        },
        # 5XX errors
        {
            "type": "metric",
            "properties": {
                "title": "Invocation5XXErrors",
                "metrics": [
                    ["AWS/SageMaker", "Invocation5XXErrors",
                     "EndpointName", endpoint_name, {"stat": "Sum"}]
                ],
                "period": 300,
                "view": "timeSeries",
            },
        },
        # CPU or GPU utilization
        {
            "type": "metric",
            "properties": {
                "title": util_metric,
                "metrics": [
                    ["AWS/SageMaker", util_metric, "EndpointName", endpoint_name]
                ],
                "period": 300,
                "view": "timeSeries",
            },
        },
    ]

    cw_client.put_dashboard(
        DashboardName="retail-cv-inference-dashboard",
        DashboardBody=json.dumps({"widgets": widgets}),
    )


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


def run_setup(
    cw_client,
    endpoint_name: str,
    deployment_type: str,
    use_gpu_metrics: bool = False,
) -> None:
    """Provision all CloudWatch resources for the given deployment type."""
    sns_arn = os.environ.get("CLOUDWATCH_ALARM_SNS_ARN") or None

    if deployment_type == "realtime":
        log_group = "/aws/sagemaker/Endpoints/retail-cv-inference"
    else:
        log_group = "/aws/sagemaker/TransformJobs/retail-cv-batch"

    ensure_log_group(cw_client, log_group)
    put_latency_alarm(cw_client, endpoint_name, sns_arn)
    put_5xx_alarm(cw_client, endpoint_name, sns_arn)
    put_dashboard(cw_client, endpoint_name, use_gpu_metrics)

    print(f"CloudWatch setup complete for endpoint: {endpoint_name}")
    print(f"  Log group:  {log_group}")
    print(f"  Dashboard:  retail-cv-inference-dashboard")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="setup_cloudwatch",
        description="Provision CloudWatch log groups, alarms, and dashboard (idempotent).",
    )
    parser.add_argument("--endpoint-name", required=True)
    parser.add_argument(
        "--deployment-type",
        choices=["realtime", "batch"],
        default="realtime",
    )
    parser.add_argument(
        "--use-gpu-metrics",
        action="store_true",
        default=False,
        help="Use GPUUtilization instead of CPUUtilization in the dashboard.",
    )
    args = parser.parse_args()

    env = load_env_and_validate(
        ["AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_DEFAULT_REGION"]
    )

    cw_client = boto3.client("logs", region_name=env["AWS_DEFAULT_REGION"])
    # CloudWatch metrics/alarms/dashboards use the 'cloudwatch' client.
    cw_metrics = boto3.client("cloudwatch", region_name=env["AWS_DEFAULT_REGION"])

    # ensure_log_group uses the logs client; alarms/dashboard use cloudwatch client.
    sns_arn = os.environ.get("CLOUDWATCH_ALARM_SNS_ARN") or None

    if args.deployment_type == "realtime":
        log_group = "/aws/sagemaker/Endpoints/retail-cv-inference"
    else:
        log_group = "/aws/sagemaker/TransformJobs/retail-cv-batch"

    ensure_log_group(cw_client, log_group)
    put_latency_alarm(cw_metrics, args.endpoint_name, sns_arn)
    put_5xx_alarm(cw_metrics, args.endpoint_name, sns_arn)
    put_dashboard(cw_metrics, args.endpoint_name, args.use_gpu_metrics)

    print(f"CloudWatch setup complete for endpoint: {args.endpoint_name}")
    print(f"  Log group:  {log_group}")
    print(f"  Dashboard:  retail-cv-inference-dashboard")


if __name__ == "__main__":
    main()
