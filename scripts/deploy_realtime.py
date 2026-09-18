"""
scripts/deploy_realtime.py — SageMaker Real-Time Endpoint deployer

COST NOTE: Delete this endpoint after 30 minutes of idle time
(zero invocations) to avoid unnecessary charges. Run:
    aws sagemaker delete-endpoint --endpoint-name <name>

Usage:
    python -m scripts.deploy_realtime \\
        --model-name    retail-cv-yolo-run_001 \\
        --endpoint-name retail-cv-inference

Exit codes:
    0  — endpoint is InService
    1  — creation failed or CLI error
"""

import argparse
import sys
import time
from datetime import datetime, timezone

import boto3

from scripts.common.env_loader import load_env_and_validate


# ---------------------------------------------------------------------------
# Endpoint deployment
# ---------------------------------------------------------------------------


def deploy_realtime_endpoint(
    sm_client,
    aas_client,
    model_name: str,
    endpoint_name: str,
    instance_type: str = "ml.g4dn.xlarge",
    region: str = "us-east-1",
) -> None:
    """Create a SageMaker real-time endpoint and register auto-scaling.

    Polls until InService or Failed. On InService, prints the invocation URL
    and registers auto-scaling (1–4 instances, scale-out >10 req/min,
    scale-in <5 req/min for 5 consecutive minutes).
    """
    config_name = f"{endpoint_name}-config"

    # Create endpoint config.
    sm_client.create_endpoint_config(
        EndpointConfigName=config_name,
        ProductionVariants=[
            {
                "VariantName": "AllTraffic",
                "ModelName": model_name,
                "InstanceType": instance_type,
                "InitialInstanceCount": 1,
                "InitialVariantWeight": 1.0,
            }
        ],
    )

    # Create endpoint.
    sm_client.create_endpoint(
        EndpointName=endpoint_name,
        EndpointConfigName=config_name,
    )
    print(f"Creating endpoint: {endpoint_name} ...")

    # Poll until terminal state.
    while True:
        resp = sm_client.describe_endpoint(EndpointName=endpoint_name)
        status = resp["EndpointStatus"]
        ts = datetime.now(tz=timezone.utc).isoformat()
        print(f"[{ts}] Endpoint: {endpoint_name} | Status: {status}")

        if status == "InService":
            break
        if status == "Failed":
            reason = resp.get("FailureReason", "Unknown failure reason")
            print(f"Error: endpoint creation failed. Reason: {reason}")
            sys.exit(1)

        time.sleep(30)

    # Print confirmation (Req 7.5).
    invocation_url = (
        f"https://runtime.sagemaker.{region}.amazonaws.com"
        f"/endpoints/{endpoint_name}/invocations"
    )
    print(f"\nEndpoint name:    {endpoint_name}")
    print(f"Status:           InService")
    print(f"Invocation URL:   {invocation_url}")

    # Register auto-scaling (Req 7.2).
    resource_id = f"endpoint/{endpoint_name}/variant/AllTraffic"

    aas_client.register_scalable_target(
        ServiceNamespace="sagemaker",
        ResourceId=resource_id,
        ScalableDimension="sagemaker:variant:DesiredInstanceCount",
        MinCapacity=1,
        MaxCapacity=4,
    )

    # Scale-out: InvocationsPerInstance > 10/min.
    aas_client.put_scaling_policy(
        PolicyName=f"{endpoint_name}-scale-out",
        ServiceNamespace="sagemaker",
        ResourceId=resource_id,
        ScalableDimension="sagemaker:variant:DesiredInstanceCount",
        PolicyType="TargetTrackingScaling",
        TargetTrackingScalingPolicyConfiguration={
            "TargetValue": 10.0,
            "CustomizedMetricSpecification": {
                "MetricName": "InvocationsPerInstance",
                "Namespace": "AWS/SageMaker",
                "Dimensions": [{"Name": "EndpointName", "Value": endpoint_name}],
                "Statistic": "Sum",
            },
            "ScaleOutCooldown": 60,
            "ScaleInCooldown": 300,
        },
    )
    print("Auto-scaling registered (1–4 instances).")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="deploy_realtime",
        description="Deploy a SageMaker real-time inference endpoint.",
        epilog=(
            "COST NOTE: Delete the endpoint after 30 minutes of idle time.\n"
            "  aws sagemaker delete-endpoint --endpoint-name <name>"
        ),
    )
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--endpoint-name", required=True)
    parser.add_argument("--instance-type", default="ml.g4dn.xlarge")
    args = parser.parse_args()

    env = load_env_and_validate(
        ["AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_DEFAULT_REGION"]
    )
    region = env["AWS_DEFAULT_REGION"]

    sm_client = boto3.client("sagemaker", region_name=region)
    aas_client = boto3.client("application-autoscaling", region_name=region)

    deploy_realtime_endpoint(
        sm_client=sm_client,
        aas_client=aas_client,
        model_name=args.model_name,
        endpoint_name=args.endpoint_name,
        instance_type=args.instance_type,
        region=region,
    )


if __name__ == "__main__":
    main()
