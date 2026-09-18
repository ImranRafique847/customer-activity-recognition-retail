"""
scripts/iam_validator.py

IAM_Validator: Verifies that the Computer-Vision IAM user has the exact
permissions required by every pipeline script.

Usage:
    python scripts/iam_validator.py

Exit codes:
    0  — all required permissions present (IAM validation passed)
    1  — permissions missing, connection failed, or missing env vars
"""

import json
import sys
from dataclasses import dataclass, field
from typing import Literal

import boto3
from botocore.exceptions import ClientError, NoCredentialsError

from scripts.common.env_loader import load_env_and_validate

# ---------------------------------------------------------------------------
# Required action constants
# ---------------------------------------------------------------------------

REQUIRED_S3_ACTIONS = [
    "s3:GetObject",
    "s3:PutObject",
    "s3:DeleteObject",
    "s3:ListBucket",
]

REQUIRED_SAGEMAKER_ACTIONS = [
    "sagemaker:CreateModel",
    "sagemaker:CreateEndpointConfig",
    "sagemaker:CreateEndpoint",
    "sagemaker:InvokeEndpoint",
    "sagemaker:DeleteEndpoint",
    "sagemaker:DescribeEndpoint",
]

REQUIRED_LOGS_ACTIONS = [
    "logs:CreateLogGroup",
    "logs:CreateLogStream",
    "logs:PutLogEvents",
]


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class ValidationResult:
    """Encapsulates the outcome of an IAM permission validation run.

    Attributes:
        verification_result: One of ``"PASSED"``, ``"PERMISSIONS_MISSING"``,
            or ``"CONNECTION_FAILED"``.
        permissions_missing: List of AWS policy statement dicts, each
            representing a missing permission that can be attached directly
            via ``aws iam put-user-policy``.  Empty when *verification_result*
            is ``"PASSED"``.
    """

    verification_result: Literal["PASSED", "PERMISSIONS_MISSING", "CONNECTION_FAILED"]
    permissions_missing: list = field(default_factory=list)


# ---------------------------------------------------------------------------
# Core validator (stub — implemented in task 5.2)
# ---------------------------------------------------------------------------


def validate_iam_permissions(
    iam_client,
    user_name: str,
    sagemaker_role_arn: str,
) -> ValidationResult:
    """Simulate and collect effective permissions for *user_name*.

    Uses ``iam:SimulatePrincipalPolicy`` to check every required action
    against its required resource ARN.  Builds a list of missing policy
    statement objects for any absent action.

    Args:
        iam_client: A boto3 IAM client already authenticated with valid
            credentials.
        user_name: The IAM user name whose permissions are being validated
            (e.g. ``"Computer-Vision"``).
        sagemaker_role_arn: The ARN of the SageMaker execution role, used to
            check the ``iam:PassRole`` permission.

    Returns:
        :class:`ValidationResult` with ``verification_result`` set to
        ``"PASSED"`` or ``"PERMISSIONS_MISSING"`` and *permissions_missing*
        populated accordingly.  Never raises — errors are encoded in the
        result.
    """
    try:
        # Step 1 — resolve the IAM user ARN
        user_arn = iam_client.get_user(UserName=user_name)["User"]["Arn"]

        # Step 2 — define all permission checks as (actions, resource_arn) tuples
        permission_checks = [
            # s3:ListBucket must be tested against the bucket ARN (no trailing /*)
            (REQUIRED_S3_ACTIONS, "arn:aws:s3:::retaildata-cv-2026"),
            # object-level S3 operations require the /* resource
            (REQUIRED_S3_ACTIONS, "arn:aws:s3:::retaildata-cv-2026/*"),
            # SageMaker actions are wildcard-resourced
            (REQUIRED_SAGEMAKER_ACTIONS, "*"),
            # CloudWatch Logs actions scoped to SageMaker log groups
            (REQUIRED_LOGS_ACTIONS, "arn:aws:logs:*:*:log-group:/aws/sagemaker/*"),
            # iam:PassRole must be tested against the concrete SageMaker role ARN
            (["iam:PassRole"], sagemaker_role_arn),
        ]

        # Step 3 & 4 — simulate each group and collect denied actions
        permissions_missing = []

        for actions, resource_arn in permission_checks:
            response = iam_client.simulate_principal_policy(
                PolicySourceArn=user_arn,
                ActionNames=actions,
                ResourceArns=[resource_arn],
            )

            for eval_result in response["EvaluationResults"]:
                if eval_result["EvalDecision"] != "allowed":
                    # Step 5 — build one policy statement per missing action
                    permissions_missing.append(
                        {
                            "Effect": "Allow",
                            "Action": [eval_result["EvalActionName"]],
                            "Resource": resource_arn,
                        }
                    )

        # Steps 6 & 7 — return appropriate result
        if not permissions_missing:
            return ValidationResult(verification_result="PASSED")

        return ValidationResult(
            verification_result="PERMISSIONS_MISSING",
            permissions_missing=permissions_missing,
        )

    except Exception as exc:  # Step 8 — never raise; encode all errors in the result
        return ValidationResult(
            verification_result="CONNECTION_FAILED",
            permissions_missing=[
                {
                    "error": str(exc),
                    "error_type": type(exc).__name__,
                }
            ],
        )


# ---------------------------------------------------------------------------
# CLI entrypoint
# ---------------------------------------------------------------------------


def run_validator() -> None:
    """CLI entrypoint for the IAM_Validator.

    Workflow:
    1. Load and validate required environment variables from the project
       ``.env`` file via :func:`~scripts.common.env_loader.load_env_and_validate`.
       If any variable is absent/empty the function prints the missing names
       and exits with code 1 before touching AWS.
    2. Create a boto3 IAM client using the loaded credentials.
    3. Call :func:`validate_iam_permissions` (stub — raises NotImplementedError
       until task 5.2 is complete).
    4. Print JSON result to stdout and exit with the appropriate code.

    On ``NoCredentialsError`` or a ``ClientError`` that indicates a connection
    or authentication problem, sets ``verification_result`` to
    ``"CONNECTION_FAILED"``, prints a descriptive JSON error to stdout that
    names the environment variables checked, and exits with code 1.
    """
    required_vars = [
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
        "AWS_DEFAULT_REGION",
        "SAGEMAKER_EXECUTION_ROLE_ARN",
    ]

    # Step 1 — load and validate env vars (exits 1 on any missing/empty var)
    env = load_env_and_validate(required_vars)

    # Step 2 — bootstrap the IAM client from the validated env vars
    try:
        iam_client = boto3.client(
            "iam",
            aws_access_key_id=env["AWS_ACCESS_KEY_ID"],
            aws_secret_access_key=env["AWS_SECRET_ACCESS_KEY"],
            region_name=env["AWS_DEFAULT_REGION"],
        )

        # Step 3 — delegate to validate_iam_permissions (task 5.2 stub)
        sagemaker_role_arn = env["SAGEMAKER_EXECUTION_ROLE_ARN"]
        result = validate_iam_permissions(
            iam_client=iam_client,
            user_name="Computer-Vision",
            sagemaker_role_arn=sagemaker_role_arn,
        )

    except NoCredentialsError as exc:
        result = ValidationResult(
            verification_result="CONNECTION_FAILED",
            permissions_missing=[
                {
                    "error": str(exc),
                    "credential_source_checked": [
                        "AWS_ACCESS_KEY_ID",
                        "AWS_SECRET_ACCESS_KEY",
                        "AWS_DEFAULT_REGION",
                    ],
                }
            ],
        )
        print(json.dumps({"verification_result": result.verification_result,
                          "error": str(exc),
                          "credential_source_checked": [
                              "AWS_ACCESS_KEY_ID",
                              "AWS_SECRET_ACCESS_KEY",
                              "AWS_DEFAULT_REGION",
                          ]},
                         indent=2))
        sys.exit(1)

    except ClientError as exc:
        error_code = exc.response.get("Error", {}).get("Code", "")
        # Treat authentication / endpoint-unreachable errors as CONNECTION_FAILED
        connection_error_codes = {
            "InvalidClientTokenId",
            "AuthFailure",
            "ExpiredTokenException",
            "UnrecognizedClientException",
            "NetworkingError",
            "EndpointResolutionError",
        }
        if error_code in connection_error_codes or not error_code:
            result = ValidationResult(
                verification_result="CONNECTION_FAILED",
                permissions_missing=[],
            )
            print(json.dumps({
                "verification_result": "CONNECTION_FAILED",
                "error": str(exc),
                "credential_source_checked": [
                    "AWS_ACCESS_KEY_ID",
                    "AWS_SECRET_ACCESS_KEY",
                    "AWS_DEFAULT_REGION",
                ],
            }, indent=2))
            sys.exit(1)
        # Re-raise non-connection ClientErrors so they surface as unexpected
        raise

    # Step 4 — output result and exit with appropriate code
    if result.verification_result == "PASSED":
        print("IAM validation passed")
        sys.exit(0)
    elif result.verification_result == "PERMISSIONS_MISSING":
        print(json.dumps({
            "verification_result": result.verification_result,
            "permissions_missing": result.permissions_missing,
        }, indent=2))
        sys.exit(1)
    else:
        # CONNECTION_FAILED already handled above; this branch is a safety net
        print(json.dumps({
            "verification_result": result.verification_result,
            "permissions_missing": result.permissions_missing,
        }, indent=2))
        sys.exit(1)


if __name__ == "__main__":
    run_validator()
