"""
Unit tests for scripts/iam_validator.py (IAM_Validator).

Covers task 5.6:
  - IAM validation success path: exit 0 and prints 'IAM validation passed'
  - Connection failure: CONNECTION_FAILED result shape with credential source info
  - All-missing-permissions case produces valid attachable policy statements
  - JSON output shape for PERMISSIONS_MISSING
  - Never raises — exceptions are encoded in ValidationResult

Requirements: 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7, 3.9
"""

import json
import sys
from unittest.mock import MagicMock, patch

import pytest
from botocore.exceptions import ClientError, NoCredentialsError

from scripts.iam_validator import (
    REQUIRED_LOGS_ACTIONS,
    REQUIRED_S3_ACTIONS,
    REQUIRED_SAGEMAKER_ACTIONS,
    ValidationResult,
    run_validator,
    validate_iam_permissions,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

FAKE_USER_ARN = "arn:aws:iam::123456789012:user/Computer-Vision"
FAKE_ROLE_ARN = "arn:aws:iam::123456789012:role/SageMakerExecutionRole"

S3_BUCKET_ARN = "arn:aws:s3:::retaildata-cv-2026"
S3_OBJECTS_ARN = "arn:aws:s3:::retaildata-cv-2026/*"
SAGEMAKER_RESOURCE = "*"
LOGS_RESOURCE = "arn:aws:logs:*:*:log-group:/aws/sagemaker/*"


def _make_iam_client(allowed_actions: set[str]) -> MagicMock:
    """
    Build a mock IAM client where ``simulate_principal_policy`` grants exactly
    the actions listed in *allowed_actions* and denies everything else.
    """
    client = MagicMock()
    client.get_user.return_value = {"User": {"Arn": FAKE_USER_ARN}}

    def _simulate(PolicySourceArn, ActionNames, ResourceArns):  # noqa: N803
        results = []
        for action in ActionNames:
            decision = "allowed" if action in allowed_actions else "implicitDeny"
            results.append(
                {
                    "EvalActionName": action,
                    "EvalDecision": decision,
                    "EvalResourceName": ResourceArns[0],
                }
            )
        return {"EvaluationResults": results}

    client.simulate_principal_policy.side_effect = _simulate
    return client


def _all_required_actions() -> set[str]:
    return (
        set(REQUIRED_S3_ACTIONS)
        | set(REQUIRED_SAGEMAKER_ACTIONS)
        | set(REQUIRED_LOGS_ACTIONS)
        | {"iam:PassRole"}
    )


# ---------------------------------------------------------------------------
# validate_iam_permissions — success path
# ---------------------------------------------------------------------------


class TestValidateIamPermissionsPassedPath:
    """All required permissions are present → PASSED."""

    def test_returns_passed_result(self):
        client = _make_iam_client(_all_required_actions())
        result = validate_iam_permissions(client, "Computer-Vision", FAKE_ROLE_ARN)

        assert result.verification_result == "PASSED"

    def test_permissions_missing_is_empty(self):
        client = _make_iam_client(_all_required_actions())
        result = validate_iam_permissions(client, "Computer-Vision", FAKE_ROLE_ARN)

        assert result.permissions_missing == []

    def test_simulate_called_for_s3_bucket_arn(self):
        client = _make_iam_client(_all_required_actions())
        validate_iam_permissions(client, "Computer-Vision", FAKE_ROLE_ARN)

        calls = client.simulate_principal_policy.call_args_list
        resource_arns_used = [
            call.kwargs.get("ResourceArns", call.args[2] if len(call.args) > 2 else [])[0]
            for call in calls
        ]
        assert S3_BUCKET_ARN in resource_arns_used

    def test_simulate_called_for_s3_objects_arn(self):
        client = _make_iam_client(_all_required_actions())
        validate_iam_permissions(client, "Computer-Vision", FAKE_ROLE_ARN)

        calls = client.simulate_principal_policy.call_args_list
        resource_arns_used = [
            call.kwargs.get("ResourceArns", call.args[2] if len(call.args) > 2 else [])[0]
            for call in calls
        ]
        assert S3_OBJECTS_ARN in resource_arns_used

    def test_simulate_called_for_sagemaker_wildcard(self):
        client = _make_iam_client(_all_required_actions())
        validate_iam_permissions(client, "Computer-Vision", FAKE_ROLE_ARN)

        calls = client.simulate_principal_policy.call_args_list
        resource_arns_used = [
            call.kwargs.get("ResourceArns", call.args[2] if len(call.args) > 2 else [])[0]
            for call in calls
        ]
        assert SAGEMAKER_RESOURCE in resource_arns_used

    def test_simulate_called_for_logs_arn(self):
        client = _make_iam_client(_all_required_actions())
        validate_iam_permissions(client, "Computer-Vision", FAKE_ROLE_ARN)

        calls = client.simulate_principal_policy.call_args_list
        resource_arns_used = [
            call.kwargs.get("ResourceArns", call.args[2] if len(call.args) > 2 else [])[0]
            for call in calls
        ]
        assert LOGS_RESOURCE in resource_arns_used

    def test_pass_role_checked_against_sagemaker_role_arn(self):
        client = _make_iam_client(_all_required_actions())
        validate_iam_permissions(client, "Computer-Vision", FAKE_ROLE_ARN)

        calls = client.simulate_principal_policy.call_args_list
        pass_role_call = next(
            (
                c for c in calls
                if "iam:PassRole" in (c.kwargs.get("ActionNames") or c.args[1])
            ),
            None,
        )
        assert pass_role_call is not None
        resource_arns = (
            pass_role_call.kwargs.get("ResourceArns")
            or pass_role_call.args[2]
        )
        assert resource_arns == [FAKE_ROLE_ARN]


# ---------------------------------------------------------------------------
# validate_iam_permissions — missing permissions path
# ---------------------------------------------------------------------------


class TestValidateIamPermissionsMissing:
    """Some permissions denied → PERMISSIONS_MISSING."""

    def test_returns_permissions_missing_result(self):
        # Deny all SageMaker actions
        allowed = _all_required_actions() - set(REQUIRED_SAGEMAKER_ACTIONS)
        client = _make_iam_client(allowed)
        result = validate_iam_permissions(client, "Computer-Vision", FAKE_ROLE_ARN)

        assert result.verification_result == "PERMISSIONS_MISSING"

    def test_missing_list_contains_denied_sagemaker_actions(self):
        allowed = _all_required_actions() - set(REQUIRED_SAGEMAKER_ACTIONS)
        client = _make_iam_client(allowed)
        result = validate_iam_permissions(client, "Computer-Vision", FAKE_ROLE_ARN)

        missing_actions = [
            stmt["Action"][0] for stmt in result.permissions_missing
        ]
        for action in REQUIRED_SAGEMAKER_ACTIONS:
            assert action in missing_actions

    def test_missing_list_does_not_contain_allowed_actions(self):
        # Only deny s3:DeleteObject
        denied = {"s3:DeleteObject"}
        allowed = _all_required_actions() - denied
        client = _make_iam_client(allowed)
        result = validate_iam_permissions(client, "Computer-Vision", FAKE_ROLE_ARN)

        missing_actions = {stmt["Action"][0] for stmt in result.permissions_missing}
        for action in allowed:
            assert action not in missing_actions

    def test_policy_statement_shape_is_valid(self):
        """Each element must be an AWS policy statement with Effect/Action/Resource."""
        allowed = _all_required_actions() - {"iam:PassRole"}
        client = _make_iam_client(allowed)
        result = validate_iam_permissions(client, "Computer-Vision", FAKE_ROLE_ARN)

        assert result.permissions_missing  # at least one entry
        for stmt in result.permissions_missing:
            assert stmt["Effect"] == "Allow"
            assert isinstance(stmt["Action"], list)
            assert len(stmt["Action"]) == 1
            assert isinstance(stmt["Resource"], str)

    def test_missing_pass_role_uses_sagemaker_role_arn_as_resource(self):
        allowed = _all_required_actions() - {"iam:PassRole"}
        client = _make_iam_client(allowed)
        result = validate_iam_permissions(client, "Computer-Vision", FAKE_ROLE_ARN)

        pass_role_stmt = next(
            (s for s in result.permissions_missing if s["Action"] == ["iam:PassRole"]),
            None,
        )
        assert pass_role_stmt is not None
        assert pass_role_stmt["Resource"] == FAKE_ROLE_ARN

    def test_all_permissions_missing(self):
        """When nothing is allowed, every required action appears in missing list."""
        client = _make_iam_client(set())  # deny everything
        result = validate_iam_permissions(client, "Computer-Vision", FAKE_ROLE_ARN)

        assert result.verification_result == "PERMISSIONS_MISSING"
        missing_actions = {stmt["Action"][0] for stmt in result.permissions_missing}

        for action in REQUIRED_S3_ACTIONS:
            assert action in missing_actions, f"{action} not in missing"
        for action in REQUIRED_SAGEMAKER_ACTIONS:
            assert action in missing_actions, f"{action} not in missing"
        for action in REQUIRED_LOGS_ACTIONS:
            assert action in missing_actions, f"{action} not in missing"
        assert "iam:PassRole" in missing_actions

    def test_missing_logs_actions_use_log_group_resource(self):
        allowed = _all_required_actions() - set(REQUIRED_LOGS_ACTIONS)
        client = _make_iam_client(allowed)
        result = validate_iam_permissions(client, "Computer-Vision", FAKE_ROLE_ARN)

        for stmt in result.permissions_missing:
            if stmt["Action"][0] in REQUIRED_LOGS_ACTIONS:
                assert stmt["Resource"] == LOGS_RESOURCE


# ---------------------------------------------------------------------------
# validate_iam_permissions — error handling / never raises
# ---------------------------------------------------------------------------


class TestValidateIamPermissionsNeverRaises:
    """Exceptions must be caught and encoded in the result."""

    def test_get_user_raises_client_error(self):
        client = MagicMock()
        client.get_user.side_effect = ClientError(
            {"Error": {"Code": "NoSuchEntity", "Message": "User not found"}},
            "GetUser",
        )
        result = validate_iam_permissions(client, "Computer-Vision", FAKE_ROLE_ARN)

        assert result.verification_result == "CONNECTION_FAILED"
        assert result.permissions_missing  # error info encoded

    def test_simulate_raises_exception(self):
        client = MagicMock()
        client.get_user.return_value = {"User": {"Arn": FAKE_USER_ARN}}
        client.simulate_principal_policy.side_effect = RuntimeError("network error")
        result = validate_iam_permissions(client, "Computer-Vision", FAKE_ROLE_ARN)

        assert result.verification_result == "CONNECTION_FAILED"

    def test_error_info_includes_error_type(self):
        client = MagicMock()
        client.get_user.side_effect = ValueError("unexpected value")
        result = validate_iam_permissions(client, "Computer-Vision", FAKE_ROLE_ARN)

        assert result.verification_result == "CONNECTION_FAILED"
        assert result.permissions_missing
        error_entry = result.permissions_missing[0]
        assert "error_type" in error_entry
        assert error_entry["error_type"] == "ValueError"

    def test_error_info_includes_error_message(self):
        client = MagicMock()
        client.get_user.side_effect = ValueError("something went wrong")
        result = validate_iam_permissions(client, "Computer-Vision", FAKE_ROLE_ARN)

        error_entry = result.permissions_missing[0]
        assert "error" in error_entry
        assert "something went wrong" in error_entry["error"]


# ---------------------------------------------------------------------------
# run_validator — CLI entrypoint
# ---------------------------------------------------------------------------


class TestRunValidatorCLI:
    """Tests for the run_validator() CLI entrypoint."""

    _VALID_ENV = {
        "AWS_ACCESS_KEY_ID": "AKIAIOSFODNN7EXAMPLE",
        "AWS_SECRET_ACCESS_KEY": "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY",
        "AWS_DEFAULT_REGION": "ap-southeast-2",
        "SAGEMAKER_EXECUTION_ROLE_ARN": FAKE_ROLE_ARN,
    }

    def _run_with_patched_env(self, iam_client_mock, capsys):
        """Helper: patch env loading + boto3 client creation, run run_validator."""
        with (
            patch(
                "scripts.iam_validator.load_env_and_validate",
                return_value=self._VALID_ENV,
            ),
            patch("scripts.iam_validator.boto3.client", return_value=iam_client_mock),
            pytest.raises(SystemExit) as exc_info,
        ):
            run_validator()
        return exc_info.value.code

    # -----------------------------------------------------------------------
    # PASSED path
    # -----------------------------------------------------------------------

    def test_exits_zero_on_passed(self, capsys):
        client = _make_iam_client(_all_required_actions())
        exit_code = self._run_with_patched_env(client, capsys)
        assert exit_code == 0

    def test_prints_iam_validation_passed_on_success(self, capsys):
        client = _make_iam_client(_all_required_actions())
        self._run_with_patched_env(client, capsys)
        captured = capsys.readouterr()
        assert "IAM validation passed" in captured.out

    # -----------------------------------------------------------------------
    # PERMISSIONS_MISSING path
    # -----------------------------------------------------------------------

    def test_exits_one_on_permissions_missing(self, capsys):
        allowed = _all_required_actions() - {"iam:PassRole"}
        client = _make_iam_client(allowed)
        exit_code = self._run_with_patched_env(client, capsys)
        assert exit_code == 1

    def test_json_output_contains_permissions_missing_field(self, capsys):
        allowed = _all_required_actions() - {"iam:PassRole", "sagemaker:CreateModel"}
        client = _make_iam_client(allowed)
        self._run_with_patched_env(client, capsys)
        captured = capsys.readouterr()
        output = json.loads(captured.out)
        assert output["verification_result"] == "PERMISSIONS_MISSING"
        assert "permissions_missing" in output
        assert isinstance(output["permissions_missing"], list)
        assert len(output["permissions_missing"]) > 0

    def test_json_output_statements_are_attachable(self, capsys):
        """Each statement should have Effect/Action/Resource for direct attachment."""
        allowed = _all_required_actions() - set(REQUIRED_LOGS_ACTIONS)
        client = _make_iam_client(allowed)
        self._run_with_patched_env(client, capsys)
        captured = capsys.readouterr()
        output = json.loads(captured.out)
        for stmt in output["permissions_missing"]:
            assert "Effect" in stmt
            assert "Action" in stmt
            assert "Resource" in stmt

    # -----------------------------------------------------------------------
    # CONNECTION_FAILED path — NoCredentialsError
    # -----------------------------------------------------------------------

    def test_exits_one_on_no_credentials_error(self, capsys):
        with (
            patch(
                "scripts.iam_validator.load_env_and_validate",
                return_value=self._VALID_ENV,
            ),
            patch(
                "scripts.iam_validator.boto3.client",
                side_effect=NoCredentialsError(),
            ),
            pytest.raises(SystemExit) as exc_info,
        ):
            run_validator()
        assert exc_info.value.code == 1

    def test_connection_failed_output_names_credential_env_vars(self, capsys):
        with (
            patch(
                "scripts.iam_validator.load_env_and_validate",
                return_value=self._VALID_ENV,
            ),
            patch(
                "scripts.iam_validator.boto3.client",
                side_effect=NoCredentialsError(),
            ),
            pytest.raises(SystemExit),
        ):
            run_validator()
        captured = capsys.readouterr()
        output = json.loads(captured.out)
        assert output["verification_result"] == "CONNECTION_FAILED"
        # The output must identify the credential source env var names
        output_str = json.dumps(output)
        assert "AWS_ACCESS_KEY_ID" in output_str

    # -----------------------------------------------------------------------
    # CONNECTION_FAILED path — authentication ClientError
    # -----------------------------------------------------------------------

    def test_exits_one_on_auth_client_error(self, capsys):
        auth_error = ClientError(
            {"Error": {"Code": "InvalidClientTokenId", "Message": "token invalid"}},
            "GetUser",
        )
        client = MagicMock()
        client.get_user.side_effect = auth_error

        with (
            patch(
                "scripts.iam_validator.load_env_and_validate",
                return_value=self._VALID_ENV,
            ),
            patch("scripts.iam_validator.boto3.client", return_value=client),
            pytest.raises(SystemExit) as exc_info,
        ):
            run_validator()
        assert exc_info.value.code == 1

    def test_connection_failed_result_on_auth_client_error(self, capsys):
        """
        InvalidClientTokenId from get_user is caught inside validate_iam_permissions
        and returned as CONNECTION_FAILED (never raises to run_validator).
        """
        auth_error = ClientError(
            {"Error": {"Code": "InvalidClientTokenId", "Message": "token invalid"}},
            "GetUser",
        )
        client = MagicMock()
        client.get_user.side_effect = auth_error

        with (
            patch(
                "scripts.iam_validator.load_env_and_validate",
                return_value=self._VALID_ENV,
            ),
            patch("scripts.iam_validator.boto3.client", return_value=client),
            pytest.raises(SystemExit),
        ):
            run_validator()

        captured = capsys.readouterr()
        output = json.loads(captured.out)
        assert output["verification_result"] == "CONNECTION_FAILED"


# ---------------------------------------------------------------------------
# ValidationResult dataclass
# ---------------------------------------------------------------------------


class TestValidationResultDataclass:
    def test_passed_result_has_empty_permissions_missing(self):
        result = ValidationResult(verification_result="PASSED")
        assert result.permissions_missing == []

    def test_permissions_missing_result_stores_statements(self):
        stmts = [{"Effect": "Allow", "Action": ["s3:GetObject"], "Resource": "*"}]
        result = ValidationResult(
            verification_result="PERMISSIONS_MISSING",
            permissions_missing=stmts,
        )
        assert result.permissions_missing == stmts

    def test_connection_failed_result(self):
        result = ValidationResult(
            verification_result="CONNECTION_FAILED",
            permissions_missing=[{"error": "no network", "error_type": "OSError"}],
        )
        assert result.verification_result == "CONNECTION_FAILED"
