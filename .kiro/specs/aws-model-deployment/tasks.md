# Implementation Plan: AWS Model Deployment

## Overview

Implement the full AWS infrastructure layer for the retail computer vision pipeline:
shared utilities, S3 sync tool, IAM validator, deployment architecture docs, model
packager, SageMaker inference handler, batch transform script, real-time endpoint
deploy script, CloudWatch monitor, and the handoff contract doc. All scripts are
Python with `boto3` and `python-dotenv`. Tests use `pytest` and `hypothesis`.

---

## Tasks

- [ ] 1. Project scaffolding — directory structure, .gitignore, shared utilities
  - [x] 1.1 Create directory layout and update `.gitignore`
    - Create `scripts/`, `scripts/common/`, `code/`, `docs/`, `tests/`, `models/`, `data/` directories
    - Add `.env`, `models/`, and `data/` to `.gitignore`
    - Create `scripts/common/__init__.py` (empty)
    - _Requirements: 9.6_

  - [-] 1.2 Implement `scripts/common/env_loader.py`
    - Implement `load_env_and_validate(required_vars: list[str]) -> dict[str, str]`
    - Call `dotenv.load_dotenv(find_dotenv(), override=False)`; raise `SystemExit(1)` if `.env` absent
    - Check each variable; print `Missing required environment variable(s): {names}` and exit 1 if any are absent/empty
    - Never print credential values — only variable names
    - _Requirements: 9.6_

  - [-] 1.3 Implement `scripts/common/retry.py`
    - Implement `exponential_backoff_retry(fn, max_retries=3, base_seconds=1.0, retryable_exceptions=(ClientError, ConnectionError, TimeoutError))`
    - Attempt 0: call `fn()` immediately; on retryable exception wait `base_seconds * (2 ** attempt)` then retry
    - Re-raise last exception after `max_retries` exhausted; non-retryable exceptions propagate immediately
    - _Requirements: 1.5, 6.9_

  - [ ]* 1.4 Write unit tests for `env_loader` and `retry` utilities
    - Test `load_env_and_validate` exits 1 when `.env` is absent
    - Test `load_env_and_validate` exits 1 and names every missing variable for any subset of missing vars
    - Test `exponential_backoff_retry` succeeds on first attempt, retries on retryable exceptions, raises after exhaustion
    - _Requirements: 9.6, 1.5_

- [ ] 2. S3_Sync_Tool — download mode (`scripts/s3_sync.py`)
  - [-] 2.1 Implement argument parsing and credential bootstrap for `s3_sync.py`
    - Add CLI parser with `--subset`, `--max-files`, `--source` (default `images`), `--store-ids`, `--local-dest`
    - Validate `--max-files > 0`; print error and exit 1 if `≤ 0`
    - Validate `--source` is `images` or `videos`; print error listing valid values and exit 1 otherwise
    - Call `load_env_and_validate` at entry point
    - _Requirements: 1.2, 1.9_

  - [-] 2.2 Implement `compute_etag_match` and `exponential_backoff_retry` usage in download
    - Implement `compute_etag_match(local_path: Path, s3_etag: str) -> bool`
    - Return `False` for multipart ETags (contain `-`) — conservative re-download
    - Return `True` only when `MD5(local_path).hexdigest() == s3_etag.strip('"')`
    - _Requirements: 1.4_

  - [-] 2.3 Implement `download_dataset` core logic
    - List objects under `extracted_images_and_labels/data/{subset}/` or `extracted_videos/{store_id}/`
    - Exit 1 with error message including missing path when `--subset` or `--store-ids` path does not exist in S3
    - Skip files where `compute_etag_match` returns True; re-download when False
    - Apply `--max-files` limit to new downloads; stop after limit is reached
    - Use `exponential_backoff_retry` for each download; exit 1 after exhausted retries
    - _Requirements: 1.1, 1.3, 1.4, 1.5, 1.6, 1.10_

  - [~] 2.4 Implement `download_manifest.json` writer
    - Write `download_manifest.json` to `local_dest` after every invocation (even zero downloads)
    - Include `invocation_timestamp`, `source`, `subset`, `total_files_downloaded`, `total_files_skipped`, `files` array
    - `files` contains only newly transferred files (not skipped), each with `s3_key` and `size_bytes`
    - `subset` is `null` when `--subset` was not provided
    - _Requirements: 1.7, 1.8_

  - [ ]* 2.5 Write property test for invalid `--max-files` rejection (Property 1)
    - **Property 1: Invalid --max-files values are always rejected**
    - Use `st.integers(max_value=0)` to generate invalid values; assert non-zero exit and error message
    - **Validates: Requirements 1.2**

  - [ ]* 2.6 Write property test for ETag skip/re-download consistency (Property 2)
    - **Property 2: ETag-based skip/re-download is consistent for all files**
    - Generate arbitrary file content and matching/non-matching ETags; assert skip or re-download accordingly
    - **Validates: Requirements 1.4**

  - [ ]* 2.7 Write property test for download manifest accuracy (Property 3)
    - **Property 3: Download manifest accurately reflects every download run**
    - Generate a set of files with varying ETag match statuses; assert manifest counts and `files` array match
    - **Validates: Requirements 1.7**

  - [ ]* 2.8 Write property test for invalid `--source` rejection (Property 4)
    - **Property 4: Any invalid --source value is rejected**
    - Use `st.text()` filtered to exclude `"images"` and `"videos"`; assert non-zero exit and valid-values message
    - **Validates: Requirements 1.9**

  - [ ]* 2.9 Write unit tests for download mode
    - Test default `--source images` behavior (no `--source` flag)
    - Test `--subset` omitted downloads entire `extracted_images_and_labels/data/`
    - Test missing S3 subset path exits 1 with path in error message
    - Test all-files-skipped scenario: exit 0, manifest with `total_files_downloaded=0`
    - _Requirements: 1.3, 1.6, 1.7, 1.8_

- [ ] 3. S3_Sync_Tool — upload mode (`scripts/s3_sync.py` continued)
  - [x] 3.1 Implement `validate_run_name` and `generate_run_name`
    - `validate_run_name(name: str)`: raise `ValueError` if name contains chars outside `[a-zA-Z0-9_\-]` or exceeds 128 chars
    - `generate_run_name()`: return `run_{YYYYMMDDTHHmmssZ}` using `datetime.utcnow()`; print to stdout
    - Add `--run-name` and `--overwrite` CLI flags; wire to upload mode
    - _Requirements: 2.2, 2.3_

  - [~] 3.2 Implement `upload_model` core logic
    - Validate `.pt` extension and file existence; exit 1 with error identifying invalid path otherwise
    - Check for existing S3 object at `models/{run_name}/best.pt`; exit 1 with existing key unless `--overwrite` is set
    - Upload using `abort_multipart_on_failure` context manager; verify S3 object size == local file size post-upload
    - Print full S3 URI on success; exit 1 with size-mismatch error if sizes differ
    - _Requirements: 2.1, 2.4, 2.5, 2.7_

  - [~] 3.3 Implement `abort_multipart_on_failure` context manager in `scripts/common/retry.py`
    - Track multipart upload ID; on any exception inside the block call `s3.abort_multipart_upload`
    - Ensure no orphaned parts remain in S3 after any failure
    - _Requirements: 2.7_

  - [~] 3.4 Implement `run_metadata.json` upload
    - After successful model upload, build `run_metadata.json` with `run_name`, `upload_timestamp` (UTC ISO 8601), `local_file_size_bytes`, `python_script_version`
    - Upload to `s3://retaildata-cv-2026/models/{run_name}/run_metadata.json`
    - If metadata upload fails: print error, exit 1; leave already-uploaded model file intact (do NOT delete)
    - _Requirements: 2.6, 2.8_

  - [ ]* 3.5 Write property test for `run_name` validation (Property 5)
    - **Property 5: run_name validation rejects all non-conforming strings**
    - Use `st.text()` with invalid characters or length > 128 to assert rejection; valid strings within bounds assert acceptance
    - **Validates: Requirements 2.2, 5.10**

  - [ ]* 3.6 Write property test for upload size verification (Property 6)
    - **Property 6: Upload size verification holds for all files**
    - Mock S3 to return matching and mismatching sizes; assert exit 0 on match and exit 1 on mismatch
    - **Validates: Requirements 2.5**

  - [ ]* 3.7 Write property test for `run_metadata.json` required fields (Property 7)
    - **Property 7: run_metadata.json always contains required fields after upload**
    - For any successful upload, assert returned metadata dict contains `run_name`, `upload_timestamp`, `local_file_size_bytes`, `python_script_version` with correct types
    - **Validates: Requirements 2.6**

  - [ ]* 3.8 Write unit tests for upload mode
    - Test `--overwrite` guard: exit 1 with existing S3 key when object exists and flag absent
    - Test auto-generated run name format matches `run_{YYYYMMDDTHHmmssZ}`
    - Test successful upload prints full S3 URI and exits 0
    - Test metadata-upload failure after model success: model stays, script exits 1
    - _Requirements: 2.3, 2.4, 2.5, 2.8_

- [~] 4. Checkpoint — Ensure all tests pass for S3_Sync_Tool
  - Ensure all tests in `tests/test_s3_sync.py` pass, ask the user if questions arise.

- [ ] 5. IAM_Validator (`scripts/iam_validator.py`)
  - [-] 5.1 Implement env-var validation and AWS client bootstrap
    - Require `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_DEFAULT_REGION`, `SAGEMAKER_EXECUTION_ROLE_ARN` via `load_env_and_validate`
    - Print missing variable names and exit 1 before any AWS call if any are absent/empty
    - Catch credential/connection errors; set `verification_result = CONNECTION_FAILED` and print credential source (env var names)
    - _Requirements: 3.8, 3.9_

  - [-] 5.2 Implement `validate_iam_permissions` using `iam:SimulatePrincipalPolicy`
    - Define `REQUIRED_S3_ACTIONS`, `REQUIRED_SAGEMAKER_ACTIONS`, `REQUIRED_LOGS_ACTIONS` constants
    - Simulate each action group against its required resource ARN; collect denied actions
    - Simulate `iam:PassRole` on `SAGEMAKER_EXECUTION_ROLE_ARN`
    - Build `permissions_missing` list of AWS policy statement dicts for each absent action
    - Return `ValidationResult` dataclass with `verification_result` and `permissions_missing`
    - _Requirements: 3.1, 3.2, 3.3, 3.4_

  - [~] 5.3 Implement `run_validator` CLI entrypoint and JSON output
    - On `PERMISSIONS_MISSING`: print JSON with `verification_result` and `permissions_missing` array; exit 1
    - On `PASSED`: print `IAM validation passed`; exit 0
    - On `CONNECTION_FAILED`: print JSON with `verification_result` and descriptive error; exit 1
    - _Requirements: 3.5, 3.6, 3.7_

  - [ ]* 5.4 Write property test for missing permissions output (Property 8)
    - **Property 8: Missing permissions output contains exactly the absent permissions**
    - Use `st.frozensets(st.sampled_from(ALL_REQUIRED_ACTIONS))` to generate absent action subsets
    - Assert `permissions_missing` array covers exactly those absent actions — no more, no fewer
    - **Validates: Requirements 3.5**

  - [ ]* 5.5 Write property test for missing env vars reporting (Property 9)
    - **Property 9: Any combination of missing env vars is fully reported**
    - Generate all non-empty subsets of required env vars; assert all missing names appear in error output and exit code is non-zero
    - **Validates: Requirements 3.9, 9.6**

  - [ ]* 5.6 Write unit tests for IAM_Validator
    - Test IAM validation success path: exit 0 and prints `IAM validation passed`
    - Test connection failure: `CONNECTION_FAILED` result shape with credential source info
    - Test all-missing-permissions case produces valid attachable policy statements
    - _Requirements: 3.6, 3.7_

- [ ] 6. Deployment architecture doc (`docs/deployment_architecture.md`)
  - [-] 6.1 Write architecture evaluation table and primary architecture selection
    - Create `docs/deployment_architecture.md`
    - Write comparison table: SageMaker Batch Transform, SageMaker Real-Time Endpoint, Lambda + API Gateway
    - Columns: cost per inference, cold-start latency tolerance, input payload size, GPU availability, operational complexity; verdict per cell
    - Document selection of SageMaker Batch Transform with justifications (a) read-after-completion pattern, (b) no always-on cost
    - _Requirements: 4.1, 4.2_

  - [x] 6.2 Write instance type justification and optional real-time config
    - Document `ml.g4dn.xlarge` justification: NVIDIA T4 GPU, ~3–8 ms/frame on GPU vs ~150–350 ms/frame on CPU-only (`ml.m5.xlarge`)
    - Include batch-of-1000-frames comparison (GPU: ~5–8 s total; CPU: ~150–350 s total)
    - Document optional real-time endpoint config: instance type `ml.g4dn.xlarge`, payload format (single JPEG/PNG ≤20 MB, ≤1920×1080), cold-start tolerance ≤30 s
    - Note that endpoint should be deleted after 30 minutes of idle time
    - _Requirements: 4.3, 4.4, 4.5_

- [ ] 7. Model_Packager (`scripts/package_model.py`)
  - [x] 7.1 Implement `validate_training_summary`
    - Load and parse `training_summary.json`; validate presence and types of all six required fields
    - `num_classes` must be `int > 0`; `class_names` must be `list[str]` with `len == num_classes`
    - Raise `ValueError` identifying the first invalid/missing field; exit 1
    - _Requirements: 9.2_

  - [~] 7.2 Implement `create_model_archive`
    - Accept `--model-path` and `--run-name` CLI args; validate `--run-name` via `validate_run_name`
    - Verify `best.pt`, `code/inference.py`, `code/requirements.txt` exist; exit 1 naming every missing file
    - Generate `model_config.json` from training summary (`num_classes`, `class_names`, `input_size`)
    - Build `model.tar.gz` with SageMaker layout: `best.pt`, `code/inference.py`, `code/requirements.txt`, `code/model_config.json`
    - _Requirements: 5.1, 5.2, 5.5, 9.2, 9.3_

  - [~] 7.3 Implement `verify_archive_integrity`
    - Extract archive to `tempfile.mkdtemp()`; assert `best.pt` exists and size ≥ 1024 bytes
    - Assert `code/inference.py` exists and size ≥ 1 byte; raise `ValueError` identifying failing file
    - Clean up temp dir on exit (success or failure)
    - _Requirements: 5.4_

  - [~] 7.4 Implement `upload_archive` with partial-upload protection
    - Upload `model.tar.gz` to `s3://retaildata-cv-2026/models/{run_name}/model.tar.gz`
    - Use `abort_multipart_on_failure` context manager; raise descriptive error on failure
    - Return full S3 URI on success
    - _Requirements: 5.3_

  - [ ]* 7.5 Write property test for `model.tar.gz` contents (Property 10)
    - **Property 10: model.tar.gz always contains all required files**
    - For any valid input file set, assert archive contains `best.pt`, `code/inference.py`, `code/requirements.txt`
    - **Validates: Requirements 5.1**

  - [ ]* 7.6 Write property test for archive integrity thresholds (Property 11)
    - **Property 11: Archive integrity check enforces minimum size thresholds for all archives**
    - Generate archives with undersized or absent `best.pt` / `code/inference.py`; assert descriptive `ValueError` raised before upload
    - **Validates: Requirements 5.4**

  - [ ]* 7.7 Write property test for missing packaging inputs (Property 12)
    - **Property 12: Missing required packaging inputs always cause early failure**
    - Generate any combination of missing files from `{best.pt, code/inference.py, code/requirements.txt}`; assert error identifies every missing file and no archive is created
    - **Validates: Requirements 5.5**

  - [ ]* 7.8 Write property test for `training_summary.json` validation (Property 21)
    - **Property 21: training_summary.json field validation identifies all missing/invalid fields**
    - Generate JSON documents missing fields or with wrong types; assert `ValueError` names the specific field
    - **Validates: Requirements 9.2**

  - [ ]* 7.9 Write property test for `model_config.json` values in archive (Property 22)
    - **Property 22: model_config.json embedded in archive matches training_summary values**
    - For any valid training summary, assert embedded `model_config.json` has identical `num_classes`, `class_names`, `input_size`
    - **Validates: Requirements 9.3**

  - [ ]* 7.10 Write unit tests for Model_Packager
    - Test invalid `--run-name` (bad chars, too long): exit 1 with constraint description
    - Test missing file: exit 1 with file name in error message, no archive created
    - Test archive integrity failure: upload NOT attempted when `best.pt` < 1 KB
    - _Requirements: 5.2, 5.3, 5.4, 5.5, 5.10_

- [~] 8. Checkpoint — Ensure all tests pass for IAM_Validator and Model_Packager
  - Ensure all tests in `tests/test_iam_validator.py` and `tests/test_package_model.py` pass, ask the user if questions arise.

- [ ] 9. Inference handler (`code/inference.py`)
  - [x] 9.1 Implement `model_fn` with class-count validation and error state
    - Load `best.pt` using `ultralytics.YOLO(model_path)` from `model_dir`
    - Read `model_config.json` from `model_dir`; validate `model.model.nc == config['num_classes']`
    - On mismatch: log structured JSON error with `model_class_count`, `expected_num_classes`, path to `model_config.json`; set `_ERROR_STATE = True`
    - Return the YOLO model object
    - _Requirements: 5.6, 9.4_

  - [~] 9.2 Implement `input_fn` with image decoding and normalization
    - Accept `content_type` in `{'image/jpeg', 'image/png'}`; raise `ValueError('Unsupported content type: {content_type}')` for any other type
    - Decode bytes via `Pillow Image.open(BytesIO(request_body))`; convert to RGB; resize to (416, 416)
    - Convert to `float32` tensor; normalize pixel values to `[0.0, 1.0]` by dividing by 255.0
    - Return tensor of shape `[1, 3, 416, 416]`
    - _Requirements: 5.7, 5.8_

  - [~] 9.3 Implement `predict_fn` and `output_fn` with structured error logging
    - `predict_fn`: record start time, run `model(input_tensor)`, record end time; return dict with raw results and `inference_time_ms`
    - `output_fn`: serialize prediction to `InferenceResponse` JSON (`detections`, `inference_time_ms`, `model_run_name`)
    - On exception from `predict_fn`: log structured JSON with `level`, `error_type`, `error_message`, `input_shape`, `timestamp`; return HTTP 500 without re-raising
    - If `_ERROR_STATE` is True: immediately return HTTP 500 directing operator to redeploy
    - If log write itself fails: catch secondary exception and still return HTTP 500
    - _Requirements: 5.9, 5.11, 8.2_

  - [x] 9.4 Create `code/requirements.txt` with pinned dependencies
    - Add `ultralytics>=8.0,<9`, `torch>=2.1,<3`, `Pillow>=10.0,<11` with pinned version ranges
    - _Requirements: 5.1_

  - [ ]* 9.5 Write property test for `input_fn` tensor normalization (Property 13)
    - **Property 13: input_fn produces correctly normalised tensors for all valid images**
    - Use `st.binary()` wrapped in valid JPEG/PNG headers to generate arbitrary images; assert output dtype is `float32`, shape is `[1, 3, 416, 416]`, all values in `[0.0, 1.0]`
    - **Validates: Requirements 5.7**

  - [ ]* 9.6 Write property test for unsupported content type rejection (Property 14)
    - **Property 14: Unsupported Content-Type always triggers an error response**
    - Use `st.from_regex` to generate content-type strings excluding `image/jpeg` and `image/png`; assert `ValueError` raised and HTTP 400 returned without invoking model
    - **Validates: Requirements 5.8**

  - [ ]* 9.7 Write property test for `output_fn` response structure (Property 15)
    - **Property 15: output_fn always produces a structurally valid JSON response**
    - Generate mock prediction outputs; assert returned JSON parses to dict with `detections`, `inference_time_ms > 0.0`, `model_run_name`; assert each detection has valid `label`, `confidence` in [0,1], `bbox` of four floats in [0,1]
    - **Validates: Requirements 5.9**

  - [ ]* 9.8 Write property test for `predict_fn` exception handling (Property 16)
    - **Property 16: predict_fn exceptions always yield HTTP 500 with structured error log**
    - Raise arbitrary exception types from `predict_fn`; assert `output_fn` catches them, logs entry with `error_type`, `error_message`, `timestamp`, and returns HTTP 500 without propagating
    - **Validates: Requirements 5.11**

  - [ ]* 9.9 Write property test for class count mismatch error state (Property 23)
    - **Property 23: Class count mismatch always prevents inference and returns HTTP 500**
    - For any pair `(model_class_count, config_num_classes)` where values differ, assert `model_fn` logs structured error, sets `_ERROR_STATE`, and all subsequent `output_fn` calls return HTTP 500
    - **Validates: Requirements 9.4**

  - [ ]* 9.10 Write property test for inference error log fields (Property 19)
    - **Property 19: inference handler logs always contain required fields on error**
    - For any unhandled exception, assert logged JSON entry contains `level`, `error_type`, `error_message`, `input_shape`, `timestamp`
    - **Validates: Requirements 8.2**

  - [ ]* 9.11 Write unit tests for inference handler
    - Test successful inference path: `output_fn` returns valid JSON and `application/json`
    - Test `_ERROR_STATE = True`: all requests return HTTP 500 with redeploy message
    - Test log-write failure: HTTP 500 returned without raising secondary exception
    - _Requirements: 5.6, 5.9, 5.11, 8.2, 9.4_

- [~] 10. Checkpoint — Ensure all tests pass for inference handler
  - Ensure all tests in `tests/test_inference.py` pass, ask the user if questions arise.

- [ ] 11. Batch transform script (`scripts/run_batch_transform.py`)
  - [ ] 11.1 Implement argument parsing and `submit_batch_transform`
    - Add `--input-s3-uri`, `--output-s3-uri`, `--model-name`, `--instance-type` (default `ml.g4dn.xlarge`) CLI args
    - Exit 1 naming each missing required arg if `--input-s3-uri`, `--output-s3-uri`, or `--model-name` is absent
    - Call `CreateTransformJob` with job name `retail-cv-batch-{timestamp}`, tags `[{Project: retail-cv-analytics}, {ManagedBy: aws-model-deployment-spec}]`, `BatchStrategy: SingleRecord`, `TransformInput.SplitType: None`
    - On `CreateTransformJob` failure: print API error and exit 1 without entering polling loop
    - _Requirements: 6.2, 6.3, 6.8, 6.10_

  - [~] 11.2 Implement `poll_until_terminal` polling loop
    - Poll `DescribeTransformJob` every 60 seconds; print current status and job name on each poll
    - Exit 1 if total polling duration exceeds 24 hours (86 400 seconds)
    - On `DescribeTransformJob` exception during polling: print exception message, retry up to 3 times with 30-second fixed interval; exit 1 after all retries fail
    - Stop polling on terminal state (`Completed`, `Failed`, `Stopped`)
    - _Requirements: 6.4, 6.5, 6.9_

  - [~] 11.3 Implement `handle_terminal_state`
    - `Completed`: print S3 output URI and total job duration in seconds; exit 0
    - `Failed`/`Stopped`: retrieve `FailureReason` from SageMaker API; print `Failure reason not available` if absent or null; exit non-zero
    - _Requirements: 6.6, 6.7_

  - [ ]* 11.4 Write property test for required job tags (Property 17)
    - **Property 17: Batch transform jobs always include required tags**
    - For any valid invocation, mock `CreateTransformJob` and assert call includes both `Project=retail-cv-analytics` and `ManagedBy=aws-model-deployment-spec` tags
    - **Validates: Requirements 6.8**

  - [ ]* 11.5 Write property test for polling exception retry behavior (Property 18)
    - **Property 18: DescribeTransformJob polling exceptions always trigger 3 retries**
    - For any exception type raised by `DescribeTransformJob`, assert exactly 3 retry attempts with 30-second interval, then exit non-zero
    - **Validates: Requirements 6.9**

  - [ ]* 11.6 Write unit tests for batch transform script
    - Test default `--instance-type ml.g4dn.xlarge` used when flag omitted
    - Test `Completed` terminal state: output URI and duration printed, exit 0
    - Test `Failed` terminal state with null failure reason: prints `Failure reason not available`, exits non-zero
    - Test 24-hour polling timeout: exits 1
    - _Requirements: 6.3, 6.6, 6.7_

- [ ] 12. Real-time endpoint deploy script (`scripts/deploy_realtime.py`)
  - [~] 12.1 Implement `deploy_realtime_endpoint` — create and poll
    - Accept `--model-name`, `--endpoint-name`, `--instance-type` (default `ml.g4dn.xlarge`), `--initial-instance-count` (default 1) CLI args
    - Create `EndpointConfig` then `Endpoint` via boto3
    - Poll `DescribeEndpoint` until status is `InService` or `Failed`
    - On success: confirm `InService` via `DescribeEndpoint`; print endpoint name, status, and HTTPS invocation URL
    - On failure: retrieve `FailureReason` from `DescribeEndpoint`; print it and exit non-zero
    - _Requirements: 7.4, 7.5_

  - [~] 12.2 Implement auto-scaling registration
    - Register Application Auto Scaling for the endpoint variant
    - ScaleOut policy: `InvocationsPerInstance > 10 req/min`, 1-period 1-min evaluation
    - ScaleIn policy: `InvocationsPerInstance < 5 req/min` for 5 consecutive 1-min periods
    - Min instances: 1; max instances: 4
    - _Requirements: 7.2_

  - [~] 12.3 Add idle-endpoint note to script docstring/README
    - Add note in script docstring and `docs/deployment_architecture.md` that the endpoint should be deleted after 30 minutes of idle time (zero invocations) to avoid unnecessary cost
    - _Requirements: 7.3_

  - [ ]* 12.4 Write unit tests for deploy_realtime endpoint script
    - Test endpoint creation failure: `FailureReason` retrieved and printed, exit non-zero
    - Test `InService` confirmation: prints endpoint name, status, HTTPS URL
    - Test auto-scaling registration called with correct min/max instance counts and thresholds
    - _Requirements: 7.2, 7.4, 7.5_

- [ ] 13. CloudWatch_Monitor (`scripts/setup_cloudwatch.py`)
  - [~] 13.1 Implement `ensure_log_group` (idempotent)
    - Create log group if absent; update retention policy to 30 days if it differs from existing
    - Never recreate an existing log group
    - Create `/aws/sagemaker/Endpoints/retail-cv-inference` for real-time; `/aws/sagemaker/TransformJobs/retail-cv-batch` for batch
    - _Requirements: 8.1_

  - [~] 13.2 Implement `put_latency_alarm` (idempotent)
    - Create or update alarm `retail-cv-latency-p99` on `ModelLatency` metric (namespace `AWS/SageMaker`)
    - `ExtendedStatistic=p99`, `Threshold=4000 ms`, `Period=300s`, `EvaluationPeriods=1`, `DatapointsToAlarm=1`, `ComparisonOperator=GreaterThanThreshold`, `TreatMissingData=notBreaching`
    - Add SNS action if `CLOUDWATCH_ALARM_SNS_ARN` env var is set; print warning to stdout if unset
    - _Requirements: 8.3, 8.7_

  - [~] 13.3 Implement `put_5xx_alarm` (idempotent)
    - Create or update alarm `retail-cv-5xx-errors` on `Invocation5XXErrors` metric
    - `Statistic=Sum`, `Threshold=5`, `Period=300s`, `EvaluationPeriods=1`, `DatapointsToAlarm=1`, `TreatMissingData=notBreaching`
    - _Requirements: 8.4_

  - [~] 13.4 Implement `put_dashboard` (idempotent)
    - Create or update dashboard `retail-cv-inference-dashboard`
    - Widgets: `ModelLatency` (p50, p90, p99), `Invocations`, `Invocation5XXErrors`
    - Final widget: `GPUUtilization` if `use_gpu_metrics=True` else `CPUUtilization`
    - Use `put_dashboard` CloudWatch API (idempotent by design)
    - _Requirements: 8.5_

  - [~] 13.5 Implement `run_setup` CLI entrypoint
    - Accept `--endpoint-name` and `--deployment-type` (`realtime` or `batch`) args
    - Read optional `CLOUDWATCH_ALARM_SNS_ARN` from env; call `ensure_log_group`, `put_latency_alarm`, `put_5xx_alarm`, `put_dashboard`
    - _Requirements: 8.1, 8.6, 8.7_

  - [ ]* 13.6 Write property test for CloudWatch monitor idempotency (Property 20)
    - **Property 20: CloudWatch monitor setup is idempotent for any number of invocations**
    - Mock CloudWatch client; run setup script multiple times with pre-existing alarms/dashboards; assert no duplicate resources created and final config matches declared config
    - **Validates: Requirements 8.6**

  - [ ]* 13.7 Write unit tests for CloudWatch_Monitor
    - Test `put_metric_alarm` called with exact expected parameters (threshold, period, evaluation periods)
    - Test `put_dashboard` produces correct widget JSON including p50/p90/p99 for ModelLatency
    - Test SNS ARN unset: alarm created without SNS action, warning printed to stdout
    - Test log group update: retention updated if different, not recreated
    - _Requirements: 8.1, 8.3, 8.4, 8.5, 8.7_

- [~] 14. Checkpoint — Ensure all tests pass for batch transform, real-time endpoint, and CloudWatch monitor
  - Ensure all tests in `tests/test_batch_transform.py`, `tests/test_cloudwatch_monitor.py` pass, ask the user if questions arise.

- [ ] 15. Handoff contract doc and pytest configuration
  - [~] 15.1 Write `docs/handoff_contract.md`
    - Specify that local training pipeline MUST produce `best.pt` in Ultralytics YOLO format and `training_summary.json` before `Model_Packager` is invoked
    - Document all six required `training_summary.json` fields with types and constraints
    - Specify the invocation sequence: (1) run `iam_validator.py`, (2) run `s3_sync.py --upload-model`, (3) run `package_model.py`, (4) run `run_batch_transform.py` or `deploy_realtime.py`, (5) run `setup_cloudwatch.py`
    - Include step-by-step checklist of actions the engineer must complete before running deployment scripts
    - _Requirements: 9.1, 9.2, 9.5_

  - [-] 15.2 Create `pytest.ini` and `tests/conftest.py`
    - Create `pytest.ini` with `addopts = --tb=short`
    - Create `tests/conftest.py` with Hypothesis settings: `max_examples=100`, `deadline=None`; register `ci` profile with `max_examples=200`
    - Create empty `tests/test_s3_sync.py`, `tests/test_iam_validator.py`, `tests/test_package_model.py`, `tests/test_inference.py`, `tests/test_batch_transform.py`, `tests/test_cloudwatch_monitor.py` stub files
    - _Requirements: (testing infrastructure)_

- [~] 16. Final checkpoint — Full test suite passes
  - Run `pytest tests/` and ensure all tests pass, ask the user if questions arise.

---

## Notes

- Tasks marked with `*` are optional and can be skipped for faster MVP delivery
- Each task references specific requirements for traceability
- Checkpoints ensure incremental validation at logical milestones
- Property tests (Properties 1–23) validate universal correctness guarantees using Hypothesis
- Unit tests validate specific examples, defaults, and edge cases
- All scripts use `scripts/common/env_loader.py` and `scripts/common/retry.py` — implement these first (Task 1)
- `abort_multipart_on_failure` context manager is implemented in Task 3.3 and reused by both `s3_sync.py` and `package_model.py`
- The design uses Python throughout; no language selection is required

## Task Dependency Graph

```json
{
  "waves": [
    {
      "id": 0,
      "tasks": ["1.1", "1.2", "1.3", "15.2"]
    },
    {
      "id": 1,
      "tasks": ["2.1", "2.2", "5.1", "6.1", "9.4"]
    },
    {
      "id": 2,
      "tasks": ["2.3", "3.1", "5.2", "6.2", "7.1", "9.1", "11.1"]
    },
    {
      "id": 3,
      "tasks": ["2.4", "3.2", "3.3", "5.3", "7.2", "9.2", "11.2", "12.1"]
    },
    {
      "id": 4,
      "tasks": ["3.4", "7.3", "7.4", "9.3", "11.3", "12.2", "13.1"]
    },
    {
      "id": 5,
      "tasks": ["13.2", "13.3", "13.4", "15.1"]
    },
    {
      "id": 6,
      "tasks": ["13.5", "12.3"]
    },
    {
      "id": 7,
      "tasks": ["1.4", "2.5", "2.6", "2.7", "2.8", "2.9", "3.5", "3.6", "3.7", "3.8"]
    },
    {
      "id": 8,
      "tasks": ["5.4", "5.5", "5.6", "7.5", "7.6", "7.7", "7.8", "7.9", "7.10"]
    },
    {
      "id": 9,
      "tasks": ["9.5", "9.6", "9.7", "9.8", "9.9", "9.10", "9.11"]
    },
    {
      "id": 10,
      "tasks": ["11.4", "11.5", "11.6", "12.4", "13.6", "13.7"]
    }
  ]
}
```
