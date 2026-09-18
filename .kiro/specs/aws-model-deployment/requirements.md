# Requirements Document

## Introduction

This feature covers the AWS infrastructure layer for a retail computer vision pipeline.
A YOLO-based model is trained locally on a GPU laptop and then deployed to AWS for
inference. The scope includes:

1. **S3 data sync** — boto3/CLI scripts to download labeled training subsets from S3
   to local storage and upload trained model weights back to S3.
2. **IAM/auth verification** — validating and correcting the `Computer-Vision` IAM
   user permissions so every downstream AWS service can be reached.
3. **Inference deployment** — selecting and implementing the right AWS deployment
   architecture (SageMaker real-time endpoint, SageMaker batch transform, or
   Lambda + API Gateway) for the trained `best.pt` weights.
4. **CloudWatch monitoring** — structured logging, metric alarms, and dashboards for
   the live inference endpoint.
5. **Local-to-AWS handoff** — the contract between the local training pipeline and the
   deployed endpoint (model artifact packaging, environment specification, invocation
   protocol).

Out of scope: the YOLO training loop, epoch monitoring, and any cell-by-cell Jupyter
notebook work.

---

## Glossary

- **S3_Sync_Tool**: The boto3-based Python scripts (and optional AWS CLI wrappers)
  responsible for transferring data between S3 and local storage.
- **IAM_Validator**: The script or checklist that verifies and corrects the
  `Computer-Vision` IAM user's attached policies and inline policies.
- **Deployment_Selector**: The decision logic (documented as code comments and a
  README section) that justifies the chosen AWS inference architecture.
- **Inference_Endpoint**: The live AWS resource that accepts an image or video frame
  and returns detection/tracking/action-recognition results (JSON).
- **Model_Packager**: The script that bundles `best.pt`, `requirements.txt`, and the
  SageMaker inference handler into a `model.tar.gz` artifact for S3 upload.
- **CloudWatch_Monitor**: The set of CloudWatch log groups, metric filters, alarms,
  and dashboards that observe the Inference_Endpoint at runtime.
- **Handoff_Contract**: The agreed-upon input/output schema and artifact layout that
  the local training pipeline must satisfy before invoking deployment scripts.
- **Computer-Vision**: The AWS IAM user whose credentials are used by all scripts in
  this project.
- **retaildata-cv-2026**: The S3 bucket storing raw footage, labeled images, and
  trained model artifacts.

---

## Requirements

### Requirement 1: S3 Data Download for Training

**User Story:** As a computer vision engineer, I want to download subsets of labeled
images and raw CCTV footage from S3 to my local machine, so that I can train YOLO
models without downloading the full 21 GB dataset every time.

#### Acceptance Criteria

1. THE S3_Sync_Tool SHALL accept a `--subset` parameter that filters the download to a
   specified activity category subfolder under
   `extracted_images_and_labels/data/` (e.g., `Picking And Returning`).
2. THE S3_Sync_Tool SHALL accept a `--max-files` integer parameter (positive integer
   greater than 0) that limits the number of image/label pairs downloaded per
   invocation; IF `--max-files` is provided with a value less than or equal to 0,
   THEN THE S3_Sync_Tool SHALL print an error message stating the valid range and
   exit with a non-zero return code.
3. WHEN the `--subset` parameter is omitted, THE S3_Sync_Tool SHALL download all
   objects under `extracted_images_and_labels/data/`; WHEN `--max-files` is also
   omitted, THE S3_Sync_Tool SHALL download all objects with no file count limit.
4. WHEN a downloaded file already exists locally and its S3 ETag matches the local
   MD5 checksum, THE S3_Sync_Tool SHALL skip re-downloading that file; WHEN the
   S3 ETag does not match the local MD5 checksum, THE S3_Sync_Tool SHALL
   re-download and overwrite the local file.
5. WHEN a network error or S3 throttling response is received during a download, THE
   S3_Sync_Tool SHALL retry the failed transfer up to 3 times with exponential
   backoff starting at a 1-second base interval; IF all 3 retries are exhausted,
   THEN THE S3_Sync_Tool SHALL print the error for the failed file and exit with a
   non-zero return code.
6. IF a requested `--subset` path does not exist in the S3 bucket, THEN THE
   S3_Sync_Tool SHALL print an error message that includes the missing S3 path and
   exit with a non-zero return code.
7. THE S3_Sync_Tool SHALL write a download manifest file (`download_manifest.json`)
   to the local destination directory listing only the files newly transferred in
   that invocation (not skipped files), each with its S3 key and byte size, plus a
   `total_files_downloaded` field and a `total_files_skipped` field; WHEN zero
   files are downloaded, THE S3_Sync_Tool SHALL still write a
   `download_manifest.json` with an empty file list, `total_files_downloaded` set
   to 0, and `total_files_skipped` set to the count of files skipped due to ETag
   match.
8. WHEN zero files are transferred because all local files match their S3 ETags, THE
   S3_Sync_Tool SHALL exit with return code 0 and write the manifest as specified
   in criterion 7 with `total_files_downloaded` equal to 0.
9. THE S3_Sync_Tool SHALL accept a `--source` parameter with valid values `images`
   (default) and `videos`; IF `--source` is provided with any other value, THEN THE
   S3_Sync_Tool SHALL print an error listing the valid values and exit with a
   non-zero return code; WHEN `--source` is omitted, THE S3_Sync_Tool SHALL default
   to `images`.
10. WHEN a video download is requested via `--source videos`, THE S3_Sync_Tool SHALL
    accept a `--store-ids` parameter accepting a comma-separated list of store folder
    names under `extracted_videos/` to restrict the download scope; IF a specified
    store folder does not exist in S3, THEN THE S3_Sync_Tool SHALL print an error
    identifying the missing store folder name and exit with a non-zero return code.

---

### Requirement 2: S3 Model Artifact Upload

**User Story:** As a computer vision engineer, I want to upload trained model weights
and associated metadata to S3, so that the deployment pipeline can retrieve the
correct artifact version without manual file management.

#### Acceptance Criteria

1. THE S3_Sync_Tool SHALL accept an `--upload-model` flag that takes a local path to
   a `.pt` file; IF the path does not exist on the local filesystem or the file does
   not have a `.pt` extension, THEN THE S3_Sync_Tool SHALL print an error
   identifying the invalid path and exit with a non-zero return code; WHEN the path
   is valid, THE S3_Sync_Tool SHALL upload the file to
   `s3://retaildata-cv-2026/models/{run_name}/best.pt`.
2. THE S3_Sync_Tool SHALL accept a `--run-name` parameter containing only
   alphanumeric characters, underscores, or hyphens, with a maximum length of 128
   characters; IF `--run-name` contains invalid characters or exceeds 128
   characters, THEN THE S3_Sync_Tool SHALL print an error describing the constraint
   and exit with a non-zero return code.
3. WHEN `--run-name` is omitted during upload, THE S3_Sync_Tool SHALL generate a
   run name using the format `run_{YYYYMMDDTHHmmssZ}` (compact UTC ISO 8601,
   no colons or spaces) and print it to stdout.
4. WHEN an object with the same key already exists in S3, THE S3_Sync_Tool SHALL
   require an explicit `--overwrite` flag before replacing it; without the flag, THE
   S3_Sync_Tool SHALL exit with an error and print the existing S3 key.
5. WHEN the upload completes, THE S3_Sync_Tool SHALL verify the upload by comparing
   the S3 object size to the local file size; IF the sizes do not match, THEN THE
   S3_Sync_Tool SHALL print an error identifying the size mismatch and exit with a
   non-zero return code; WHEN sizes match, THE S3_Sync_Tool SHALL print the full
   S3 URI of the uploaded artifact.
6. WHEN the upload of the model file completes successfully, THE S3_Sync_Tool SHALL
   upload a `run_metadata.json` file to
   `s3://retaildata-cv-2026/models/{run_name}/run_metadata.json` containing at
   minimum: `run_name`, `upload_timestamp` (UTC ISO 8601), `local_file_size_bytes`
   (integer), and `python_script_version` (the S3_Sync_Tool's own version string).
7. IF the upload fails for any reason, THEN THE S3_Sync_Tool SHALL not leave a
   partial object in S3 (including any incomplete multipart upload parts) and SHALL
   raise an error with the failure reason; a pre-existing object at the same S3 key
   SHALL remain intact and unmodified.
8. IF the upload of `run_metadata.json` fails after the model file has been
   successfully uploaded, THEN THE S3_Sync_Tool SHALL print an error identifying the
   metadata upload failure and exit with a non-zero return code; the already-uploaded
   model file SHALL remain in S3 and SHALL NOT be deleted.

---

### Requirement 3: IAM Permission Verification and Correction

**User Story:** As a DevOps engineer, I want to validate that the `Computer-Vision`
IAM user has exactly the permissions needed for all pipeline scripts, so that
deployments do not fail with access-denied errors at runtime.

#### Acceptance Criteria

1. WHEN THE IAM_Validator is invoked, THE IAM_Validator SHALL verify that the
   `Computer-Vision` user has an attached or inline policy granting `s3:GetObject`,
   `s3:PutObject`, `s3:DeleteObject`, and `s3:ListBucket` on
   `arn:aws:s3:::retaildata-cv-2026` and `arn:aws:s3:::retaildata-cv-2026/*`.
2. WHEN THE IAM_Validator is invoked, THE IAM_Validator SHALL verify that the
   `Computer-Vision` user has a policy granting exactly the following SageMaker
   actions on resource `*`: `sagemaker:CreateModel`,
   `sagemaker:CreateEndpointConfig`, `sagemaker:CreateEndpoint`,
   `sagemaker:InvokeEndpoint`, `sagemaker:DeleteEndpoint`, and
   `sagemaker:DescribeEndpoint`.
3. WHEN THE IAM_Validator is invoked, THE IAM_Validator SHALL verify that the
   `Computer-Vision` user has a policy granting `logs:CreateLogGroup`,
   `logs:CreateLogStream`, and `logs:PutLogEvents` on
   `arn:aws:logs:*:*:log-group:/aws/sagemaker/*`.
4. WHEN THE IAM_Validator is invoked, THE IAM_Validator SHALL verify that the
   `Computer-Vision` user has `iam:PassRole` on the SageMaker execution role ARN
   specified by the environment variable `SAGEMAKER_EXECUTION_ROLE_ARN`.
5. WHEN one or more required permissions are missing, THE IAM_Validator SHALL write
   to stdout a JSON object containing a `permissions_missing` array (each element
   being an AWS policy statement that can be attached directly via the AWS CLI), a
   `verification_result` field set to `PERMISSIONS_MISSING`, and exit with a
   non-zero return code.
6. WHEN all required permissions are present, THE IAM_Validator SHALL exit with
   return code 0 and print `IAM validation passed` to stdout.
7. IF the IAM_Validator cannot connect to AWS (invalid credentials or no network),
   THEN THE IAM_Validator SHALL set `verification_result` to `CONNECTION_FAILED`,
   print a descriptive error to stdout that identifies the credential source checked
   (e.g., the environment variable names), and exit with a non-zero return code.
8. THE IAM_Validator SHALL read credentials exclusively from the environment
   variables `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, and
   `AWS_DEFAULT_REGION` sourced from the project `.env` file; THE IAM_Validator
   SHALL NOT hard-code credential values.
9. IF any required environment variable (`AWS_ACCESS_KEY_ID`,
   `AWS_SECRET_ACCESS_KEY`, `AWS_DEFAULT_REGION`, or `SAGEMAKER_EXECUTION_ROLE_ARN`)
   is missing or empty when THE IAM_Validator is invoked, THEN THE IAM_Validator
   SHALL print an error identifying each missing variable by name and exit with a
   non-zero return code before making any AWS API calls.

---

### Requirement 4: Deployment Architecture Selection

**User Story:** As a machine learning engineer, I want a documented and justified
deployment architecture decision, so that the inference endpoint matches the actual
usage pattern (batch offline analysis of recorded footage, not sub-second live
queries).

#### Acceptance Criteria

1. THE Deployment_Selector SHALL document the evaluation of three candidate
   architectures — SageMaker real-time endpoint, SageMaker batch transform, and
   Lambda + API Gateway — in a structured comparison table that includes a verdict
   (one of: `Suitable`, `Unsuitable`, or `Conditional`) for each architecture
   against each of the following criteria: cost per inference, cold-start latency
   tolerance, input payload size, GPU availability, and operational complexity.
2. THE Deployment_Selector SHALL select **SageMaker batch transform** as the
   primary architecture, justified by: (a) inference results are consumed only after
   full job completion and not per-request, and (b) batch transform incurs no
   always-on endpoint costs between jobs, making it lower cost for scheduled offline
   workloads.
3. WHERE a stakeholder requires on-demand single-frame inference (e.g., a store
   dashboard querying one frame at a time), THE Deployment_Selector SHALL specify a
   SageMaker real-time endpoint as a secondary optional configuration, including at
   minimum: the instance type, the expected request payload format (single image
   frame), and a cold-start latency tolerance that SHALL NOT exceed 30 seconds.
4. THE Deployment_Selector documentation SHALL specify the minimum `ml.g4dn.xlarge`
   instance type for GPU-accelerated inference on YOLO nano models and justify this
   choice by comparing the inference time per frame in milliseconds on GPU versus
   CPU-only instance types, using values derived from either direct profiling
   measurements or AWS documentation and published benchmarks.
5. THE Deployment_Selector documentation SHALL include at minimum the following
   sections: architecture evaluation table, selected architecture with justification,
   instance type selection with justification, and optional real-time endpoint
   configuration.

---

### Requirement 5: Model Artifact Packaging

**User Story:** As a machine learning engineer, I want a packaging script that
bundles the trained model and its inference handler into a SageMaker-compatible
artifact, so that deployment is repeatable and does not rely on manual steps.

#### Acceptance Criteria

1. THE Model_Packager SHALL bundle `best.pt`, `code/inference.py` (the SageMaker
   inference handler), and `code/requirements.txt` into a single `model.tar.gz`
   archive following the SageMaker PyTorch model directory layout.
2. THE Model_Packager SHALL accept a `--model-path` argument pointing to the local
   `best.pt` file and a `--run-name` argument matching the S3 version directory.
3. WHEN archive integrity verification succeeds, THE Model_Packager SHALL upload
   `model.tar.gz` to `s3://retaildata-cv-2026/models/{run_name}/model.tar.gz`; IF
   the S3 upload fails, THEN THE Model_Packager SHALL raise a descriptive error
   indicating the failure reason and SHALL NOT leave a partial object at the
   destination key.
4. WHEN the `model.tar.gz` archive is created, THE Model_Packager SHALL verify
   integrity by extracting it to a temporary directory and confirming that `best.pt`
   is present and at least 1 KB in size and that `code/inference.py` is present and
   at least 1 byte in size; IF either file is absent or below its minimum size,
   THEN THE Model_Packager SHALL raise a descriptive error identifying the missing
   or undersized file and SHALL NOT proceed with the upload.
5. IF any required file (`best.pt`, `code/inference.py`, `code/requirements.txt`)
   is missing from the local filesystem during packaging, THEN THE Model_Packager
   SHALL raise a descriptive error identifying the missing file and SHALL NOT create
   or upload a partial archive.
6. THE Model_Packager's `code/inference.py` handler SHALL implement the
   `model_fn`, `input_fn`, `predict_fn`, and `output_fn` functions required by the
   SageMaker PyTorch inference container; `model_fn` SHALL load `best.pt` and
   return a callable model object ready for inference.
7. WHEN `input_fn` receives a request with `Content-Type: image/jpeg` or
   `image/png`, THE inference handler SHALL decode the image bytes into a float32
   tensor with pixel values normalized to the range [0.0, 1.0] and resize it to
   416×416 pixels before passing it to the model.
8. IF `input_fn` receives a request with a `Content-Type` other than `image/jpeg`
   or `image/png`, THEN THE inference handler SHALL return an error response
   indicating the unsupported content type without invoking the model.
9. WHEN `output_fn` is called, THE inference handler SHALL return a JSON-serialized
   response containing: `detections` (list of objects each with `label` (string),
   `confidence` (float in the range 0.0 to 1.0 inclusive), and `bbox` (list of
   four normalized floats [x_min, y_min, x_max, y_max] each in the range 0.0 to
   1.0 inclusive)), `inference_time_ms` (float greater than 0), and
   `model_run_name` (string matching the `--run-name` argument used during
   packaging).
10. THE Model_Packager SHALL validate that the `--run-name` argument contains only
    alphanumeric characters, underscores, or hyphens and does not exceed 128
    characters; IF the value is invalid, THEN THE Model_Packager SHALL print a
    descriptive error stating the constraint and exit with a non-zero return code
    before creating any archive.
11. IF `predict_fn` raises an unhandled exception, THEN `output_fn` SHALL catch the
    exception, log a structured JSON error entry containing `error_type`,
    `error_message`, and `timestamp`, and return an HTTP 500 response without
    propagating the exception.

---

### Requirement 6: SageMaker Batch Transform Job Execution

**User Story:** As a machine learning engineer, I want a script that launches a
SageMaker batch transform job against a set of input images stored in S3, so that I
can run inference over an entire store's footage without managing compute myself.

#### Acceptance Criteria

1. THE batch transform script SHALL accept an S3 input prefix containing JPEG or PNG
   images and write one JSON output file per input image to a specified S3 output
   prefix.
2. THE batch transform script SHALL accept `--input-s3-uri`, `--output-s3-uri`,
   `--model-name`, and `--instance-type` as parameters; IF any of `--input-s3-uri`,
   `--output-s3-uri`, or `--model-name` is missing, THEN THE batch transform script
   SHALL print an error identifying the missing parameter(s) and exit with a
   non-zero return code without submitting a job.
3. WHEN `--instance-type` is omitted, THE batch transform script SHALL default to
   `ml.g4dn.xlarge`.
4. THE batch transform script SHALL poll the SageMaker `DescribeTransformJob` API
   every 60 seconds and print the current job status and job name to stdout on each
   poll; THE batch transform script SHALL stop polling and exit with a non-zero
   return code if the total polling duration exceeds 24 hours.
5. WHEN the job reaches a terminal state (`Completed`, `Failed`, or `Stopped`), THE
   batch transform script SHALL stop polling and proceed to the terminal-state
   handling logic.
6. WHEN a batch transform job completes successfully, THE batch transform script
   SHALL print the S3 output URI and the total job duration in seconds, and exit
   with return code 0.
7. IF a batch transform job enters the `Failed` or `Stopped` state, THEN THE batch
   transform script SHALL retrieve the failure reason from the SageMaker API; IF the
   failure reason is absent or null in the API response, THEN THE batch transform
   script SHALL print `Failure reason not available` before exiting with a non-zero
   return code.
8. THE batch transform script SHALL tag each SageMaker batch transform job with the
   tags `Project=retail-cv-analytics` and `ManagedBy=aws-model-deployment-spec`.
9. IF the `DescribeTransformJob` API call raises an exception during polling, THEN
   THE batch transform script SHALL print the exception message, retry the poll up
   to 3 times with a 30-second interval, and exit with a non-zero return code if
   all retries fail.
10. IF the `CreateTransformJob` API call fails, THEN THE batch transform script SHALL
    print the error response from the SageMaker API and exit with a non-zero return
    code without entering the polling loop.

---

### Requirement 7: SageMaker Real-Time Endpoint (Optional)

**User Story:** As a store operations analyst, I want an on-demand HTTP endpoint
that accepts a single image and returns detection results within 5 seconds, so that
I can query individual frames from a store dashboard without running a full batch job.

#### Acceptance Criteria

1. WHERE the real-time endpoint option is enabled, THE Inference_Endpoint SHALL
   expose an HTTPS endpoint that accepts `POST` requests with a JPEG or PNG image
   body not exceeding 20 MB and 1920×1080 pixels, and returns a JSON detection
   response containing `detections` (list with `label`, `confidence` 0.0–1.0,
   and `bbox` as four normalized floats), within 5 seconds; IF the request body is
   not a valid JPEG or PNG, exceeds 20 MB, or exceeds 1920×1080 pixels, THEN THE
   Inference_Endpoint SHALL return an HTTP 400 response with an error message
   identifying the rejection reason.
2. WHERE the real-time endpoint option is enabled, THE Inference_Endpoint SHALL use
   a minimum of one `ml.g4dn.xlarge` instance and support auto-scaling: scale out
   when `InvocationsPerInstance` exceeds 10 requests per minute, and scale in when
   `InvocationsPerInstance` falls below 5 requests per minute for 5 consecutive
   minutes, between 1 and 4 instances.
3. WHERE the real-time endpoint option is enabled, THE deployment script's README
   SHALL include a note stating that the endpoint SHALL be deleted after 30 minutes
   of idle time (zero invocations) to avoid unnecessary cost.
4. IF endpoint creation fails, THEN the deployment script SHALL retrieve the failure
   reason from `DescribeEndpoint` and print it to stdout before exiting with a
   non-zero return code.
5. WHEN endpoint creation succeeds, THE deployment script SHALL call
   `DescribeEndpoint` to confirm the endpoint status is `InService` and print a
   confirmation summary including the endpoint name, status, and HTTPS invocation
   URL to stdout.

---

### Requirement 8: CloudWatch Logging and Monitoring

**User Story:** As a DevOps engineer, I want structured logs and metric alarms for
the deployed inference endpoint, so that I can detect failures, performance
regressions, and cost overruns without manually checking the AWS console.

#### Acceptance Criteria

1. IF a real-time deployment is active, THE CloudWatch_Monitor SHALL create a
   CloudWatch log group named `/aws/sagemaker/Endpoints/retail-cv-inference` with a
   retention policy of 30 days; IF a batch deployment is active, THE
   CloudWatch_Monitor SHALL create a CloudWatch log group named
   `/aws/sagemaker/TransformJobs/retail-cv-batch` with a retention policy of 30
   days; WHEN a log group already exists, THE CloudWatch_Monitor SHALL update its
   retention policy to 30 days if it differs, without recreating the log group.
2. WHEN the inference handler raises an unhandled exception, THE inference handler
   SHALL log a structured JSON entry containing `level`, `error_type`,
   `error_message`, `input_shape`, and `timestamp` to the SageMaker container log
   stream before returning an HTTP 500 response; IF the log write itself fails, THE
   inference handler SHALL still return the HTTP 500 response without raising a
   secondary exception.
3. THE CloudWatch_Monitor SHALL create a metric alarm on the `ModelLatency` metric
   (real-time endpoint) that triggers when the p99 latency exceeds 4000 ms for 1
   out of 1 consecutive 5-minute evaluation periods, with `TreatMissingData` set to
   `notBreaching`.
4. THE CloudWatch_Monitor SHALL create a metric alarm on the `Invocation5XXErrors`
   metric using the `Sum` statistic that triggers when the count exceeds 5 for 1
   out of 1 consecutive 5-minute evaluation periods, with `TreatMissingData` set to
   `notBreaching`.
5. THE CloudWatch_Monitor SHALL create a CloudWatch dashboard named
   `retail-cv-inference-dashboard` containing widgets for: `ModelLatency` (p50,
   p90, p99), `Invocations`, `Invocation5XXErrors`, and `CPUUtilization`; WHERE
   `GPUUtilization` metrics are available for the endpoint instance type, THE
   CloudWatch_Monitor SHALL replace the `CPUUtilization` widget with a
   `GPUUtilization` widget.
6. THE CloudWatch_Monitor setup script SHALL be idempotent: running it multiple
   times SHALL update the threshold, period, evaluation periods, and alarm actions
   of existing alarms and update existing dashboards to match the declared
   configuration, without duplicating alarms or dashboards.
7. WHEN a CloudWatch alarm transitions to the `ALARM` state, THE CloudWatch_Monitor
   SHALL publish a notification to an SNS topic whose ARN is configurable via the
   environment variable `CLOUDWATCH_ALARM_SNS_ARN`; if the variable is unset, THE
   CloudWatch_Monitor SHALL create the alarm without an SNS action and print a
   warning to stdout.

---

### Requirement 9: Local-to-AWS Handoff Contract

**User Story:** As a machine learning engineer, I want a clearly defined contract
between the local training pipeline and the AWS deployment scripts, so that a
freshly trained model can be deployed without ambiguity about file layout, versions,
or invocation parameters.

#### Acceptance Criteria

1. THE Handoff_Contract SHALL specify that the local training pipeline MUST produce
   a `best.pt` file in the Ultralytics YOLO serialization format before the
   Model_Packager script is invoked.
2. THE Handoff_Contract SHALL specify that `best.pt` MUST be accompanied by a
   `training_summary.json` file; WHEN THE Model_Packager is invoked, THE
   Model_Packager SHALL verify that `best.pt` exists on the local filesystem and
   that `training_summary.json` contains at minimum: `model_architecture` (string),
   `input_size` (integer), `num_classes` (integer greater than 0), `class_names`
   (list of strings with length equal to `num_classes`), `dataset_version` (string),
   and `training_device` (string); IF `best.pt` is absent or any of these fields is
   absent or has an incorrect type, THEN THE Model_Packager SHALL raise a
   descriptive error identifying the invalid or missing field and SHALL NOT proceed
   with packaging.
3. WHEN THE Model_Packager is invoked, THE Model_Packager SHALL read
   `training_summary.json` and embed `num_classes`, `class_names`, and `input_size`
   into a `model_config.json` file placed inside `model.tar.gz`, so the inference
   handler can load the correct class map without hard-coded values.
4. WHEN the inference handler loads the model at startup, THE inference handler SHALL
   validate that the number of output classes in the loaded model matches
   `num_classes` from `model_config.json`; IF a mismatch is detected, THEN THE
   inference handler SHALL log a structured error entry containing
   `model_class_count`, `expected_num_classes`, and the path to `model_config.json`,
   and SHALL return an HTTP 500 response for all subsequent requests until the
   endpoint is redeployed with a consistent model and config.
5. THE Handoff_Contract documentation SHALL be written to `docs/handoff_contract.md`
   in the project repository and SHALL include a step-by-step checklist of actions
   the engineer must complete before running the deployment scripts.
6. THE S3_Sync_Tool, IAM_Validator, Model_Packager, batch transform script, and
   CloudWatch_Monitor setup script SHALL all read AWS credentials exclusively from
   the project `.env` file via `python-dotenv`; IF the `.env` file itself is absent,
   THEN each script SHALL print an error identifying the missing `.env` file and
   exit with a non-zero return code; IF the `.env` file exists but any required
   environment variable (`AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`,
   `AWS_DEFAULT_REGION`) is missing or empty, THEN each script SHALL raise a
   descriptive error identifying each missing variable by name and exit with a
   non-zero return code.
