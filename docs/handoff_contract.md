# Handoff Contract: Local Training → AWS Deployment

## Overview

This document defines the contract between the local YOLO training pipeline and the
AWS deployment scripts. Following this checklist ensures a trained model can be
deployed to SageMaker without ambiguity about file layout, versions, or invocation
parameters.

---

## Required Training Artifacts

Before running any deployment script, the local training pipeline **must** produce
these two files in the same directory:

### `best.pt`
- Format: Ultralytics YOLO serialization (`.pt`)
- Must be a non-empty, valid YOLO model weights file
- Minimum size enforced by `package_model.py`: ≥ 1 KB

### `training_summary.json`
All six fields are required. Missing or incorrectly typed fields cause
`package_model.py` to exit with a descriptive error before creating any archive.

| Field | Type | Constraint |
|---|---|---|
| `model_architecture` | `str` | Non-empty (e.g. `"yolo11n"`) |
| `input_size` | `int` | Positive integer (e.g. `416`) |
| `num_classes` | `int` | Greater than 0 |
| `class_names` | `list[str]` | Length must equal `num_classes` |
| `dataset_version` | `str` | Non-empty (e.g. `"v3.2.0"`) |
| `training_device` | `str` | Non-empty (e.g. `"cuda:0"` or `"cpu"`) |

**Example:**
```json
{
  "model_architecture": "yolo11n",
  "input_size": 416,
  "num_classes": 5,
  "class_names": ["Picking", "Returning", "Walking", "Standing", "Unknown"],
  "dataset_version": "v3.2.0",
  "training_device": "cuda:0"
}
```

---

## Required Environment Variables (`.env`)

```
AWS_ACCESS_KEY_ID=<your key>
AWS_SECRET_ACCESS_KEY=<your secret>
AWS_DEFAULT_REGION=us-east-1
SAGEMAKER_EXECUTION_ROLE_ARN=arn:aws:iam::<account>:role/SageMakerExecutionRole

# Optional — CloudWatch alarm SNS notifications
CLOUDWATCH_ALARM_SNS_ARN=arn:aws:sns:us-east-1:<account>:retail-cv-alerts
```

---

## Step-by-Step Deployment Checklist

Work through these steps in order. Each step must succeed before proceeding.

- [ ] **Step 1 — Verify training artifacts**
  Confirm both files exist in the same directory:
  ```
  models/<run_name>/best.pt
  models/<run_name>/training_summary.json
  ```

- [ ] **Step 2 — Validate IAM permissions**
  ```bash
  python scripts/iam_validator.py
  ```
  Expected output: `IAM validation passed`
  If permissions are missing, the script prints a JSON list of policy statements
  to attach to the `Computer-Vision` IAM user.

- [ ] **Step 3 — Upload model weights to S3**
  ```bash
  python -m scripts.s3_sync \
      --upload-model models/<run_name>/best.pt \
      --run-name <run_name>
  ```
  On success, prints the S3 URI:
  `s3://retaildata-cv-2026/models/<run_name>/best.pt`

- [ ] **Step 4 — Package model artifact**
  ```bash
  python -m scripts.package_model \
      --model-path models/<run_name>/best.pt \
      --run-name <run_name>
  ```
  Bundles `best.pt`, `code/inference.py`, `code/requirements.txt`, and
  `code/model_config.json` into `model.tar.gz` and uploads it to:
  `s3://retaildata-cv-2026/models/<run_name>/model.tar.gz`

- [ ] **Step 5a — Run batch transform (primary)**
  ```bash
  python -m scripts.run_batch_transform \
      --input-s3-uri  s3://retaildata-cv-2026/batch-input/<run_name>/ \
      --output-s3-uri s3://retaildata-cv-2026/batch-output/<run_name>/ \
      --model-name    retail-cv-yolo-<run_name>
  ```
  Polls every 60 seconds. Prints output S3 URI and duration on completion.

- [ ] **Step 5b — Deploy real-time endpoint (optional)**
  Only needed for on-demand single-frame queries (e.g. store dashboards):
  ```bash
  python -m scripts.deploy_realtime \
      --model-name    retail-cv-yolo-<run_name> \
      --endpoint-name retail-cv-inference
  ```
  > ⚠️ **COST WARNING:** Delete the endpoint after 30 minutes of idle time:
  > ```bash
  > aws sagemaker delete-endpoint --endpoint-name retail-cv-inference
  > ```

- [ ] **Step 6 — Set up CloudWatch monitoring**
  ```bash
  python -m scripts.setup_cloudwatch \
      --endpoint-name retail-cv-inference \
      --deployment-type realtime   # or: batch
  ```
  Creates log group (30-day retention), latency alarm (p99 > 4 000 ms),
  5XX alarm (sum > 5), and the `retail-cv-inference-dashboard`.

---

## Notes

- All scripts read credentials exclusively from `.env` via `python-dotenv`.
  The `.env` file must be present at the project root.
- Run `python scripts/iam_validator.py` whenever credentials or IAM policies change.
- The real-time endpoint auto-scales between 1 and 4 `ml.g4dn.xlarge` instances.
  Scale-out triggers at > 10 invocations/min; scale-in at < 5 invocations/min
  for 5 consecutive minutes.
