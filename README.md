# Customer Activity Recognition in Retail (CARR) — First External Benchmark

> **First published benchmark of YOLO11 on the CARR dataset**, establishing baseline results for customer activity recognition in retail surveillance environments.

[![Python](https://img.shields.io/badge/Python-3.11-blue)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.7-orange)](https://pytorch.org)
[![Ultralytics](https://img.shields.io/badge/Ultralytics-YOLO11-green)](https://ultralytics.com)
[![AWS](https://img.shields.io/badge/AWS-SageMaker-yellow)](https://aws.amazon.com/sagemaker)
[![Dataset](https://img.shields.io/badge/Dataset-CARR-red)](https://doi.org/10.6084/m9.figshare.29470079)

---

## Problem Statement

Retail stores generate hours of CCTV footage daily but lack automated tools to analyze customer behavior at the shelf level. Manual review is expensive, slow, and inconsistent. This project builds an end-to-end computer vision pipeline that:

1. **Detects customers** in overhead retail CCTV footage (45–90° camera angle)
2. **Classifies their activity** into 6 behavioral categories in real time
3. **Deploys to AWS** for scalable batch inference on recorded store footage
4. **Generates business analytics** — shelf engagement rates, conversion rates, dwell time

The core challenge is the **45–90° tilted camera angle** which causes severe head-shoulder compression, self-occlusions, and non-standard bounding box aspect ratios — making standard object detection models perform poorly without adaptation.

---

## Dataset — CARR (Customer Activity Recognition in Retail)

**Source:** [Figshare DOI: 10.6084/m9.figshare.29470079](https://doi.org/10.6084/m9.figshare.29470079)

**Authors:** Januar Adi Putra, Nanik Suciati, Chastine Fatichah (Institut Teknologi Sepuluh Nopember & Universitas Jember, Indonesia)

**Published:** July 2025 | **License:** CC BY 4.0

| Property | Value |
|---|---|
| Total images | 263,723 labeled frames |
| Total size | 18.55 GB |
| Camera type | Single monocular CCTV, 45–90° tilt |
| Environment | Real retail store (Alif Store, Indonesia) |
| Annotation format | YOLO bounding boxes |

### Action Classes

| ID | Class | Description |
|---|---|---|
| 0 | No Interest | Customer present but not interacting with shelf |
| 1 | Picking And Returning | Customer picks up item then puts it back |
| 2 | Picking and Putting | Customer picks item and places it in basket |
| 3 | Touching | Customer touches/examines item without picking |
| 4 | Turning to Shelf | Customer turning body toward shelf |
| 5 | Viewing | Customer looking at shelf without touching |

### Class Distribution (Raw)

| Class | Images |
|---|---|
| Picking And Returning | 54,294 |
| Touching | 52,008 |
| Turning to Shelf | 50,557 |
| No Interest | 37,481 |
| Picking and Putting | 34,241 |
| Viewing | 35,142 |

**Training split used:** 5,000 images per class (30,000 total) — balanced to prevent class bias.

---

## Results — First External Benchmark

### Baseline YOLO11s

| Class | Precision | Recall | mAP@0.5 | mAP@0.5:0.95 |
|---|---|---|---|---|
| No Interest | 0.959 | 0.925 | **0.982** | 0.880 |
| Picking And Returning | 0.896 | 0.935 | **0.969** | 0.861 |
| Picking and Putting | 0.926 | 0.945 | **0.979** | 0.868 |
| Touching | 0.943 | 0.951 | **0.986** | 0.866 |
| Turning to Shelf | 0.926 | 0.925 | **0.961** | 0.816 |
| Viewing | 0.987 | 0.990 | **0.994** | 0.888 |
| **Overall** | **0.939** | **0.945** | **0.979** | **0.863** |

### YOLO11s + CBAM (Attention Module)

| Class | Precision | Recall | mAP@0.5 | mAP@0.5:0.95 |
|---|---|---|---|---|
| No Interest | 0.953 | 0.899 | 0.974 | 0.801 |
| Picking And Returning | 0.863 | 0.941 | 0.949 | 0.772 |
| Picking and Putting | 0.938 | 0.961 | 0.982 | 0.801 |
| Touching | 0.951 | 0.946 | 0.986 | 0.808 |
| Turning to Shelf | 0.914 | 0.921 | 0.963 | 0.765 |
| Viewing | 0.982 | 0.991 | 0.994 | 0.827 |
| **Overall** | **0.933** | **0.943** | **0.975** | **0.795** |

### Comparison Summary

| Model | Params | mAP@0.5 | mAP@0.5:0.95 | Inference | Train Time |
|---|---|---|---|---|---|
| GSW-Yolo (dataset authors) | ~4.8M | ~0.90* | — | — | — |
| **YOLO11s Baseline (ours)** | **9.43M** | **0.979** | **0.863** | **3.3ms** | **15.7 hrs** |
| YOLO11s + CBAM (ours) | 9.48M | 0.975 | 0.795 | 3.5ms | 17.1 hrs |

> *GSW-Yolo reported on a different evaluation protocol. Our YOLO11s baseline surpasses their reported accuracy.*

**Key finding:** YOLO11s baseline outperforms YOLO11s+CBAM by 0.4% mAP@0.5, suggesting the standard FPN architecture is already sufficient for overhead retail surveillance without additional attention mechanisms.

---

## Architecture

### CBAM (Convolutional Block Attention Module)

CBAM was added after each neck C3k2 block in the YOLO11s FPN:

```
Backbone → SPPF → C2PSA
              ↓
         FPN Neck:
    C3k2 → CBAM (256ch)   ← P4 features
    C3k2 → CBAM (128ch)   ← P3 small objects
    C3k2 → CBAM (256ch)   ← P4 medium
    C3k2 → CBAM (512ch)   ← P5 large
              ↓
         Detection Head [P3, P4, P5]
```

CBAM adds only **22,954 parameters** (0.24% overhead) while adding channel + spatial attention.

---

## Project Structure

```
customer-activity-recognition-retail/
│
├── train_carr.py              # Main training script — YOLO11s & CBAM
├── balance_dataset.py         # Balance dataset to N images per class
├── prepare_dataset.py         # Download from S3 and prepare YOLO split
├── infer_video.py             # Run inference on videos with colored overlays
├── visualize_results.py       # Generate paper-ready charts and comparisons
├── validate.py                # Evaluate trained model on test set
├── train.py                   # Simple training wrapper
│
├── models/
│   ├── cbam.py                # CBAM attention module implementation
│   ├── register.py            # Register custom modules with ultralytics
│   └── __init__.py
│
├── configs/
│   ├── yolo11s-cbam.yaml      # YOLO11s + CBAM architecture definition
│   ├── carr_train.yaml        # Training hyperparameters
│   └── carr_dataset.yaml      # Dataset configuration
│
├── code/
│   ├── inference.py           # SageMaker inference handler (model_fn, input_fn, etc.)
│   └── requirements.txt       # Dependencies for SageMaker container
│
├── scripts/
│   ├── s3_sync.py             # Download/upload data and models to S3
│   ├── iam_validator.py       # Validate AWS IAM permissions
│   ├── package_model.py       # Package best.pt into model.tar.gz for SageMaker
│   ├── run_batch_transform.py # Launch SageMaker batch transform job
│   ├── deploy_realtime.py     # Deploy SageMaker real-time endpoint
│   ├── setup_cloudwatch.py    # Create CloudWatch alarms and dashboards
│   ├── launch_training.py     # Launch EC2 g4dn.xlarge for training
│   └── common/
│       ├── env_loader.py      # Load .env and validate credentials
│       └── retry.py           # Exponential backoff retry utilities
│
├── tests/
│   ├── test_s3_sync.py        # S3 sync unit tests
│   ├── test_iam_validator.py  # IAM validator unit tests
│   ├── test_package_model.py  # Model packager unit tests
│   ├── test_batch_transform.py# Batch transform unit tests
│   ├── test_cloudwatch_monitor.py # CloudWatch monitor unit tests
│   ├── test_inference.py      # Inference handler unit tests
│   └── conftest.py            # Pytest configuration
│
├── docs/
│   ├── deployment_architecture.md  # SageMaker vs Lambda architecture comparison
│   └── handoff_contract.md         # Local training → AWS deployment contract
│
└── .kiro/specs/aws-model-deployment/
    ├── requirements.md        # Full functional requirements (9 requirement groups)
    ├── design.md              # System design and architecture decisions
    └── tasks.md               # Implementation task breakdown
```

---

## What Each File Does

### `train_carr.py`
Main training script. Supports:
- `--model yolo11s` — train standard YOLO11s baseline
- `--model yolo11s-cbam` — train YOLO11s with CBAM attention
- `--compare` — run both automatically and print comparison table
- `--quick` — 5-epoch smoke test on 1% of data
- `--ablation` — train all model sizes (n/s/m/l)

### `balance_dataset.py`
Reads from local `data/s3_cache/` and creates balanced train/val/test path lists. Uses text file manifests instead of copying files — instant execution, zero extra disk space.

### `prepare_dataset.py`
Downloads labeled images from S3 bucket `retaildata-cv-2026` with:
- ETag-based skip (resume downloads)
- `--max-per-class` limit
- Automatic 70/20/10 train/val/test split

### `infer_video.py`
Runs inference on local or S3 videos with:
- Colored transparent overlay per action class
- Full class name labels (not abbreviated)
- On-screen legend showing all 6 action colors
- Support for single file, folder, or S3 download

### `visualize_results.py`
Generates publication-ready charts:
- Training curves (loss + mAP over epochs)
- Per-class mAP bar chart (baseline vs CBAM)
- Results summary table
- Confusion matrices side by side
- LinkedIn-ready dark theme visualization

### `models/cbam.py`
CBAM implementation — Channel Attention + Spatial Attention modules. Drop-in addition to YOLO11s neck. Adds 22,954 parameters (0.24% overhead).

### `code/inference.py`
SageMaker PyTorch inference handler implementing:
- `model_fn` — loads YOLO weights, validates class count
- `input_fn` — decodes JPEG/PNG, normalizes to float32
- `predict_fn` — runs inference, measures latency
- `output_fn` — returns JSON with detections, confidence, bbox

### `scripts/s3_sync.py`
Full-featured S3 sync with ETag checking, retry with exponential backoff, download manifest, upload with overwrite protection and size verification.

### `scripts/package_model.py`
Packages `best.pt` + `code/inference.py` + `requirements.txt` into `model.tar.gz` for SageMaker deployment. Validates `training_summary.json` fields before packaging.

### `scripts/run_batch_transform.py`
Launches SageMaker batch transform job with required tags, polls status every 60 seconds, handles terminal states with proper error reporting.

### `scripts/setup_cloudwatch.py`
Creates CloudWatch log groups, latency alarm (p99 > 4000ms), 5xx error alarm (>5 per 5 min), and dashboard with 4 widgets. Fully idempotent.

---

## Setup

### Prerequisites
- Python 3.11+
- NVIDIA GPU (optional for local training)
- AWS account with `AmazonS3FullAccess` + `AmazonSageMakerFullAccess`

### Installation

```bash
git clone https://github.com/ImranRafique847/customer-activity-recognition-retail.git
cd customer-activity-recognition-retail
pip install ultralytics boto3 python-dotenv matplotlib pandas pillow
```

### Configuration

Create a `.env` file (never committed):
```
AWS_ACCESS_KEY_ID=your_key
AWS_SECRET_ACCESS_KEY=your_secret
AWS_DEFAULT_REGION=us-east-1
S3_BUCKET=your-bucket-name
SAGEMAKER_EXECUTION_ROLE_ARN=arn:aws:iam::123456789:role/SageMakerRole
```

---

## Usage

### Download Dataset
```bash
# Download all classes (263K images, ~18GB)
python prepare_dataset.py

# Download subset for quick test
python prepare_dataset.py --max-per-class 1000
```

### Balance and Train
```bash
# Balance dataset
python balance_dataset.py --max-per-class 5000

# Train baseline
python train_carr.py --model yolo11s --epochs 200

# Train CBAM
python train_carr.py --model yolo11s-cbam --epochs 200

# Run full comparison
python train_carr.py --compare --epochs 200
```

### Run Inference on Video
```bash
# Local video with colored overlays
python infer_video.py --weights runs/carr/baseline_5k/weights/best.pt \
    --source my_video.mp4 --conf 0.25 --show

# All action classes from S3
python infer_video.py --weights runs/carr/baseline_5k/weights/best.pt \
    --s3-all --conf 0.25
```

### Generate Visualizations
```bash
python visualize_results.py
# Outputs to runs/carr/visualizations/
```

### Run Tests
```bash
python -m pytest tests/ -v
# 213 tests, all passing
```

---

## AWS Deployment

> **Note:** SageMaker deployment is planned as future work. The deployment scripts are fully implemented and tested but not yet executed in production. The current project focuses on the research benchmark.

The following scripts are ready for deployment when needed:

- `scripts/package_model.py` — packages `best.pt` into `model.tar.gz` for SageMaker
- `scripts/run_batch_transform.py` — launches SageMaker batch transform job
- `scripts/deploy_realtime.py` — deploys SageMaker real-time endpoint
- `scripts/setup_cloudwatch.py` — sets up CloudWatch monitoring
- `scripts/iam_validator.py` — validates IAM permissions before deployment

### Training Infrastructure Used

| Service | Role |
|---|---|
| **Amazon S3** | Dataset storage (263K images), model artifacts, training results |
| **Amazon EC2** (g4dn.xlarge) | Model training — Tesla T4 GPU, 16GB VRAM |
| **Amazon IAM** | Access control for S3 and EC2 |
| **SageMaker** | Planned — batch inference on store footage *(future work)* |

---

## Citation

If you use this benchmark in your research, please cite both the CARR dataset and this work:

**CARR Dataset:**
```bibtex
@misc{putra2025carr,
  author    = {Putra, Januar Adi and Suciati, Nanik and Fatichah, Chastine},
  title     = {Customer Activity Recognition in Retail (CARR) Dataset},
  year      = {2025},
  publisher = {Figshare},
  doi       = {10.6084/m9.figshare.29470079}
}
```

**GSW-Yolo (Dataset Authors' Model):**
```bibtex
@inproceedings{putra2025gswyolo,
  author    = {Putra, Januar Adi and Suciati, Nanik and Fatichah, Chastine},
  title     = {GSW-Yolo: Improved Light-Weight Person Detection Method Based on YOLOv8},
  booktitle = {Proc. 2025 IEEE ICAIIC},
  pages     = {97--102},
  year      = {2025}
}
```

---

## Related Work

| Paper | Venue | Relevance |
|---|---|---|
| RetailAction Dataset | ICCV 2025 Workshop | Multi-view retail action dataset |
| GSW-Yolo | IEEE ICAIIC 2025 | Lightweight detection for overhead cameras |
| CBAM | ECCV 2018 | Attention mechanism used in this work |
| YOLO11 | Ultralytics 2024 | Base detection architecture |

---

## License

Code: MIT License
Dataset: CC BY 4.0 (cite original authors)

---

*Established first external benchmark on the CARR dataset — as of August 2026, zero prior publications cite this dataset.*
