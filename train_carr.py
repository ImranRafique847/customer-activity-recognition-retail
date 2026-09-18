"""
train_carr.py
=============
Optimized training script for the CARR dataset.

Usage:
    py -3.11 train_carr.py                          # default: yolo11s, 200 epochs
    py -3.11 train_carr.py --model yolo11s-cbam     # YOLO11s + CBAM attention
    py -3.11 train_carr.py --model yolo11m          # larger model
    py -3.11 train_carr.py --quick                  # quick smoke test (~1000 imgs)
    py -3.11 train_carr.py --ablation               # run all model sizes
    py -3.11 train_carr.py --compare                # baseline vs CBAM comparison
"""

import argparse
import json
import time
from pathlib import Path

from ultralytics import YOLO

# Register custom modules (CBAM) before any YOLO usage
from models.register import register_custom_modules
register_custom_modules()


# ── Config ────────────────────────────────────────────────────────────────────

DATASET_YAML = "data/retail_cv_balanced/dataset.yaml"
PROJECT_DIR  = "runs/carr"

MODELS = {
    "yolo11n":      {"weights": "yolo11n.pt",                     "batch": 8,  "params": "2.5M"},
    "yolo11s":      {"weights": "yolo11s.pt",                     "batch": 16, "params": "9.4M"},
    "yolo11m":      {"weights": "yolo11m.pt",                     "batch": 8,  "params": "20M"},
    "yolo11l":      {"weights": "yolo11l.pt",                     "batch": 4,  "params": "25M"},
    "yolo11s-cbam": {"weights": "configs/yolo11s-cbam.yaml",      "batch": 16, "params": "~10M"},
    "yolo11m-cbam": {"weights": "configs/yolo11m-cbam.yaml",      "batch": 8,  "params": "~21M"},
}

TRAIN_ARGS = {
    "imgsz":           416,    # Reduced from 640 — fits in 4GB VRAM
    "optimizer":       "AdamW",
    "lr0":             0.001,
    "lrf":             0.01,
    "weight_decay":    0.0005,
    "warmup_epochs":   5,
    "hsv_h":           0.015,
    "hsv_s":           0.7,
    "hsv_v":           0.4,
    "degrees":         10.0,
    "translate":       0.1,
    "scale":           0.5,
    "shear":           2.0,
    "perspective":     0.0005,
    "flipud":          0.3,
    "fliplr":          0.5,
    "mosaic":          1.0,
    "mixup":           0.15,
    "copy_paste":      0.1,
    "close_mosaic":    20,
    "amp":             False,   # Disabled — Quadro T1000 doesn't support AMP
    "plots":           True,
    "verbose":         True,
}


# ── Training ──────────────────────────────────────────────────────────────────

def train_single(model_name: str, epochs: int, name: str | None = None,
                 extra_args: dict | None = None) -> dict:
    """Train a single model and return metrics."""
    cfg     = MODELS[model_name]
    weights = cfg["weights"]
    batch   = cfg["batch"]
    run_name = name or f"{model_name}_{epochs}ep"

    print(f"\n{'='*60}")
    print(f" Training: {model_name}  |  epochs: {epochs}  |  batch: {batch}")
    print(f" Weights/config: {weights}")
    print(f"{'='*60}\n")

    model = YOLO(weights)
    start = time.time()

    kwargs = {**TRAIN_ARGS, **(extra_args or {})}

    results = model.train(
        data=DATASET_YAML,
        epochs=epochs,
        batch=batch,
        patience=50,
        project=PROJECT_DIR,
        name=run_name,
        workers=4,
        device=0,
        **kwargs,
    )

    duration = time.time() - start

    metrics = {
        "model":        model_name,
        "weights":      weights,
        "params":       cfg["params"],
        "epochs":       epochs,
        "duration_hrs": round(duration / 3600, 2),
        "mAP50":        round(float(results.results_dict.get("metrics/mAP50(B)", 0)), 4),
        "mAP50_95":     round(float(results.results_dict.get("metrics/mAP50-95(B)", 0)), 4),
        "precision":    round(float(results.results_dict.get("metrics/precision(B)", 0)), 4),
        "recall":       round(float(results.results_dict.get("metrics/recall(B)", 0)), 4),
        "best_weights": str(Path(PROJECT_DIR) / run_name / "weights" / "best.pt"),
    }

    out = Path(PROJECT_DIR) / run_name / "carr_metrics.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(metrics, indent=2))

    print(f"\n{'='*60}")
    print(f" Results for {model_name}:")
    print(f"   mAP@0.5:      {metrics['mAP50']}")
    print(f"   mAP@0.5:0.95: {metrics['mAP50_95']}")
    print(f"   Precision:    {metrics['precision']}")
    print(f"   Recall:       {metrics['recall']}")
    print(f"   Duration:     {metrics['duration_hrs']} hours")
    print(f"{'='*60}\n")

    return metrics


def run_ablation(epochs: int = 100) -> None:
    """Train all model sizes and compare."""
    print("\nRunning ablation study across all YOLO11 variants...\n")
    all_metrics = []
    for model_name in ["yolo11n", "yolo11s", "yolo11m", "yolo11s-cbam"]:
        m = train_single(model_name, epochs=epochs, name=f"ablation_{model_name}")
        all_metrics.append(m)
    _print_comparison(all_metrics, "ABLATION RESULTS")
    _save_comparison(all_metrics, "ablation_comparison.json")


def run_compare(epochs: int = 100) -> None:
    """
    Run the key research comparison:
      YOLO11s baseline  vs  YOLO11s + CBAM
    This is the core contribution table for the paper.
    """
    print("\nRunning baseline vs CBAM comparison...\n")
    results = []
    for model_name in ["yolo11s", "yolo11s-cbam"]:
        m = train_single(model_name, epochs=epochs, name=f"compare_{model_name}")
        results.append(m)

    _print_comparison(results, "BASELINE vs CBAM")
    _save_comparison(results, "cbam_comparison.json")

    # Print improvement
    if len(results) == 2:
        base, cbam = results
        delta = cbam["mAP50"] - base["mAP50"]
        print(f"\n CBAM improvement: {delta:+.4f} mAP@0.5")
        print(f" {'POSITIVE' if delta > 0 else 'NEGATIVE'} result")


def _print_comparison(metrics: list, title: str) -> None:
    print("\n" + "="*85)
    print(f" {title}")
    print("="*85)
    print(f"{'Model':<16} {'Params':<8} {'mAP@0.5':<10} {'mAP@0.5:0.95':<14} {'Precision':<12} {'Recall':<8} {'Hrs'}")
    print("-"*85)
    for m in metrics:
        print(f"{m['model']:<16} {m['params']:<8} {m['mAP50']:<10} {m['mAP50_95']:<14} "
              f"{m['precision']:<12} {m['recall']:<8} {m['duration_hrs']}")
    print("="*85)


def _save_comparison(metrics: list, filename: str) -> None:
    out = Path(PROJECT_DIR) / filename
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(metrics, indent=2))
    print(f"Comparison saved to: {out}")


# ── Entry Point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Train YOLO11 on CARR dataset")
    parser.add_argument("--model",    default="yolo11s",
                        choices=list(MODELS.keys()),
                        help="Model variant (default: yolo11s)")
    parser.add_argument("--epochs",   type=int, default=200,
                        help="Number of training epochs (default: 200)")
    parser.add_argument("--name",     default=None,
                        help="Custom run name")
    parser.add_argument("--quick",    action="store_true",
                        help="Quick smoke test: 5 epochs on ~1000 images")
    parser.add_argument("--ablation", action="store_true",
                        help="Train all model sizes for comparison")
    parser.add_argument("--compare",  action="store_true",
                        help="Run baseline vs CBAM comparison (paper table)")
    args = parser.parse_args()

    if args.quick:
        train_single("yolo11n", epochs=5, name="smoke_test",
                     extra_args={"fraction": 0.007})
    elif args.ablation:
        run_ablation(epochs=args.epochs)
    elif args.compare:
        run_compare(epochs=args.epochs)
    else:
        train_single(args.model, epochs=args.epochs, name=args.name)


if __name__ == "__main__":
    main()
    args = parser.parse_args()

    if args.quick:
        train_single("yolo11n", epochs=5, name="smoke_test",
                     extra_args={"fraction": 0.007})  # ~1000 images only
    elif args.ablation:
        run_ablation(epochs=args.epochs)
    else:
        train_single(args.model, epochs=args.epochs, name=args.name)


if __name__ == "__main__":
    main()
