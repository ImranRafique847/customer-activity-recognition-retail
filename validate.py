"""
validate.py
===========
Run validation on the test set with a trained model.

Usage:
    python validate.py --weights runs/train/retail_cv/weights/best.pt
    python validate.py --weights runs/train/retail_cv/weights/best.pt --split test
"""

import argparse
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Validate a trained YOLO model")
    parser.add_argument("--weights", required=True, help="Path to best.pt")
    parser.add_argument(
        "--split",
        default="val",
        choices=["val", "test", "train"],
        help="Which split to evaluate on (default: val)",
    )
    parser.add_argument("--imgsz", type=int, default=416)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--data", default="dataset.yaml")
    args = parser.parse_args()

    if not Path(args.weights).exists():
        print(f"Error: weights not found: {args.weights}")
        sys.exit(1)

    from ultralytics import YOLO

    model = YOLO(args.weights)
    metrics = model.val(
        data=args.data,
        split=args.split,
        imgsz=args.imgsz,
        batch=args.batch,
        device=0,
        plots=True,
        verbose=True,
    )

    print("\n" + "=" * 50)
    print(f"VALIDATION RESULTS  (split={args.split})")
    print("=" * 50)
    print(f"mAP@0.50:      {metrics.box.map50:.4f}")
    print(f"mAP@0.50:0.95: {metrics.box.map:.4f}")
    print(f"Precision:     {metrics.box.mp:.4f}")
    print(f"Recall:        {metrics.box.mr:.4f}")

    print("\nPer-class AP@0.50:")
    names = model.names
    for i, ap in enumerate(metrics.box.ap50):
        print(f"  {names[i]:<25} {ap:.4f}")


if __name__ == "__main__":
    main()
