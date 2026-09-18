"""
extract_frames.py
=================
Downloads sample videos from S3, extracts frames at a target FPS,
auto-labels them using the trained YOLO model, and adds them to the
training dataset.

Strategy: download 1 video per class (6 total, ~100 MB) to avoid
downloading all 10 GB. Extracts frames at 2 fps — a 17s video gives
~34 labelled frames per class.

Usage:
    # Extract frames from 1 video per class, add to dataset
    python extract_frames.py --weights runs/detect/runs/train/retail_cv_300/weights/best.pt

    # Use more videos per class
    python extract_frames.py --weights best.pt --videos-per-class 3

    # Custom FPS
    python extract_frames.py --weights best.pt --fps 1
"""

import argparse
import os
import sys
from pathlib import Path

import boto3
import cv2
from dotenv import load_dotenv

load_dotenv()

BUCKET = os.environ.get("S3_BUCKET", "retaildata-cv-2026")
VIDEO_PREFIX = "extracted_videos/Video/"

CLASS_NAMES = [
    "No Interest",
    "Picking And Returning",
    "Picking and Putting",
    "Touching",
    "Turning to Shelf",
    "Viewing",
]

# Map class names to their integer IDs (must match dataset.yaml)
CLASS_TO_ID = {name: i for i, name in enumerate(CLASS_NAMES)}

# Map video filename class names to dataset class names
VIDEO_CLASS_MAP = {
    "No Interest": "No Interest",
    "Picking and Returning": "Picking And Returning",
    "Picking and Putting": "Picking and Putting",
    "Touching": "Touching",
    "Turning to Shelf": "Turning to Shelf",
    "Viewing": "Viewing",
}


def list_videos_by_class(s3) -> dict[str, list[dict]]:
    """Return dict mapping class_name -> list of S3 object dicts."""
    paginator = s3.get_paginator("list_objects_v2")
    by_class: dict[str, list] = {cls: [] for cls in CLASS_NAMES}

    for page in paginator.paginate(Bucket=BUCKET, Prefix=VIDEO_PREFIX):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            name = key.split("/")[-1]
            if not name.endswith(".mp4"):
                continue
            # Parse class from filename: "Alif Store-No Interest-1.mp4"
            parts = name.split("-")
            if len(parts) >= 2:
                video_cls = parts[1].strip()
                mapped = VIDEO_CLASS_MAP.get(video_cls)
                if mapped and mapped in by_class:
                    by_class[mapped].append(obj)

    return by_class


def download_video(s3, obj: dict, local_dir: Path) -> Path:
    """Download a single video from S3, skip if already cached."""
    key = obj["Key"]
    name = key.split("/")[-1]
    local_path = local_dir / name
    if local_path.exists():
        print(f"    Cached: {name}")
        return local_path
    print(f"    Downloading: {name} ({round(obj['Size']/1e6, 1)} MB)")
    s3.download_file(BUCKET, key, str(local_path))
    return local_path


def extract_and_label_frames(
    video_path: Path,
    class_name: str,
    output_img_dir: Path,
    output_lbl_dir: Path,
    model,
    fps: float,
    conf_threshold: float = 0.3,
) -> int:
    """Extract frames from video, run model, save YOLO-format labels.

    Returns number of frames saved.
    """
    output_img_dir.mkdir(parents=True, exist_ok=True)
    output_lbl_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    video_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frame_interval = max(1, int(video_fps / fps))

    class_id = CLASS_TO_ID.get(class_name, 0)
    safe_class = class_name.replace(" ", "_")
    stem = video_path.stem.replace(" ", "_")

    frames_saved = 0
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_interval == 0:
            # Run inference
            results = model(frame, verbose=False)

            img_name = f"{safe_class}_{stem}_f{frame_idx:05d}.jpg"
            lbl_name = f"{safe_class}_{stem}_f{frame_idx:05d}.txt"

            img_path = output_img_dir / img_name
            lbl_path = output_lbl_dir / lbl_name

            cv2.imwrite(str(img_path), frame)

            h, w = frame.shape[:2]
            label_lines = []

            for result in results:
                boxes = result.boxes
                if boxes is None:
                    continue
                for box in boxes:
                    conf = float(box.conf[0])
                    if conf < conf_threshold:
                        continue
                    # Use model-predicted class if confidence is high enough
                    pred_cls = int(box.cls[0])
                    # Convert xyxy to YOLO xywh normalised
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    cx = ((x1 + x2) / 2) / w
                    cy = ((y1 + y2) / 2) / h
                    bw = (x2 - x1) / w
                    bh = (y2 - y1) / h
                    label_lines.append(f"{pred_cls} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

            if not label_lines:
                # No detections — write ground-truth class label with full-frame box
                label_lines.append(f"{class_id} 0.500000 0.500000 1.000000 1.000000")

            lbl_path.write_text("\n".join(label_lines))
            frames_saved += 1

        frame_idx += 1

    cap.release()
    return frames_saved


def main():
    parser = argparse.ArgumentParser(
        description="Extract frames from S3 videos and auto-label with trained model"
    )
    parser.add_argument(
        "--weights",
        required=True,
        help="Path to trained best.pt",
    )
    parser.add_argument(
        "--videos-per-class",
        type=int,
        default=1,
        help="Number of videos to download per class (default: 1)",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=2.0,
        help="Frames per second to extract (default: 2)",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.3,
        help="Confidence threshold for auto-labelling (default: 0.3)",
    )
    parser.add_argument(
        "--output-dir",
        default="data/retail_cv",
        help="Output dataset root (default: data/retail_cv)",
    )
    parser.add_argument(
        "--video-cache",
        default="data/video_cache",
        help="Local directory to cache downloaded videos",
    )
    args = parser.parse_args()

    if not Path(args.weights).exists():
        print(f"Error: weights not found: {args.weights}")
        sys.exit(1)

    from ultralytics import YOLO
    print(f"Loading model: {args.weights}")
    model = YOLO(args.weights)

    s3 = boto3.client(
        "s3",
        aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"],
        region_name=os.environ.get("AWS_DEFAULT_REGION", "us-east-1"),
    )

    video_cache = Path(args.video_cache)
    video_cache.mkdir(parents=True, exist_ok=True)
    output_root = Path(args.output_dir)

    print("\nListing videos on S3...")
    by_class = list_videos_by_class(s3)

    total_frames = 0

    for class_name, videos in by_class.items():
        if not videos:
            print(f"\n  {class_name}: no videos found — skipping")
            continue

        selected = videos[: args.videos_per_class]
        print(f"\n{class_name}: processing {len(selected)} video(s)")

        for obj in selected:
            local_video = download_video(s3, obj, video_cache)

            # Add to train split
            img_dir = output_root / "train" / "images"
            lbl_dir = output_root / "train" / "labels"

            n = extract_and_label_frames(
                video_path=local_video,
                class_name=class_name,
                output_img_dir=img_dir,
                output_lbl_dir=lbl_dir,
                model=model,
                fps=args.fps,
                conf_threshold=args.conf,
            )
            print(f"    Extracted {n} frames → {img_dir}")
            total_frames += n

    print(f"\nDone. Total new frames added to training set: {total_frames}")
    print(f"Dataset location: {output_root.resolve()}")
    print("\nNow retrain with:")
    print(f"  python train.py --epochs 100 --name retail_cv_with_video_frames")


if __name__ == "__main__":
    main()
