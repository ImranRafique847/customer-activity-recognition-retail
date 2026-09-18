"""
infer_video.py
==============
Run the trained YOLO model on one or multiple videos with:
  - Full class name labels (not abbreviated)
  - Colored transparent overlay on each detected person
  - On-screen legend showing all action colors
  - Support for local files, folders, or S3 videos

Usage:
    # Single local video
    py -3.11 infer_video.py --weights best.pt --source video.mp4

    # All videos in a folder
    py -3.11 infer_video.py --weights best.pt --source-dir path/to/videos/

    # Download from S3
    py -3.11 infer_video.py --weights best.pt --s3-key extracted_videos/Video/Alif Store/Touching/Alif Store-Touching-1.mp4

    # Multiple S3 videos (one per action class)
    py -3.11 infer_video.py --weights best.pt --s3-all --conf 0.25
"""

import argparse
import os
import sys
from pathlib import Path

import boto3
import cv2
import numpy as np
from dotenv import load_dotenv

load_dotenv()

BUCKET = os.environ.get("S3_BUCKET", "retaildata-cv-2026")

# ── Class colors (BGR for OpenCV) ─────────────────────────────────────────────
# Each action class gets a distinct color for its transparent overlay

CLASS_COLORS = {
    "No_Interest":           (128, 128, 128),   # Gray
    "Picking_And_Returning": (0,   165, 255),   # Orange
    "Picking_and_Putting":   (0,   255, 0),     # Green
    "Touching":              (255, 0,   0),     # Blue
    "Turning_to_Shelf":      (255, 0,   255),   # Magenta
    "Viewing":               (0,   255, 255),   # Yellow
}

# Full display names for labels
CLASS_DISPLAY_NAMES = {
    "No_Interest":           "No Interest",
    "Picking_And_Returning": "Picking And Returning",
    "Picking_and_Putting":   "Picking and Putting",
    "Touching":              "Touching",
    "Turning_to_Shelf":      "Turning to Shelf",
    "Viewing":               "Viewing",
}

OVERLAY_ALPHA = 0.35   # Transparency of the colored overlay (0=invisible, 1=solid)
LABEL_FONT    = cv2.FONT_HERSHEY_DUPLEX
LABEL_SCALE   = 0.7
LABEL_THICK   = 2


# ── Drawing helpers ───────────────────────────────────────────────────────────

def draw_detection(frame: np.ndarray, x1: int, y1: int, x2: int, y2: int,
                   class_name: str, confidence: float) -> np.ndarray:
    """
    Draw a colored transparent overlay + solid border + label on a detection.
    The overlay fills the entire bounding box with the class color.
    """
    color   = CLASS_COLORS.get(class_name, (255, 255, 255))
    display = CLASS_DISPLAY_NAMES.get(class_name, class_name.replace("_", " "))

    # Clamp coordinates
    h, w = frame.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)

    if x2 <= x1 or y2 <= y1:
        return frame

    # ── Transparent colored overlay ──────────────────────────────────────────
    overlay = frame.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)   # filled rectangle
    cv2.addWeighted(overlay, OVERLAY_ALPHA, frame, 1 - OVERLAY_ALPHA, 0, frame)

    # ── Solid border ─────────────────────────────────────────────────────────
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)

    # ── Label background + text ───────────────────────────────────────────────
    label      = f"{display}  {confidence:.0%}"
    (lw, lh), baseline = cv2.getTextSize(label, LABEL_FONT, LABEL_SCALE, LABEL_THICK)
    label_y    = max(y1, lh + 10)
    pad        = 6

    cv2.rectangle(frame,
                  (x1, label_y - lh - pad),
                  (x1 + lw + pad * 2, label_y + baseline),
                  color, -1)

    # White text on colored background
    cv2.putText(frame, label,
                (x1 + pad, label_y),
                LABEL_FONT, LABEL_SCALE, (255, 255, 255), LABEL_THICK, cv2.LINE_AA)

    return frame


def draw_legend(frame: np.ndarray) -> np.ndarray:
    """Draw a legend in the top-right corner showing all action colors."""
    h, w = frame.shape[:2]
    box_size  = 22
    pad       = 10
    font_scale = 0.55
    thick      = 1
    items      = list(CLASS_DISPLAY_NAMES.items())

    # Legend background
    legend_w = 260
    legend_h = len(items) * (box_size + 6) + pad * 2
    lx = w - legend_w - pad
    ly = pad

    # Semi-transparent dark background
    overlay = frame.copy()
    cv2.rectangle(overlay, (lx, ly), (lx + legend_w, ly + legend_h),
                  (30, 30, 30), -1)
    cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)

    # Border
    cv2.rectangle(frame, (lx, ly), (lx + legend_w, ly + legend_h),
                  (200, 200, 200), 1)

    # Title
    cv2.putText(frame, "Action Classes",
                (lx + pad, ly + pad + 12),
                LABEL_FONT, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

    # Items
    for i, (cls, display) in enumerate(items):
        color = CLASS_COLORS[cls]
        item_y = ly + pad + 28 + i * (box_size + 6)

        # Color box
        cv2.rectangle(frame,
                      (lx + pad, item_y),
                      (lx + pad + box_size, item_y + box_size),
                      color, -1)
        cv2.rectangle(frame,
                      (lx + pad, item_y),
                      (lx + pad + box_size, item_y + box_size),
                      (255, 255, 255), 1)

        # Text
        cv2.putText(frame, display,
                    (lx + pad + box_size + 8, item_y + box_size - 5),
                    LABEL_FONT, font_scale, (255, 255, 255), thick, cv2.LINE_AA)

    return frame


# ── Inference ─────────────────────────────────────────────────────────────────

def process_video(weights: str, source: str, output: str,
                  conf: float, show: bool) -> None:
    """Process a single video with custom rendering."""
    from ultralytics import YOLO

    model = YOLO(weights)
    cap   = cv2.VideoCapture(source)

    if not cap.isOpened():
        print(f"Error: cannot open video: {source}")
        return

    fps    = cap.get(cv2.CAP_PROP_FPS) or 30
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    Path(output).parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        output,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps, (width, height)
    )

    print(f"\nProcessing: {Path(source).name}")
    print(f"Resolution: {width}x{height}  FPS: {fps:.1f}  Frames: {total}")
    print(f"Output:     {output}\n")

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Run YOLO inference (single frame, no display)
        results = model.predict(frame, conf=conf, verbose=False)[0]

        # Draw each detection with colored overlay
        if results.boxes is not None:
            for box in results.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                cls_id     = int(box.cls[0].item())
                confidence = float(box.conf[0].item())
                class_name = model.names[cls_id]
                frame = draw_detection(frame, x1, y1, x2, y2,
                                       class_name, confidence)

        # Draw legend on every frame
        frame = draw_legend(frame)

        # Frame counter
        cv2.putText(frame, f"Frame {frame_idx+1}/{total}",
                    (10, height - 15),
                    LABEL_FONT, 0.5, (200, 200, 200), 1, cv2.LINE_AA)

        writer.write(frame)

        if show:
            # Resize for display only — don't affect saved video
            display_frame = cv2.resize(frame, (1280, 720))
            cv2.imshow("Retail CV Analytics", display_frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        frame_idx += 1
        if frame_idx % 50 == 0:
            pct = frame_idx / total * 100 if total > 0 else 0
            print(f"  Progress: {frame_idx}/{total} frames ({pct:.1f}%)")

    cap.release()
    writer.release()
    if show:
        cv2.destroyAllWindows()

    print(f"\nDone. Saved: {output}")


# ── S3 helpers ────────────────────────────────────────────────────────────────

def get_s3_client():
    return boto3.client(
        "s3",
        aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"],
        region_name=os.environ.get("AWS_DEFAULT_REGION", "us-east-1"),
    )


def download_from_s3(s3_key: str, local_dir: Path) -> Path:
    s3   = get_s3_client()
    name = s3_key.split("/")[-1]
    local_path = local_dir / name
    if local_path.exists():
        print(f"Using cached: {local_path}")
        return local_path
    print(f"Downloading: {s3_key}")
    local_dir.mkdir(parents=True, exist_ok=True)
    s3.download_file(BUCKET, s3_key, str(local_path))
    return local_path


def get_s3_sample_videos() -> list[str]:
    """Get one sample video per action class from S3."""
    s3 = get_s3_client()
    classes = [
        "No Interest",
        "Touching",
        "Picking and Putting",
        "Picking And Returning",
        "Turning to Shelf",
        "Viewing",
    ]
    keys = []
    for cls in classes:
        prefix = f"extracted_videos/Video/Alif Store/{cls}/"
        r = s3.list_objects_v2(Bucket=BUCKET, Prefix=prefix)
        videos = [o["Key"] for o in r.get("Contents", [])
                  if o["Key"].endswith(".mp4")]
        if videos:
            keys.append(videos[0])
            print(f"  Found: {videos[0].split('/')[-1]}")
    return keys


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Retail CV inference with colored action overlays"
    )

    parser.add_argument("--weights", required=True,
                        help="Path to trained best.pt")
    parser.add_argument("--source", default=None,
                        help="Local video file")
    parser.add_argument("--source-dir", default=None,
                        help="Folder of video files — processes all .mp4 files")
    parser.add_argument("--s3-key", default=None,
                        help="Single S3 video key")
    parser.add_argument("--s3-all", action="store_true",
                        help="Download and process one video per class from S3")
    parser.add_argument("--output-dir", default="output/annotated_videos",
                        help="Output directory (default: output/annotated_videos)")
    parser.add_argument("--conf", type=float, default=0.25,
                        help="Confidence threshold (default: 0.25)")
    parser.add_argument("--show", action="store_true",
                        help="Show live window while processing")
    parser.add_argument("--video-cache", default="data/video_cache",
                        help="Cache dir for S3 downloads")
    args = parser.parse_args()

    if not Path(args.weights).exists():
        print(f"Error: weights not found: {args.weights}")
        sys.exit(1)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir  = Path(args.video_cache)

    # Collect videos to process
    videos = []

    if args.source:
        if not Path(args.source).exists():
            print(f"Error: video not found: {args.source}")
            sys.exit(1)
        videos.append(args.source)

    elif args.source_dir:
        folder = Path(args.source_dir)
        videos = [str(p) for p in folder.glob("*.mp4")]
        if not videos:
            print(f"No .mp4 files found in: {folder}")
            sys.exit(1)
        print(f"Found {len(videos)} videos in {folder}")

    elif args.s3_key:
        local = download_from_s3(args.s3_key, cache_dir)
        videos.append(str(local))

    elif args.s3_all:
        print("Finding one video per action class from S3...")
        keys = get_s3_sample_videos()
        for key in keys:
            local = download_from_s3(key, cache_dir)
            videos.append(str(local))

    else:
        print("Error: provide --source, --source-dir, --s3-key, or --s3-all")
        sys.exit(1)

    # Process each video
    print(f"\nProcessing {len(videos)} video(s)...\n")
    for video_path in videos:
        name   = Path(video_path).stem
        output = str(output_dir / f"{name}_annotated.mp4")
        process_video(
            weights=args.weights,
            source=video_path,
            output=output,
            conf=args.conf,
            show=args.show,
        )

    print(f"\n{'='*50}")
    print(f"All done. Results saved to: {output_dir.resolve()}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
