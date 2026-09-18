"""
prepare_dataset.py
==================
Downloads labelled images from S3 and organises them into the
train/val/test split that YOLO expects:

    data/retail_cv/
        train/images/   train/labels/
        val/images/     val/labels/
        test/images/    test/labels/

Split ratios: 70% train / 20% val / 10% test (stratified per class).

Usage:
    python prepare_dataset.py                          # download all classes
    python prepare_dataset.py --max-per-class 200      # quick smoke-test run
    python prepare_dataset.py --classes "Touching" "Viewing"  # subset

The script is idempotent — already-downloaded files are skipped via ETag check.
"""

import argparse
import hashlib
import json
import os
import random
import shutil
import sys
from pathlib import Path

import boto3
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

load_dotenv()

BUCKET = os.environ.get("S3_BUCKET", "retaildata-cv-2026")
S3_DATA_PREFIX = os.environ.get("S3_DATA_PREFIX", "extracted_images_and_labels/data")

ALL_CLASSES = [
    "No Interest",
    "Picking And Returning",
    "Picking and Putting",
    "Touching",
    "Turning to Shelf",
    "Viewing",
]

# class_name → integer ID used in YOLO label files
CLASS_TO_ID = {name: i for i, name in enumerate(ALL_CLASSES)}

LOCAL_ROOT = Path("data/retail_cv")
SPLITS = {"train": 0.70, "val": 0.20, "test": 0.10}
RANDOM_SEED = 42


# ---------------------------------------------------------------------------
# ETag helper (skip already-downloaded files)
# ---------------------------------------------------------------------------


def _md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _etag_matches(local_path: Path, s3_etag: str) -> bool:
    if not local_path.exists():
        return False
    clean = s3_etag.strip('"')
    if "-" in clean:  # multipart etag — can't verify, re-download
        return False
    return _md5(local_path) == clean


# ---------------------------------------------------------------------------
# Download helpers
# ---------------------------------------------------------------------------


def _list_objects(s3, prefix: str) -> list[dict]:
    """Return all S3 objects under *prefix*."""
    paginator = s3.get_paginator("list_objects_v2")
    objects = []
    for page in paginator.paginate(Bucket=BUCKET, Prefix=prefix):
        objects.extend(page.get("Contents", []))
        print(f"    listed {len(objects)} objects...", end="\r")
    print()
    return objects


def download_class(
    s3,
    class_name: str,
    max_files: int | None,
    cache_dir: Path,
) -> list[Path]:
    """Download all images (and labels) for one class to cache_dir/{class_name}/.

    Returns list of local image Paths that were successfully downloaded.
    """
    img_prefix = f"{S3_DATA_PREFIX}/{class_name}/images/"
    lbl_prefix = f"{S3_DATA_PREFIX}/{class_name}/labels/"

    img_objects = _list_objects(s3, img_prefix)
    lbl_objects = {
        obj["Key"].split("/")[-1].replace(".txt", ""): obj
        for obj in _list_objects(s3, lbl_prefix)
    }

    if not img_objects:
        print(f"  WARNING: no images found for class '{class_name}'")
        return []

    # Apply max_files limit
    if max_files:
        img_objects = img_objects[:max_files]

    local_img_dir = cache_dir / class_name / "images"
    local_lbl_dir = cache_dir / class_name / "labels"
    local_img_dir.mkdir(parents=True, exist_ok=True)
    local_lbl_dir.mkdir(parents=True, exist_ok=True)

    downloaded: list[Path] = []
    skipped = 0

    for obj in img_objects:
        key = obj["Key"]
        fname = Path(key).name
        stem = Path(fname).stem
        local_img = local_img_dir / fname
        local_lbl = local_lbl_dir / f"{stem}.txt"

        # Skip image if already present and ETag matches
        if _etag_matches(local_img, obj["ETag"]):
            skipped += 1
        else:
            s3.download_file(BUCKET, key, str(local_img))

        # Download label if available
        if stem in lbl_objects and not local_lbl.exists():
            lbl_key = lbl_objects[stem]["Key"]
            s3.download_file(BUCKET, lbl_key, str(local_lbl))

        downloaded.append(local_img)

    print(
        f"  {class_name}: {len(downloaded) - skipped} downloaded, "
        f"{skipped} cached  ({len(downloaded)} total)"
    )
    return downloaded


# ---------------------------------------------------------------------------
# Dataset split + YOLO layout
# ---------------------------------------------------------------------------


def create_yolo_split(
    cache_dir: Path,
    classes: list[str],
    output_root: Path,
) -> dict[str, int]:
    """Organise cached images/labels into train/val/test directories.

    Returns a dict with counts per split.
    """
    random.seed(RANDOM_SEED)

    # Clear old split dirs
    for split in SPLITS:
        shutil.rmtree(output_root / split, ignore_errors=True)

    counts: dict[str, int] = {s: 0 for s in SPLITS}

    for class_name in classes:
        class_id = CLASS_TO_ID[class_name]
        img_dir = cache_dir / class_name / "images"
        lbl_dir = cache_dir / class_name / "labels"

        if not img_dir.exists():
            continue

        imgs = sorted(img_dir.glob("*.jpg")) + sorted(img_dir.glob("*.png"))
        random.shuffle(imgs)

        n = len(imgs)
        n_train = int(n * SPLITS["train"])
        n_val = int(n * SPLITS["val"])

        split_map = (
            [("train", img) for img in imgs[:n_train]]
            + [("val", img) for img in imgs[n_train: n_train + n_val]]
            + [("test", img) for img in imgs[n_train + n_val:]]
        )

        for split, img_path in split_map:
            # Unique filename to avoid collisions across classes
            safe_class = class_name.replace(" ", "_")
            new_name = f"{safe_class}_{img_path.name}"

            dst_img = output_root / split / "images" / new_name
            dst_lbl = output_root / split / "labels" / new_name.replace(
                img_path.suffix, ".txt"
            )
            dst_img.parent.mkdir(parents=True, exist_ok=True)
            dst_lbl.parent.mkdir(parents=True, exist_ok=True)

            shutil.copy2(img_path, dst_img)

            # Copy label if it exists, otherwise create an empty one
            src_lbl = lbl_dir / img_path.with_suffix(".txt").name
            if src_lbl.exists():
                shutil.copy2(src_lbl, dst_lbl)
            else:
                dst_lbl.write_text("")  # image with no annotations

            counts[split] += 1

    return counts


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Download S3 data and prepare YOLO dataset")
    parser.add_argument(
        "--classes",
        nargs="+",
        default=ALL_CLASSES,
        help="Subset of class names to download (default: all)",
    )
    parser.add_argument(
        "--max-per-class",
        type=int,
        default=None,
        help="Limit images per class (useful for quick smoke tests)",
    )
    parser.add_argument(
        "--cache-dir",
        default="data/s3_cache",
        help="Directory to cache raw S3 downloads (default: data/s3_cache)",
    )
    parser.add_argument(
        "--output-dir",
        default="data/retail_cv",
        help="Output directory for YOLO train/val/test split (default: data/retail_cv)",
    )
    args = parser.parse_args()

    # Validate class names
    invalid = [c for c in args.classes if c not in CLASS_TO_ID]
    if invalid:
        print(f"Error: unknown class names: {invalid}")
        print(f"Valid classes: {ALL_CLASSES}")
        sys.exit(1)

    from botocore.config import Config
    s3 = boto3.client(
        "s3",
        aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"],
        region_name=os.environ.get("AWS_DEFAULT_REGION", "us-east-1"),
        config=Config(
            connect_timeout=30,
            read_timeout=60,
            retries={"max_attempts": 10, "mode": "adaptive"},
        ),
    )

    cache_dir = Path(args.cache_dir)
    output_root = Path(args.output_dir)

    print(f"\nDownloading {len(args.classes)} classes from s3://{BUCKET}/...")
    for cls in args.classes:
        download_class(s3, cls, args.max_per_class, cache_dir)

    print("\nOrganising train/val/test split...")
    counts = create_yolo_split(cache_dir, args.classes, output_root)

    total = sum(counts.values())
    print(f"\nDataset ready at: {output_root.resolve()}")
    print(f"  train: {counts['train']} images")
    print(f"  val:   {counts['val']} images")
    print(f"  test:  {counts['test']} images")
    print(f"  total: {total} images")

    # Write a dataset summary
    summary = {
        "classes": args.classes,
        "class_to_id": {c: CLASS_TO_ID[c] for c in args.classes},
        "split_counts": counts,
        "total_images": total,
    }
    (output_root / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nSummary saved to: {output_root / 'summary.json'}")


if __name__ == "__main__":
    main()
