"""
visualize_results.py
====================
Generates publication-ready visualizations for the CARR dataset paper.

Produces:
  1. Training curves comparison (Baseline vs CBAM)
  2. Per-class mAP comparison bar chart
  3. Final results summary table image
  4. Confusion matrix side by side

Usage:
    py -3.11 visualize_results.py
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from PIL import Image

# ── Paths ─────────────────────────────────────────────────────────────────────

BASELINE_DIR = Path("runs/carr/baseline_5k")
CBAM_DIR     = Path("runs/carr/cbam_5k_v2")
OUTPUT_DIR   = Path("runs/carr/visualizations")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Colors ────────────────────────────────────────────────────────────────────

BASELINE_COLOR = "#2196F3"   # Blue
CBAM_COLOR     = "#FF5722"   # Orange
GRID_COLOR     = "#EEEEEE"
BG_COLOR       = "#FAFAFA"

CLASSES = [
    "No Interest",
    "Picking And\nReturning",
    "Picking and\nPutting",
    "Touching",
    "Turning to\nShelf",
    "Viewing",
]

# Final per-class results
BASELINE_RESULTS = {
    "No Interest":           0.982,
    "Picking And Returning": 0.969,
    "Picking and Putting":   0.979,
    "Touching":              0.986,
    "Turning to Shelf":      0.961,
    "Viewing":               0.994,
    "Overall":               0.979,
}

CBAM_RESULTS = {
    "No Interest":           0.974,
    "Picking And Returning": 0.949,
    "Picking and Putting":   0.982,
    "Touching":              0.986,
    "Turning to Shelf":      0.963,
    "Viewing":               0.994,
    "Overall":               0.975,
}


# ── 1. Training Curves ────────────────────────────────────────────────────────

def plot_training_curves():
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.patch.set_facecolor(BG_COLOR)
    fig.suptitle("Training Curves: YOLO11s Baseline vs YOLO11s+CBAM\nCAR Dataset (5K per class, 200 epochs)",
                 fontsize=13, fontweight="bold", y=1.02)

    for ax in axes:
        ax.set_facecolor(BG_COLOR)
        ax.grid(True, color=GRID_COLOR, linewidth=0.8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    # Load CSVs
    baseline_csv = BASELINE_DIR / "results.csv"
    cbam_csv     = CBAM_DIR     / "results.csv"

    if not baseline_csv.exists() or not cbam_csv.exists():
        print("WARNING: results.csv not found — skipping training curves")
        return

    b_df = pd.read_csv(baseline_csv)
    c_df = pd.read_csv(cbam_csv)

    # Strip whitespace from column names
    b_df.columns = b_df.columns.str.strip()
    c_df.columns = c_df.columns.str.strip()

    epochs_b = b_df["epoch"] if "epoch" in b_df.columns else range(len(b_df))
    epochs_c = c_df["epoch"] if "epoch" in c_df.columns else range(len(c_df))

    # mAP@0.5
    ax = axes[0]
    map50_col = [c for c in b_df.columns if "mAP50" in c and "95" not in c]
    if map50_col:
        ax.plot(epochs_b, b_df[map50_col[0]], color=BASELINE_COLOR,
                linewidth=2, label="YOLO11s Baseline")
        ax.plot(epochs_c, c_df[map50_col[0]], color=CBAM_COLOR,
                linewidth=2, label="YOLO11s + CBAM", linestyle="--")
    ax.set_xlabel("Epoch", fontsize=11)
    ax.set_ylabel("mAP@0.5", fontsize=11)
    ax.set_title("mAP@0.5 over Training", fontsize=12, fontweight="bold")
    ax.legend(fontsize=10)
    ax.set_ylim(0, 1.05)

    # Box Loss
    ax = axes[1]
    loss_col = [c for c in b_df.columns if "box_loss" in c.lower() or "train/box" in c.lower()]
    if loss_col:
        ax.plot(epochs_b, b_df[loss_col[0]], color=BASELINE_COLOR,
                linewidth=2, label="YOLO11s Baseline")
        ax.plot(epochs_c, c_df[loss_col[0]], color=CBAM_COLOR,
                linewidth=2, label="YOLO11s + CBAM", linestyle="--")
    ax.set_xlabel("Epoch", fontsize=11)
    ax.set_ylabel("Box Loss", fontsize=11)
    ax.set_title("Training Box Loss", fontsize=12, fontweight="bold")
    ax.legend(fontsize=10)

    plt.tight_layout()
    out = OUTPUT_DIR / "training_curves.png"
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=BG_COLOR)
    plt.close()
    print(f"Saved: {out}")


# ── 2. Per-Class mAP Bar Chart ────────────────────────────────────────────────

def plot_per_class_map():
    fig, ax = plt.subplots(figsize=(12, 6))
    fig.patch.set_facecolor(BG_COLOR)
    ax.set_facecolor(BG_COLOR)
    ax.grid(True, axis="y", color=GRID_COLOR, linewidth=0.8, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    classes = list(BASELINE_RESULTS.keys())
    baseline_vals = list(BASELINE_RESULTS.values())
    cbam_vals     = list(CBAM_RESULTS.values())

    x = np.arange(len(classes))
    width = 0.35

    bars1 = ax.bar(x - width/2, baseline_vals, width,
                   label="YOLO11s Baseline", color=BASELINE_COLOR,
                   alpha=0.85, zorder=3)
    bars2 = ax.bar(x + width/2, cbam_vals, width,
                   label="YOLO11s + CBAM", color=CBAM_COLOR,
                   alpha=0.85, zorder=3)

    # Add value labels on bars
    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
                f"{bar.get_height():.3f}", ha="center", va="bottom",
                fontsize=8, fontweight="bold", color=BASELINE_COLOR)

    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
                f"{bar.get_height():.3f}", ha="center", va="bottom",
                fontsize=8, fontweight="bold", color=CBAM_COLOR)

    ax.set_xlabel("Class", fontsize=12)
    ax.set_ylabel("mAP@0.5", fontsize=12)
    ax.set_title("Per-Class mAP@0.5: YOLO11s Baseline vs YOLO11s+CBAM\nCAR Dataset — First External Benchmark",
                 fontsize=13, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(classes, fontsize=10)
    ax.set_ylim(0.85, 1.02)
    ax.legend(fontsize=11)

    plt.tight_layout()
    out = OUTPUT_DIR / "per_class_map.png"
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=BG_COLOR)
    plt.close()
    print(f"Saved: {out}")


# ── 3. Summary Table ──────────────────────────────────────────────────────────

def plot_summary_table():
    fig, ax = plt.subplots(figsize=(12, 4))
    fig.patch.set_facecolor(BG_COLOR)
    ax.set_facecolor(BG_COLOR)
    ax.axis("off")

    data = [
        ["YOLO11s (Baseline)", "9.43M", "0.979", "0.863", "0.939", "0.945", "3.3ms", "15.7 hrs"],
        ["YOLO11s + CBAM",     "9.48M", "0.975", "0.795", "0.933", "0.943", "3.5ms", "17.1 hrs"],
        ["GSW-Yolo (authors)", "~4.8M", "~0.90*", "-",    "-",     "-",     "-",     "-"],
    ]

    columns = ["Model", "Params", "mAP@0.5", "mAP@0.5:0.95",
               "Precision", "Recall", "Inference", "Train Time"]

    table = ax.table(
        cellText=data,
        colLabels=columns,
        loc="center",
        cellLoc="center",
    )

    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2.2)

    # Style header
    for j in range(len(columns)):
        table[0, j].set_facecolor("#1565C0")
        table[0, j].set_text_props(color="white", fontweight="bold")

    # Style baseline row
    for j in range(len(columns)):
        table[1, j].set_facecolor("#E3F2FD")

    # Style CBAM row
    for j in range(len(columns)):
        table[2, j].set_facecolor("#FFF3E0")

    # Style authors row
    for j in range(len(columns)):
        table[3, j].set_facecolor("#F5F5F5")
        table[3, j].set_text_props(color="#757575")

    ax.set_title("Results Summary — CARR Dataset Benchmark\n*GSW-Yolo reported on different evaluation protocol",
                 fontsize=12, fontweight="bold", pad=20)

    plt.tight_layout()
    out = OUTPUT_DIR / "results_table.png"
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=BG_COLOR)
    plt.close()
    print(f"Saved: {out}")


# ── 4. Confusion Matrices Side by Side ───────────────────────────────────────

def plot_confusion_matrices():
    b_cm = BASELINE_DIR / "confusion_matrix_normalized.png"
    c_cm = CBAM_DIR     / "confusion_matrix_normalized.png"

    if not b_cm.exists() or not c_cm.exists():
        print("WARNING: confusion matrix images not found — skipping")
        return

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.patch.set_facecolor(BG_COLOR)
    fig.suptitle("Confusion Matrices: Baseline vs CBAM\nCAR Dataset",
                 fontsize=13, fontweight="bold")

    axes[0].imshow(Image.open(b_cm))
    axes[0].set_title("YOLO11s Baseline\nmAP@0.5 = 0.979",
                      fontsize=12, fontweight="bold", color=BASELINE_COLOR)
    axes[0].axis("off")

    axes[1].imshow(Image.open(c_cm))
    axes[1].set_title("YOLO11s + CBAM\nmAP@0.5 = 0.975",
                      fontsize=12, fontweight="bold", color=CBAM_COLOR)
    axes[1].axis("off")

    plt.tight_layout()
    out = OUTPUT_DIR / "confusion_matrices.png"
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=BG_COLOR)
    plt.close()
    print(f"Saved: {out}")


# ── 5. LinkedIn Post Image ────────────────────────────────────────────────────

def plot_linkedin_summary():
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    fig.patch.set_facecolor("#1A1A2E")
    fig.suptitle("Customer Activity Recognition in Retail (CARR)\nFirst External Benchmark — YOLO11s",
                 fontsize=16, fontweight="bold", color="white", y=1.02)

    classes_short = ["No Interest", "Picking &\nReturning",
                     "Picking &\nPutting", "Touching",
                     "Turning to\nShelf", "Viewing"]

    baseline_vals = [0.982, 0.969, 0.979, 0.986, 0.961, 0.994]
    cbam_vals     = [0.974, 0.949, 0.982, 0.986, 0.963, 0.994]
    colors        = ["#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4", "#FFEAA7", "#DDA0DD"]

    for i, ax in enumerate(axes.flat):
        ax.set_facecolor("#16213E")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color("#444")
        ax.spines["bottom"].set_color("#444")

        bars = ax.bar(["Baseline", "CBAM"],
                      [baseline_vals[i], cbam_vals[i]],
                      color=[colors[i], colors[i]],
                      width=0.5)
        bars[0].set_alpha(1.0)
        bars[1].set_alpha(0.65)

        ax.set_ylim(0.85, 1.02)
        ax.set_title(classes_short[i], fontsize=11,
                     fontweight="bold", color="white", pad=8)
        ax.tick_params(colors="white", labelsize=9)
        ax.yaxis.grid(True, color="#333", linewidth=0.5)

        for bar in bars:
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + 0.001,
                    f"{bar.get_height():.3f}",
                    ha="center", va="bottom",
                    fontsize=9, color="white", fontweight="bold")

    plt.tight_layout()
    out = OUTPUT_DIR / "linkedin_summary.png"
    plt.savefig(out, dpi=150, bbox_inches="tight",
                facecolor="#1A1A2E")
    plt.close()
    print(f"Saved: {out}")


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Generating visualizations...\n")
    plot_training_curves()
    plot_per_class_map()
    plot_summary_table()
    plot_confusion_matrices()
    plot_linkedin_summary()
    print(f"\nAll visualizations saved to: {OUTPUT_DIR.resolve()}")
    print("\nFiles:")
    for f in sorted(OUTPUT_DIR.glob("*.png")):
        print(f"  {f.name}")
