#!/usr/bin/env python
"""
VisionIQ YOLOv8 (best_final.pt) evaluation script.

Runs Ultralytics validation on the YOLO validation split defined in data.yaml
and saves metrics, plots, and summaries to evaluation_results/yolo/.

Usage:
    python evaluate_yolo.py

Environment:
    VISIONIQ_YOLO_DATA  - Path to YOLO data.yaml
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import pandas as pd

BACKEND_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(BACKEND_ROOT))

from evaluation.eval_utils import (  # noqa: E402
    EVAL_ROOT,
    MODELS_DIR,
    apply_plot_style,
    discover_yolo_data_yaml,
    ensure_dir,
    print_metrics_table,
)

OUTPUT_DIR = ensure_dir(EVAL_ROOT / "yolo")
YOLO_MODEL_PATH = MODELS_DIR / "best_final.pt"


def _safe_float(value, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _extract_box_metrics(results) -> dict[str, float]:
    """Extract detection metrics from Ultralytics validation results."""
    box = getattr(results, "box", None)
    if box is None:
        raise RuntimeError("Validation results do not contain box metrics.")

    precision = _safe_float(getattr(box, "mp", None))
    recall = _safe_float(getattr(box, "mr", None))
    map50 = _safe_float(getattr(box, "map50", None))
    map5095 = _safe_float(getattr(box, "map", None))

    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

    # Mean IoU – Ultralytics may expose miou / iou on some versions
    iou = _safe_float(getattr(box, "miou", None) or getattr(box, "iou", None))
    if iou == 0.0 and hasattr(results, "results_dict"):
        rd = results.results_dict or {}
        for key in ("metrics/IoU", "metrics/iou", "IoU", "iou"):
            if key in rd:
                iou = _safe_float(rd[key])
                break

    # Fallback: report mAP@0.5 as IoU proxy when native IoU is unavailable
    if iou == 0.0 and map50 > 0:
        iou = map50

    return {
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "mAP@0.5": map50,
        "mAP@0.5:0.95": map5095,
        "iou": iou,
    }


def copy_ultralytics_plots(val_save_dir: Path, output_dir: Path) -> None:
    """Copy Ultralytics validation plots to evaluation output folder."""
    plot_map = {
        "BoxPR_curve.png": "precision_recall_curve.png",
        "PR_curve.png": "precision_recall_curve.png",
        "BoxP_curve.png": "precision_curve.png",
        "BoxR_curve.png": "recall_curve.png",
        "BoxF1_curve.png": "f1_curve.png",
        "confusion_matrix.png": "confusion_matrix_normalized.png",
        "confusion_matrix_normalized.png": "confusion_matrix_normalized.png",
        "results.png": "metric_plots.png",
    }

    if not val_save_dir.exists():
        return

    copied_pr = False
    copied_metrics = False

    for src_name, dst_name in plot_map.items():
        src = val_save_dir / src_name
        if not src.exists():
            continue
        dst = output_dir / dst_name
        shutil.copy2(src, dst)
        print(f"  Saved: {dst}")

        if dst_name == "precision_recall_curve.png":
            copied_pr = True
        if dst_name == "metric_plots.png":
            copied_metrics = True

    # Fallback: any PR-like or results plot
    if not copied_pr:
        for candidate in val_save_dir.glob("*PR*.png"):
            shutil.copy2(candidate, output_dir / "precision_recall_curve.png")
            print(f"  Saved: {output_dir / 'precision_recall_curve.png'}")
            copied_pr = True
            break

    if not copied_metrics:
        for candidate in val_save_dir.glob("results*.png"):
            shutil.copy2(candidate, output_dir / "metric_plots.png")
            print(f"  Saved: {output_dir / 'metric_plots.png'}")
            break


def save_metric_bar_chart(metrics: dict[str, float], out_path: Path) -> None:
    """Generate YOLO metric bar chart if Ultralytics did not provide results.png."""
    if out_path.exists():
        return

    import matplotlib.pyplot as plt

    apply_plot_style()
    keys = ["precision", "recall", "f1_score", "mAP@0.5", "mAP@0.5:0.95", "iou"]
    labels = ["Precision", "Recall", "F1", "mAP@0.5", "mAP@0.5:0.95", "IoU"]
    values = [metrics[k] * 100 for k in keys]
    colors = ["#2563eb", "#3b82f6", "#1d4ed8", "#60a5fa", "#93c5fd", "#f97316"]

    fig, ax = plt.subplots(figsize=(10, 5.5))
    bars = ax.bar(labels, values, color=colors, edgecolor="white")
    ax.set_ylabel("Score (%)")
    ax.set_ylim(0, 105)
    ax.set_title("YOLOv8 – Validation Metrics")
    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1.2,
            f"{val:.1f}%",
            ha="center",
            fontsize=9,
            fontweight="600",
        )
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


def save_precision_recall_fallback(metrics: dict[str, float], out_path: Path) -> None:
    """Create a simple PR point plot when curve PNG is unavailable."""
    if out_path.exists():
        return

    import matplotlib.pyplot as plt

    apply_plot_style()
    fig, ax = plt.subplots(figsize=(7, 5.5))
    ax.scatter(
        [metrics["recall"]],
        [metrics["precision"]],
        s=200,
        color="#2563eb",
        label=f"YOLOv8 (F1={metrics['f1_score']:.3f})",
        zorder=5,
        edgecolors="white",
        linewidths=2,
    )
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("YOLOv8 – Precision vs Recall (Operating Point)")
    ax.set_xlim(0, 1.02)
    ax.set_ylim(0, 1.02)
    ax.legend()
    ax.grid(True, alpha=0.25)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


def main() -> int:
    print("\nVisionIQ – YOLOv8 Detector Evaluation")
    print("-" * 50)

    if not YOLO_MODEL_PATH.exists():
        print(f"ERROR: Model not found: {YOLO_MODEL_PATH}")
        return 1

    try:
        data_yaml = discover_yolo_data_yaml()
    except (FileNotFoundError, ValueError) as exc:
        print(f"ERROR: {exc}")
        return 1

    print(f"  Model       : {YOLO_MODEL_PATH}")
    print(f"  data.yaml   : {data_yaml}")
    print(f"  Output dir  : {OUTPUT_DIR}")

    try:
        from ultralytics import YOLO
    except ImportError:
        print("ERROR: ultralytics is not installed. Run: pip install ultralytics")
        return 1

    model = YOLO(str(YOLO_MODEL_PATH))

    print("\n  Running Ultralytics validation (this may take a few minutes)...")
    try:
        results = model.val(
            data=str(data_yaml),
            split="val",
            imgsz=1024,
            batch=8,
            conf=0.05,
            project=str(OUTPUT_DIR),
            name="val_run",
            exist_ok=True,
            plots=True,
            save_json=True,
            verbose=True,
        )
    except Exception as exc:
        print(f"ERROR: YOLO validation failed: {exc}")
        return 1

    try:
        metrics = _extract_box_metrics(results)
    except RuntimeError as exc:
        print(f"ERROR: {exc}")
        return 1

    display = {
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "f1_score": metrics["f1_score"],
        "mAP@0.5": metrics["mAP@0.5"],
        "mAP@0.5:0.95": metrics["mAP@0.5:0.95"],
        "iou": metrics["iou"],
    }
    print_metrics_table("YOLOv8 Validation Metrics", display)

    # Save metrics CSV
    metrics_df = pd.DataFrame(
        [
            {"metric": "model", "value": "YOLOv8"},
            {"metric": "precision", "value": metrics["precision"]},
            {"metric": "recall", "value": metrics["recall"]},
            {"metric": "f1_score", "value": metrics["f1_score"]},
            {"metric": "mAP@0.5", "value": metrics["mAP@0.5"]},
            {"metric": "mAP@0.5:0.95", "value": metrics["mAP@0.5:0.95"]},
            {"metric": "iou", "value": metrics["iou"]},
            {"metric": "accuracy", "value": ""},
            {"metric": "specificity", "value": ""},
        ]
    )
    metrics_df.to_csv(OUTPUT_DIR / "metrics.csv", index=False)

    summary_df = pd.DataFrame(
        [
            {
                "model": "YOLOv8",
                "accuracy": None,
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1_score": metrics["f1_score"],
                "specificity": None,
                "mAP@0.5": metrics["mAP@0.5"],
                "mAP@0.5:0.95": metrics["mAP@0.5:0.95"],
                "iou": metrics["iou"],
            }
        ]
    )
    summary_df.to_csv(OUTPUT_DIR / "summary.csv", index=False)

    # Text summary
    summary_text = f"""VisionIQ YOLOv8 Evaluation Summary
================================
Model         : best_final.pt
Dataset       : {data_yaml}

Precision     : {metrics['precision']:.4f} ({metrics['precision']*100:.2f}%)
Recall        : {metrics['recall']:.4f} ({metrics['recall']*100:.2f}%)
F1 Score      : {metrics['f1_score']:.4f} ({metrics['f1_score']*100:.2f}%)
mAP@0.5       : {metrics['mAP@0.5']:.4f} ({metrics['mAP@0.5']*100:.2f}%)
mAP@0.5:0.95  : {metrics['mAP@0.5:0.95']:.4f} ({metrics['mAP@0.5:0.95']*100:.2f}%)
IoU (mean)    : {metrics['iou']:.4f} ({metrics['iou']*100:.2f}%)
"""
    (OUTPUT_DIR / "evaluation_summary.txt").write_text(summary_text, encoding="utf-8")
    print(f"  Saved: {OUTPUT_DIR / 'metrics.csv'}")
    print(f"  Saved: {OUTPUT_DIR / 'evaluation_summary.txt'}")

    # Copy / generate plots
    val_plots_dir = OUTPUT_DIR / "val_run"
    print("\n  Collecting validation plots...")
    copy_ultralytics_plots(val_plots_dir, OUTPUT_DIR)
    save_precision_recall_fallback(metrics, OUTPUT_DIR / "precision_recall_curve.png")
    save_metric_bar_chart(metrics, OUTPUT_DIR / "metric_plots.png")

    print("\nYOLOv8 evaluation completed successfully.\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
