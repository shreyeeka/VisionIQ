#!/usr/bin/env python
"""
VisionIQ combined model comparison report.

Reads ResNet50 and YOLOv8 evaluation outputs and generates:
  - evaluation_results/final_report.csv
  - Comparison bar charts
  - Best model summaries printed to terminal

Usage:
    python compare_models.py

Prerequisites:
    python evaluate_resnet.py
    python evaluate_yolo.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

BACKEND_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(BACKEND_ROOT))

from evaluation.eval_utils import EVAL_ROOT, apply_plot_style, ensure_dir  # noqa: E402

RESNET_DIR = EVAL_ROOT / "resnet"
YOLO_DIR = EVAL_ROOT / "yolo"
OUTPUT_DIR = ensure_dir(EVAL_ROOT)

FINAL_COLUMNS = [
    "Model",
    "Accuracy",
    "Precision",
    "Recall",
    "F1 Score",
    "Specificity",
    "mAP@0.5",
    "mAP@0.5:0.95",
    "IoU",
]


def _load_summary(path: Path, model_name: str) -> dict | None:
    """Load summary.csv or parse metrics.csv from an evaluation folder."""
    summary_path = path / "summary.csv"
    metrics_path = path / "metrics.csv"

    if summary_path.exists():
        row = pd.read_csv(summary_path).iloc[0].to_dict()
        return {
            "Model": row.get("model", model_name),
            "Accuracy": _fmt(row.get("accuracy")),
            "Precision": _fmt(row.get("precision")),
            "Recall": _fmt(row.get("recall")),
            "F1 Score": _fmt(row.get("f1_score")),
            "Specificity": _fmt(row.get("specificity")),
            "mAP@0.5": _fmt(row.get("mAP@0.5")),
            "mAP@0.5:0.95": _fmt(row.get("mAP@0.5:0.95")),
            "IoU": _fmt(row.get("iou")),
        }

    if metrics_path.exists():
        df = pd.read_csv(metrics_path)
        lookup = dict(zip(df["metric"], df["value"]))
        return {
            "Model": lookup.get("model", model_name),
            "Accuracy": _fmt(lookup.get("accuracy")),
            "Precision": _fmt(lookup.get("precision")),
            "Recall": _fmt(lookup.get("recall")),
            "F1 Score": _fmt(lookup.get("f1_score")),
            "Specificity": _fmt(lookup.get("specificity")),
            "mAP@0.5": _fmt(lookup.get("mAP@0.5")),
            "mAP@0.5:0.95": _fmt(lookup.get("mAP@0.5:0.95")),
            "IoU": _fmt(lookup.get("iou")),
        }

    return None


def _fmt(value) -> float | str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    if value == "":
        return ""
    try:
        return round(float(value), 4)
    except (TypeError, ValueError):
        return ""


def _to_float(value) -> float:
    if value == "" or value is None:
        return -1.0
    try:
        return float(value)
    except (TypeError, ValueError):
        return -1.0


def save_comparison_charts(report_df: pd.DataFrame) -> None:
    """Generate combined comparison visualizations."""
    import matplotlib.pyplot as plt
    import numpy as np

    apply_plot_style()

    models = report_df["Model"].tolist()

    # Accuracy / F1 comparison (classification-focused)
    fig, ax = plt.subplots(figsize=(8, 5.5))
    x = np.arange(len(models))
    width = 0.35
    accuracy = [_to_float(v) * 100 for v in report_df["Accuracy"]]
    f1 = [_to_float(v) * 100 for v in report_df["F1 Score"]]

    ax.bar(x - width / 2, accuracy, width, label="Accuracy", color="#2563eb")
    ax.bar(x + width / 2, f1, width, label="F1 Score", color="#f97316")
    ax.set_ylabel("Score (%)")
    ax.set_title("Classification Metrics – Model Comparison")
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.set_ylim(0, 105)
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    path1 = OUTPUT_DIR / "accuracy_f1_comparison.png"
    fig.savefig(path1, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path1}")

    # Multi-metric comparison bar chart
    metric_cols = ["Precision", "Recall", "F1 Score", "mAP@0.5", "mAP@0.5:0.95", "IoU"]
    fig, axes = plt.subplots(1, len(models), figsize=(6 * len(models), 5.5), squeeze=False)

    colors = ["#2563eb", "#3b82f6", "#1d4ed8", "#60a5fa", "#93c5fd", "#f97316"]

    for ax, (_, row) in zip(axes[0], report_df.iterrows()):
        vals = [_to_float(row[c]) * 100 for c in metric_cols]
        vals = [max(v, 0) for v in vals]
        bars = ax.bar(metric_cols, vals, color=colors, edgecolor="white")
        ax.set_title(str(row["Model"]))
        ax.set_ylabel("Score (%)")
        ax.set_ylim(0, 105)
        ax.tick_params(axis="x", rotation=30)
        for bar, val in zip(bars, vals):
            if val > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 1,
                    f"{val:.0f}%",
                    ha="center",
                    fontsize=8,
                )

    plt.suptitle("VisionIQ – Metric Comparison", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    path2 = OUTPUT_DIR / "metric_comparison.png"
    fig.savefig(path2, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path2}")

    # Precision vs Recall scatter
    fig, ax = plt.subplots(figsize=(7, 5.5))
    for _, row in report_df.iterrows():
        p = _to_float(row["Precision"])
        r = _to_float(row["Recall"])
        if p < 0 or r < 0:
            continue
        ax.scatter(r * 100, p * 100, s=180, label=str(row["Model"]), edgecolors="white", linewidths=2)
        ax.annotate(
            str(row["Model"]),
            (r * 100, p * 100),
            textcoords="offset points",
            xytext=(8, 6),
            fontsize=10,
        )
    ax.set_xlabel("Recall (%)")
    ax.set_ylabel("Precision (%)")
    ax.set_title("Precision vs Recall – Model Comparison")
    ax.set_xlim(0, 105)
    ax.set_ylim(0, 105)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="lower left")
    plt.tight_layout()
    path3 = OUTPUT_DIR / "precision_recall_comparison.png"
    fig.savefig(path3, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path3}")


def pick_best_classification(report_df: pd.DataFrame) -> str:
    """Best classifier by F1, then accuracy."""
    resnet_rows = report_df[report_df["Model"].str.contains("ResNet", case=False, na=False)]
    if resnet_rows.empty:
        resnet_rows = report_df.head(1)

    best = None
    best_score = -1.0
    for _, row in resnet_rows.iterrows():
        f1 = _to_float(row["F1 Score"])
        acc = _to_float(row["Accuracy"])
        score = f1 if f1 >= 0 else acc
        if score > best_score:
            best_score = score
            best = row["Model"]
    return best or "N/A"


def pick_best_detection(report_df: pd.DataFrame) -> str:
    """Best detector by mAP@0.5, then F1."""
    yolo_rows = report_df[report_df["Model"].str.contains("YOLO", case=False, na=False)]
    if yolo_rows.empty:
        yolo_rows = report_df.tail(1)

    best = None
    best_score = -1.0
    for _, row in yolo_rows.iterrows():
        map50 = _to_float(row["mAP@0.5"])
        f1 = _to_float(row["F1 Score"])
        score = map50 if map50 >= 0 else f1
        if score > best_score:
            best_score = score
            best = row["Model"]
    return best or "N/A"


def main() -> int:
    print("\nVisionIQ – Combined Model Comparison")
    print("-" * 50)

    missing = []
    resnet_ok = RESNET_DIR.exists() and (
        (RESNET_DIR / "metrics.csv").exists() or (RESNET_DIR / "summary.csv").exists()
    )
    yolo_ok = YOLO_DIR.exists() and (
        (YOLO_DIR / "metrics.csv").exists() or (YOLO_DIR / "summary.csv").exists()
    )
    if not resnet_ok:
        missing.append("ResNet (run: python evaluate_resnet.py)")
    if not yolo_ok:
        missing.append("YOLO (run: python evaluate_yolo.py)")

    resnet = _load_summary(RESNET_DIR, "ResNet50")
    yolo = _load_summary(YOLO_DIR, "YOLOv8")

    rows = [r for r in (resnet, yolo) if r is not None]
    if not rows:
        print("ERROR: No evaluation results found.")
        for msg in missing:
            print(f"  - {msg}")
        return 1

    if missing:
        print("WARNING: Incomplete evaluations:")
        for msg in missing:
            print(f"  - {msg}")

    report_df = pd.DataFrame(rows, columns=FINAL_COLUMNS)
    out_csv = OUTPUT_DIR / "final_report.csv"
    report_df.to_csv(out_csv, index=False)

    print(f"\n  Saved: {out_csv}\n")
    print("Final Report:")
    print(report_df.to_string(index=False))

    print("\n  Generating comparison charts...")
    save_comparison_charts(report_df)

    best_cls = pick_best_classification(report_df)
    best_det = pick_best_detection(report_df)

    print("\n" + "=" * 60)
    print(f'  Best Classification Model : {best_cls}')
    print(f'  Best Detection Model      : {best_det}')
    print("=" * 60 + "\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
