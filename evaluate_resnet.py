#!/usr/bin/env python
"""
VisionIQ ResNet50 (fracture_classifier.pt) evaluation script.

Evaluates the multi-task ResNet classifier on the validation image folder,
computes classification metrics, and saves reports to evaluation_results/resnet/.

Usage:
    python evaluate_resnet.py

Environment:
    VISIONIQ_VAL_DIR  - Path to validation folder (class subdirectories)
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import classification_report, precision_recall_curve

# Ensure backend root is on path for imports
BACKEND_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(BACKEND_ROOT))

from evaluation.eval_utils import (  # noqa: E402
    EVAL_ROOT,
    apply_plot_style,
    binary_metrics,
    collect_validation_samples,
    confusion_matrix_2x2,
    discover_validation_dir,
    ensure_dir,
    print_metrics_table,
)

OUTPUT_DIR = ensure_dir(EVAL_ROOT / "resnet")


def _import_classifier():
    """Import model and inference helpers (same pipeline as production)."""
    try:
        from mainapp.ai import predict_pytorch as pt
    except ImportError as exc:
        raise ImportError(
            "Could not import predict_pytorch. Run from the backend/ directory "
            "and ensure dependencies are installed."
        ) from exc
    return pt


def predict_sample(pt_module, image_path: Path) -> tuple[str, float, float]:
    """
    Run inference using the same logic as predict_fracture().
    Returns (binary_label, confidence_pct, fracture_probability).
    """
    image = Image.open(image_path).convert("RGB")
    tensor = pt_module.transform(image).unsqueeze(0).to(pt_module.device)

    with torch.no_grad():
        outputs = pt_module.model(tensor)

    fracture_prob = torch.sigmoid(outputs["fracture_logit"]).item()
    type_idx = int(torch.argmax(outputs["type_logits"], dim=1).item())
    fracture_labels = pt_module.label_maps.get("fracture_type", ["fracture_unspecified", "none"])
    label = (
        fracture_labels[type_idx]
        if type_idx < len(fracture_labels)
        else "fracture_unspecified"
    )

    if fracture_prob <= 0.5:
        label = "none"

    binary = "fracture" if label != "none" else "none"
    confidence = fracture_prob * 100
    return binary, confidence, fracture_prob


def save_confusion_matrix_plot(cm: np.ndarray, out_path: Path) -> None:
    """Save confusion matrix heatmap PNG."""
    import matplotlib.pyplot as plt
    import seaborn as sns

    apply_plot_style()
    labels = ["No Fracture", "Fracture"]
    fig, ax = plt.subplots(figsize=(7, 5.5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        linewidths=0.5,
        linecolor="white",
        cbar_kws={"label": "Count"},
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("ResNet50 – Confusion Matrix (Validation)")
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


def save_precision_recall_plot(
    y_true: list[str],
    scores: list[float],
    metrics: dict[str, float],
    out_path: Path,
) -> None:
    """Save precision-recall curve (threshold sweep)."""
    import matplotlib.pyplot as plt

    apply_plot_style()
    y_bin = np.array([1 if y == "fracture" else 0 for y in y_true])
    precision, recall, thresholds = precision_recall_curve(y_bin, scores)

    fig, ax = plt.subplots(figsize=(7, 5.5))
    ax.plot(recall, precision, color="#2563eb", linewidth=2.5, label="PR curve")
    ax.scatter(
        [metrics["recall"]],
        [metrics["precision"]],
        color="#f97316",
        s=120,
        zorder=5,
        label=f"Operating point (F1={metrics['f1_score']:.3f})",
        edgecolors="white",
        linewidths=1.5,
    )
    ax.set_xlabel("Recall (Sensitivity)")
    ax.set_ylabel("Precision")
    ax.set_title("ResNet50 – Precision vs Recall")
    ax.set_xlim(0, 1.02)
    ax.set_ylim(0, 1.02)
    ax.legend(loc="lower left")
    ax.grid(True, alpha=0.25)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


def save_metric_bar_chart(metrics: dict[str, float], out_path: Path) -> None:
    """Save bar chart of primary classification metrics."""
    import matplotlib.pyplot as plt

    apply_plot_style()
    keys = ["accuracy", "precision", "recall", "f1_score", "specificity"]
    values = [metrics[k] * 100 for k in keys]
    labels = ["Accuracy", "Precision", "Recall", "F1 Score", "Specificity"]
    colors = ["#2563eb", "#3b82f6", "#60a5fa", "#1d4ed8", "#93c5fd"]

    fig, ax = plt.subplots(figsize=(9, 5.5))
    bars = ax.bar(labels, values, color=colors, edgecolor="white", linewidth=0.8)
    ax.set_ylabel("Score (%)")
    ax.set_ylim(0, 105)
    ax.set_title("ResNet50 – Validation Metrics")
    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1.2,
            f"{val:.1f}%",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="600",
        )
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


def main() -> int:
    print("\nVisionIQ – ResNet50 Classifier Evaluation")
    print("-" * 50)

    # Validate model checkpoint
    checkpoint = BACKEND_ROOT / "mainapp" / "ai" / "models" / "fracture_classifier.pt"
    if not checkpoint.exists():
        print(f"ERROR: Model not found: {checkpoint}")
        return 1

    try:
        val_dir = discover_validation_dir()
    except (FileNotFoundError, NotADirectoryError) as exc:
        print(f"ERROR: {exc}")
        return 1

    print(f"  Model       : {checkpoint}")
    print(f"  Validation  : {val_dir}")
    print(f"  Output dir  : {OUTPUT_DIR}")
    print(f"  Device      : {'CUDA' if torch.cuda.is_available() else 'CPU'}")

    try:
        pt = _import_classifier()
    except ImportError as exc:
        print(f"ERROR: {exc}")
        return 1

    samples = collect_validation_samples(val_dir)
    print(f"\n  Running inference on {len(samples)} images...")

    y_true: list[str] = []
    y_pred: list[str] = []
    scores: list[float] = []
    rows: list[dict] = []

    for idx, (image_path, true_label) in enumerate(samples, start=1):
        try:
            pred_label, confidence, score = predict_sample(pt, image_path)
        except Exception as exc:
            print(f"  [warn] Skipping {image_path.name}: {exc}")
            continue

        y_true.append(true_label)
        y_pred.append(pred_label)
        scores.append(score)
        rows.append(
            {
                "image": image_path.name,
                "true_label": true_label,
                "predicted_label": pred_label,
                "fracture_probability": round(score, 4),
                "confidence_pct": round(confidence, 2),
                "correct": true_label == pred_label,
            }
        )

        if idx % 50 == 0 or idx == len(samples):
            print(f"    Processed {idx}/{len(samples)}", end="\r")

    print(f"\n  Completed {len(y_true)} predictions.")

    if not y_true:
        print("ERROR: No successful predictions.")
        return 1

    metrics = binary_metrics(y_true, y_pred)
    cm = confusion_matrix_2x2(y_true, y_pred)

    # sklearn classification report
    report_dict = classification_report(
        y_true,
        y_pred,
        labels=["none", "fracture"],
        target_names=["No Fracture", "Fracture"],
        output_dict=True,
        zero_division=0,
    )
    report_df = pd.DataFrame(report_dict).transpose()

    # Print results
    display_metrics = {
        "accuracy": metrics["accuracy"],
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "f1_score": metrics["f1_score"],
        "specificity": metrics["specificity"],
        "tp": metrics["tp"],
        "tn": metrics["tn"],
        "fp": metrics["fp"],
        "fn": metrics["fn"],
        "support": metrics["support"],
    }
    print_metrics_table("ResNet50 Validation Metrics", display_metrics)

    print("Confusion Matrix (rows=actual, cols=predicted) [none, fracture]:")
    print(cm)
    print("\nClassification Report:")
    print(classification_report(
        y_true,
        y_pred,
        labels=["none", "fracture"],
        target_names=["No Fracture", "Fracture"],
        zero_division=0,
    ))

    # Save CSV files
    metrics_df = pd.DataFrame(
        [
            {"metric": "accuracy", "value": metrics["accuracy"]},
            {"metric": "precision", "value": metrics["precision"]},
            {"metric": "recall", "value": metrics["recall"]},
            {"metric": "f1_score", "value": metrics["f1_score"]},
            {"metric": "specificity", "value": metrics["specificity"]},
            {"metric": "mAP@0.5", "value": ""},
            {"metric": "mAP@0.5:0.95", "value": ""},
            {"metric": "iou", "value": ""},
            {"metric": "model", "value": "ResNet50"},
        ]
    )
    metrics_df.to_csv(OUTPUT_DIR / "metrics.csv", index=False)
    report_df.to_csv(OUTPUT_DIR / "classification_report.csv")
    pd.DataFrame(rows).to_csv(OUTPUT_DIR / "predictions.csv", index=False)

    print("  Saved CSV reports:")
    print(f"    {OUTPUT_DIR / 'metrics.csv'}")
    print(f"    {OUTPUT_DIR / 'classification_report.csv'}")
    print(f"    {OUTPUT_DIR / 'predictions.csv'}")

    # Save plots
    print("\n  Generating plots...")
    save_confusion_matrix_plot(cm, OUTPUT_DIR / "confusion_matrix.png")
    save_precision_recall_plot(y_true, scores, metrics, OUTPUT_DIR / "precision_recall_curve.png")
    save_metric_bar_chart(metrics, OUTPUT_DIR / "metric_plots.png")

    # Summary file for compare script
    summary = {
        "model": "ResNet50",
        "accuracy": metrics["accuracy"],
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "f1_score": metrics["f1_score"],
        "specificity": metrics["specificity"],
        "mAP@0.5": None,
        "mAP@0.5:0.95": None,
    }
    pd.DataFrame([summary]).to_csv(OUTPUT_DIR / "summary.csv", index=False)

    print("\nResNet50 evaluation completed successfully.\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
