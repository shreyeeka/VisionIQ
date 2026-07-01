"""
Shared utilities for VisionIQ model evaluation.
Handles dataset discovery, metrics computation, and plot styling.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np

# Project root = backend/ (parent of evaluation/)
BACKEND_ROOT = Path(__file__).resolve().parent.parent
AI_DIR = BACKEND_ROOT / "mainapp" / "ai"
MODELS_DIR = AI_DIR / "models"
EVAL_ROOT = BACKEND_ROOT / "evaluation_results"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

# Folder name aliases -> canonical binary label
FRACTURE_ALIASES = {
    "fracture",
    "fractured",
    "fracture_unspecified",
    "positive",
    "pos",
    "1",
    "yes",
}
NORMAL_ALIASES = {
    "none",
    "normal",
    "negative",
    "neg",
    "0",
    "no",
    "healthy",
    "non_fractured",
    "nonfractured",
}


def ensure_dir(path: Path) -> Path:
    """Create directory if missing and return path."""
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_label_maps() -> dict[str, list[str]]:
    """Load fracture label mapping used by the classifier."""
    label_path = MODELS_DIR / "label_maps.json"
    if not label_path.exists():
        raise FileNotFoundError(f"Label map not found: {label_path}")
    with label_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def normalize_folder_label(folder_name: str) -> str:
    """
    Map dataset folder names to canonical labels: 'fracture' or 'none'.
    """
    key = folder_name.strip().lower().replace(" ", "_").replace("-", "_")
    if key in FRACTURE_ALIASES:
        return "fracture"
    if key in NORMAL_ALIASES:
        return "none"
    if "fracture" in key and "non" not in key:
        return "fracture"
    raise ValueError(
        f"Unrecognized class folder '{folder_name}'. "
        f"Expected aliases like {sorted(FRACTURE_ALIASES | NORMAL_ALIASES)}"
    )


def discover_validation_dir() -> Path:
    """
    Resolve validation image directory for ResNet evaluation.

    Priority:
      1. VISIONIQ_VAL_DIR environment variable
      2. evaluation_config.json -> resnet.validation_dir
      3. Common relative paths under project / parent folders
    """
    env_path = os.environ.get("VISIONIQ_VAL_DIR")
    if env_path:
        path = Path(env_path).expanduser().resolve()
        _validate_image_dir(path)
        return path

    config_path = BACKEND_ROOT / "evaluation_config.json"
    if config_path.exists():
        with config_path.open("r", encoding="utf-8") as f:
            cfg = json.load(f)
        val_dir = cfg.get("resnet", {}).get("validation_dir")
        if val_dir:
            path = Path(val_dir).expanduser()
            if not path.is_absolute():
                path = (BACKEND_ROOT / path).resolve()
            else:
                path = path.resolve()
            _validate_image_dir(path)
            return path

    candidates = [
        BACKEND_ROOT / "dataset" / "validation",
        BACKEND_ROOT / "dataset" / "val",
        BACKEND_ROOT / "data" / "validation",
        BACKEND_ROOT / "data" / "val",
        BACKEND_ROOT.parent / "dataset" / "validation",
        BACKEND_ROOT.parent / "dataset" / "val",
        BACKEND_ROOT.parent / "data" / "validation",
        BACKEND_ROOT.parent / "data" / "val",
    ]

    for candidate in candidates:
        if candidate.exists() and _has_class_subdirs(candidate):
            return candidate.resolve()

    raise FileNotFoundError(
        "Validation dataset not found.\n"
        "Set VISIONIQ_VAL_DIR to your validation folder, or create evaluation_config.json:\n"
        '  {"resnet": {"validation_dir": "path/to/validation"}}\n'
        "Expected structure:\n"
        "  validation/\n"
        "    fracture_unspecified/  (or fracture/, positive/)\n"
        "    none/                  (or normal/, negative/)\n"
        f"Searched: {[str(c) for c in candidates]}"
    )


def discover_yolo_data_yaml() -> Path:
    """
    Resolve YOLO data.yaml for validation.

    Priority:
      1. VISIONIQ_YOLO_DATA environment variable
      2. evaluation_config.json -> yolo.data_yaml
      3. Common locations
    """
    env_path = os.environ.get("VISIONIQ_YOLO_DATA")
    if env_path:
        path = Path(env_path).expanduser().resolve()
        _validate_data_yaml(path)
        return path

    config_path = BACKEND_ROOT / "evaluation_config.json"
    if config_path.exists():
        with config_path.open("r", encoding="utf-8") as f:
            cfg = json.load(f)
        data_yaml = cfg.get("yolo", {}).get("data_yaml")
        if data_yaml:
            path = Path(data_yaml).expanduser()
            if not path.is_absolute():
                path = (BACKEND_ROOT / path).resolve()
            else:
                path = path.resolve()
            _validate_data_yaml(path)
            return path

    candidates = [
        BACKEND_ROOT / "dataset" / "data.yaml",
        BACKEND_ROOT / "dataset" / "dataset.yaml",
        BACKEND_ROOT / "data.yaml",
        BACKEND_ROOT.parent / "dataset" / "data.yaml",
        BACKEND_ROOT.parent / "data" / "data.yaml",
        BACKEND_ROOT / "yolo_dataset" / "data.yaml",
    ]

    for candidate in candidates:
        if candidate.exists():
            _validate_data_yaml(candidate)
            return candidate.resolve()

    raise FileNotFoundError(
        "YOLO data.yaml not found.\n"
        "Set VISIONIQ_YOLO_DATA or add to evaluation_config.json:\n"
        '  {"yolo": {"data_yaml": "path/to/data.yaml"}}\n'
        f"Searched: {[str(c) for c in candidates]}"
    )


def _has_class_subdirs(path: Path) -> bool:
    """True if path contains at least one subfolder with images."""
    if not path.is_dir():
        return False
    for child in path.iterdir():
        if child.is_dir() and any(_iter_images(child)):
            return True
    return False


def _iter_images(directory: Path):
    """Yield image paths under directory (non-recursive)."""
    for item in directory.iterdir():
        if item.is_file() and item.suffix.lower() in IMAGE_EXTENSIONS:
            yield item


def _validate_image_dir(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Validation directory does not exist: {path}")
    if not path.is_dir():
        raise NotADirectoryError(f"Validation path is not a directory: {path}")
    if not _has_class_subdirs(path):
        raise FileNotFoundError(
            f"No class subfolders with images found in: {path}\n"
            "Expected: validation/<class_name>/*.jpg"
        )


def _validate_data_yaml(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"data.yaml does not exist: {path}")
    if path.suffix.lower() not in {".yaml", ".yml"}:
        raise ValueError(f"Expected .yaml file: {path}")


def collect_validation_samples(val_dir: Path) -> list[tuple[Path, str]]:
    """
    Collect (image_path, true_label) pairs from class-folder layout.
    true_label is 'fracture' or 'none'.
    """
    samples: list[tuple[Path, str]] = []
    for class_dir in sorted(val_dir.iterdir()):
        if not class_dir.is_dir():
            continue
        try:
            label = normalize_folder_label(class_dir.name)
        except ValueError as exc:
            print(f"  [skip] {class_dir.name}: {exc}")
            continue
        for image_path in _iter_images(class_dir):
            samples.append((image_path, label))
        # Also search one level deeper (e.g. val/hand/fracture/)
        for sub in class_dir.iterdir():
            if sub.is_dir():
                try:
                    sub_label = normalize_folder_label(sub.name)
                except ValueError:
                    continue
                for image_path in _iter_images(sub):
                    samples.append((image_path, sub_label))

    if not samples:
        raise RuntimeError(f"No validation images found under {val_dir}")
    return samples


def binary_metrics(y_true: list[str], y_pred: list[str]) -> dict[str, float]:
    """
    Compute binary classification metrics (positive class = 'fracture').
    """
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length")

    labels = ["none", "fracture"]
    label_to_idx = {label: i for i, label in enumerate(labels)}

    yt = np.array([label_to_idx.get(y, 0) for y in y_true])
    yp = np.array([label_to_idx.get(y, 0) for y in y_pred])

    tp = int(np.sum((yt == 1) & (yp == 1)))
    tn = int(np.sum((yt == 0) & (yp == 0)))
    fp = int(np.sum((yt == 0) & (yp == 1)))
    fn = int(np.sum((yt == 1) & (yp == 0)))

    total = len(yt)
    accuracy = (tp + tn) / total if total else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "specificity": specificity,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "support": total,
    }


def confusion_matrix_2x2(y_true: list[str], y_pred: list[str]) -> np.ndarray:
    """Return 2x2 confusion matrix [none, fracture] x [none, fracture]."""
    labels = ["none", "fracture"]
    idx = {label: i for i, label in enumerate(labels)}
    cm = np.zeros((2, 2), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[idx.get(t, 0), idx.get(p, 0)] += 1
    return cm


def print_metrics_table(title: str, metrics: dict[str, Any]) -> None:
    """Pretty-print metrics to terminal."""
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)
    for key, value in metrics.items():
        if key in {"tp", "tn", "fp", "fn", "support"}:
            print(f"  {key:18s}: {value}")
        elif isinstance(value, float):
            print(f"  {key:18s}: {value:.4f} ({value * 100:.2f}%)")
        else:
            print(f"  {key:18s}: {value}")
    print("=" * 60 + "\n")


def apply_plot_style():
    """Consistent matplotlib styling for evaluation plots."""
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "figure.figsize": (8, 6),
            "figure.dpi": 120,
            "font.family": "sans-serif",
            "font.size": 11,
            "axes.titlesize": 14,
            "axes.labelsize": 12,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
