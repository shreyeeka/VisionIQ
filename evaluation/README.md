# VisionIQ Model Evaluation

Evaluation pipeline for **ResNet50** (classification) and **YOLOv8** (detection).

## Quick Start

1. Copy and edit config:
   ```bash
   copy evaluation_config.example.json evaluation_config.json
   ```

2. Install evaluation dependencies:
   ```bash
   pip install numpy pandas scikit-learn matplotlib seaborn PyYAML
   ```

3. Run evaluations:
   ```bash
   python evaluate_resnet.py
   python evaluate_yolo.py
   python compare_models.py
   ```

## Dataset Setup

### ResNet50 (folder-based)

```
validation/
  fracture_unspecified/   # or fracture/, positive/
    img001.jpg
  none/                   # or normal/, negative/
    img002.jpg
```

Set path via:
- `VISIONIQ_VAL_DIR=C:\path\to\validation`
- or `evaluation_config.json` → `resnet.validation_dir`

### YOLOv8 (data.yaml)

Standard Ultralytics layout:

```yaml
path: C:/path/to/dataset
train: images/train
val: images/val
names:
  0: fracture
```

Set path via:
- `VISIONIQ_YOLO_DATA=C:\path\to\data.yaml`
- or `evaluation_config.json` → `yolo.data_yaml`

## Output Structure

```
evaluation_results/
├── resnet/
│   ├── metrics.csv
│   ├── classification_report.csv
│   ├── predictions.csv
│   ├── confusion_matrix.png
│   ├── precision_recall_curve.png
│   └── metric_plots.png
├── yolo/
│   ├── metrics.csv
│   ├── evaluation_summary.txt
│   ├── precision_recall_curve.png
│   └── metric_plots.png
├── final_report.csv
├── accuracy_f1_comparison.png
├── metric_comparison.png
└── precision_recall_comparison.png
```
