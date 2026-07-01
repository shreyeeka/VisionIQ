from __future__ import annotations

from pathlib import Path
from uuid import uuid4

from django.conf import settings
from ultralytics import YOLO

from .predict_pytorch import predict_fracture

MODELS_DIR = Path(__file__).resolve().parent / "models"
YOLO_MODEL_PATH = MODELS_DIR / "best_final.pt"

yolo_model = YOLO(str(YOLO_MODEL_PATH))


def detect_fracture(image_path: str) -> dict:
    image_file = Path(image_path)
    predictions_dir = Path(settings.MEDIA_ROOT) / "predictions"
    predictions_dir.mkdir(parents=True, exist_ok=True)

    run_name = f"visioniq_{uuid4().hex[:10]}"
    results = yolo_model.predict(
        source=str(image_file),
        conf=0.05,
        imgsz=1024,
        save=True,
        project=str(predictions_dir),
        name=run_name,
        exist_ok=True,
    )

    run_dir = predictions_dir / run_name
    yolo_output = run_dir / image_file.name
    if not yolo_output.exists():
        candidates = list(run_dir.glob("*"))
        if not candidates:
            raise RuntimeError("YOLO inference completed but no output image was produced.")
        yolo_output = candidates[0]

    final_name = f"detected_{image_file.stem}_{uuid4().hex[:8]}{yolo_output.suffix}"
    final_output = predictions_dir / final_name
    yolo_output.replace(final_output)

    # Keep media/predictions tidy by removing run directory if empty.
    if run_dir.exists():
        for child in run_dir.glob("*"):
            if child.exists():
                child.unlink(missing_ok=True)
        run_dir.rmdir()

    fracture_type, confidence = predict_fracture(str(image_file))
    has_detection = len(results[0].boxes) > 0

    if has_detection and fracture_type != "none":
        message = f"Fracture Detected ({fracture_type})"
    elif has_detection:
        message = "Potential fracture area localized"
    else:
        message = "No fracture localized"

    return {
        "image": f"/media/predictions/{final_name}",
        "message": message,
        "confidence": round(confidence, 2),
        "label": fracture_type,
    }