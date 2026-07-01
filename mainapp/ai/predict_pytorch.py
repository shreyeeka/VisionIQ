from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
CHECKPOINT_PATH = MODELS_DIR / "fracture_classifier.pt"
LABEL_MAP_PATH = MODELS_DIR / "label_maps.json"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

with LABEL_MAP_PATH.open("r", encoding="utf-8") as f:
    label_maps = json.load(f)


class MultiTaskResNet(nn.Module):
    def __init__(self, num_locations: int, num_fracture_types: int) -> None:
        super().__init__()
        backbone = models.resnet50(weights=None)
        in_features = backbone.fc.in_features
        backbone.fc = nn.Identity()
        self.backbone = backbone
        self.location_head = nn.Linear(in_features, num_locations)
        self.type_head = nn.Linear(in_features, num_fracture_types)
        self.fracture_head = nn.Linear(in_features, 1)
        self.severity_head = nn.Linear(in_features, 1)
        self.risk_head = nn.Linear(in_features, 1)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        features = self.backbone(x)
        return {
            "location_logits": self.location_head(features),
            "type_logits": self.type_head(features),
            "fracture_logit": self.fracture_head(features).squeeze(-1),
            "severity_logit": self.severity_head(features).squeeze(-1),
            "risk_logit": self.risk_head(features).squeeze(-1),
        }


def _extract_state_dict(checkpoint: Any) -> dict[str, torch.Tensor]:
    if isinstance(checkpoint, dict):
        if "model_state" in checkpoint:
            return checkpoint["model_state"]
        if "state_dict" in checkpoint:
            return checkpoint["state_dict"]
    if isinstance(checkpoint, nn.Module):
        return checkpoint.state_dict()
    raise RuntimeError("Unsupported classifier checkpoint format.")


def _clean_state_dict_keys(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    cleaned = {}
    for key, value in state_dict.items():
        if key.startswith("module."):
            cleaned[key[7:]] = value
        else:
            cleaned[key] = value
    return cleaned


def _load_classifier_model() -> nn.Module:
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
    state_dict = _clean_state_dict_keys(_extract_state_dict(checkpoint))

    model = MultiTaskResNet(
        num_locations=len(label_maps["location"]),
        num_fracture_types=len(label_maps["fracture_type"]),
    )
    model.load_state_dict(state_dict, strict=True)
    model = model.to(device)
    model.eval()
    return model


model = _load_classifier_model()

transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ]
)


def predict_fracture(image_path: str) -> tuple[str, float]:
    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(image_tensor)

    fracture_prob = torch.sigmoid(outputs["fracture_logit"]).item()
    predicted_idx = int(torch.argmax(outputs["type_logits"], dim=1).item())

    fracture_labels = label_maps.get("fracture_type", ["fracture_unspecified", "none"])
    predicted_label = fracture_labels[predicted_idx] if predicted_idx < len(fracture_labels) else "fracture_unspecified"

    if fracture_prob <= 0.5:
        predicted_label = "none"

    confidence = fracture_prob * 100
    return predicted_label, confidence