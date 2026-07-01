import os

from django.contrib.auth.models import User
from django.db import models


class Prediction(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name="predictions")
    uploaded_image = models.ImageField(upload_to="uploads/")
    result_image = models.ImageField(upload_to="predictions/", blank=True, null=True)
    confidence = models.FloatField(default=0.0)
    result_text = models.CharField(max_length=255)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-created_at"]

    def __str__(self) -> str:
        return f"{self.user.username} - {self.result_text} ({self.confidence:.1f}%)"

    @property
    def clean_message(self) -> str:
        text = self.result_text or ""
        if " | Type:" in text:
            text = text.split(" | Type:")[0]
        return text.strip()

    @property
    def is_fractured(self) -> bool:
        text = self.clean_message.lower()
        if "no fracture" in text:
            return False
        return "fracture" in text

    @property
    def display_result(self) -> str:
        if self.is_fractured:
            return "Fracture Detected"
        return "No Fracture Detected"

    @property
    def filename(self) -> str:
        if self.uploaded_image:
            return os.path.basename(self.uploaded_image.name)
        return "xray.jpg"
