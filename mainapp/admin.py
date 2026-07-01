from django.contrib import admin

from .models import Prediction


@admin.register(Prediction)
class PredictionAdmin(admin.ModelAdmin):
    list_display = ("user", "result_text", "confidence", "created_at")
    list_filter = ("result_text", "created_at")
    search_fields = ("user__username", "result_text")
