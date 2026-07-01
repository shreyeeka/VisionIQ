import logging

from django.contrib import messages
from django.contrib.auth import login, update_session_auth_hash
from django.contrib.auth.decorators import login_required
from django.contrib.auth.forms import PasswordChangeForm
from django.core.paginator import Paginator
from django.db.models import Avg, Q
from django.shortcuts import redirect, render

from .ai.yolo_predict import detect_fracture
from .forms import CustomLoginForm, CustomSignupForm, ProfileUpdateForm, UploadImageForm
from .models import Prediction

logger = logging.getLogger(__name__)


def home(request):
    total_scans = Prediction.objects.count()
    total_users = Prediction.objects.values("user").distinct().count()
    positive_cases = Prediction.objects.filter(result_text__icontains="fracture").count()
    stats = {
        "total_scans": total_scans,
        "total_users": total_users,
        "positive_cases": positive_cases,
    }
    return render(request, "home.html", {"stats": stats})


@login_required
def upload_image(request):
    prediction = None
    form = UploadImageForm()

    if request.method == "POST":
        form = UploadImageForm(request.POST, request.FILES)
        if form.is_valid():
            prediction = form.save(commit=False)
            prediction.user = request.user

            try:
                prediction.save()
                ai_result = detect_fracture(prediction.uploaded_image.path)
                message = ai_result.get("message", "No result")
                prediction.result_text = message
                prediction.confidence = float(ai_result.get("confidence", 0.0))
                result_image_path = ai_result.get("image")

                if result_image_path:
                    normalized = result_image_path.replace("\\", "/")
                    if normalized.startswith("/media/"):
                        normalized = normalized.replace("/media/", "", 1)
                    prediction.result_image = normalized

                prediction.save(update_fields=["result_text", "confidence", "result_image"])
                messages.success(request, "Analysis completed successfully.")
            except Exception as exc:
                logger.exception("AI inference failed: %s", exc)
                prediction.result_text = "Analysis failed. Please try again."
                prediction.confidence = 0.0
                prediction.save(update_fields=["result_text", "confidence"])
                messages.error(request, "AI inference failed. Please try again later.")
        else:
            messages.error(request, "Please provide a valid image file.")

    context = {"form": form, "prediction": prediction}
    return render(request, "upload.html", context)


def dashboard(request):
    if request.user.is_authenticated:
        qs = Prediction.objects.filter(user=request.user)
    else:
        qs = Prediction.objects.none()

    predictions = qs[:50]
    total_scans = qs.count()
    fractured_count = sum(1 for p in qs if p.is_fractured)
    normal_count = total_scans - fractured_count
    avg = qs.aggregate(avg=Avg("confidence"))["avg"]
    avg_confidence = round(avg, 1) if avg is not None else None

    return render(
        request,
        "dashboard.html",
        {
            "predictions": predictions,
            "total_scans": total_scans,
            "fractured_count": fractured_count,
            "normal_count": normal_count,
            "avg_confidence": avg_confidence,
        },
    )

def about(request):
    return render(request, "about.html")

def contact(request):
    return render(request, "contact.html")

def history(request):
    if not request.user.is_authenticated:
        return render(request, "history.html", {"login_required_message": True})

    search_query = request.GET.get("q", "").strip()
    predictions = Prediction.objects.filter(user=request.user)

    if search_query:
        filters = Q(result_text__icontains=search_query) | Q(created_at__icontains=search_query)
        try:
            filters |= Q(confidence=float(search_query))
        except ValueError:
            pass
        predictions = predictions.filter(filters)

    paginator = Paginator(predictions, 9)
    page_obj = paginator.get_page(request.GET.get("page"))

    return render(
        request,
        "history.html",
        {"page_obj": page_obj, "search_query": search_query},
    )


@login_required
def profile(request):
    profile_form = ProfileUpdateForm(instance=request.user)
    password_form = PasswordChangeForm(request.user)

    if request.method == "POST":
        action = request.POST.get("action")

        if action == "update_profile":
            profile_form = ProfileUpdateForm(request.POST, instance=request.user)
            if profile_form.is_valid():
                profile_form.save()
                messages.success(request, "Profile updated successfully.")
                return redirect("profile")
            messages.error(request, "Please fix profile form errors.")

        elif action == "update_password":
            password_form = PasswordChangeForm(request.user, request.POST)
            if password_form.is_valid():
                user = password_form.save()
                update_session_auth_hash(request, user)
                messages.success(request, "Password changed successfully.")
                return redirect("profile")
            messages.error(request, "Please fix password form errors.")

    user_predictions = Prediction.objects.filter(user=request.user)
    context = {
        "profile_form": profile_form,
        "password_form": password_form,
        "total_scans": user_predictions.count(),
        "detections": user_predictions.filter(result_text__icontains="fracture").count(),
    }
    return render(request, "profile.html", context)


def login_view(request):
    if request.user.is_authenticated:
        return redirect("home")

    form = CustomLoginForm(request, request.POST or None)
    if request.method == "POST":
        if form.is_valid():
            login(request, form.get_user())
            messages.success(request, "Welcome back to VisionIQ.")
            next_url = request.GET.get("next")
            return redirect(next_url or "home")
        messages.error(request, "Invalid username/email or password.")

    return render(request, "login.html", {"form": form})


def signup_view(request):
    if request.user.is_authenticated:
        return redirect("home")

    form = CustomSignupForm(request.POST or None)
    if request.method == "POST":
        if form.is_valid():
            user = form.save()
            login(request, user)
            messages.success(request, "Your account has been created.")
            return redirect("home")
        messages.error(request, "Please fix the signup form errors.")

    return render(request, "signup.html", {"form": form})
