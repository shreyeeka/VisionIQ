from django import forms
from django.contrib.auth import authenticate
from django.contrib.auth.forms import UserCreationForm
from django.contrib.auth.models import User

from .models import Prediction


class UploadImageForm(forms.ModelForm):
    class Meta:
        model = Prediction
        fields = ["uploaded_image"]
        widgets = {
            "uploaded_image": forms.ClearableFileInput(
                attrs={
                    "accept": "image/*",
                    "capture": "environment",
                    "class": "hidden-file-input",
                    "id": "imageInput",
                }
            )
        }

    def clean_uploaded_image(self):
        image = self.cleaned_data.get("uploaded_image")
        if not image:
            raise forms.ValidationError("Please upload an X-ray image.")
        if image.size > 10 * 1024 * 1024:
            raise forms.ValidationError("Image size must be less than 10MB.")
        return image


class CustomSignupForm(UserCreationForm):
    email = forms.EmailField(required=False)

    class Meta:
        model = User
        fields = ["username", "email", "password1", "password2"]


class ProfileUpdateForm(forms.ModelForm):
    class Meta:
        model = User
        fields = ["username", "email"]


class CustomLoginForm(forms.Form):
    identity = forms.CharField(label="Username or Email")
    password = forms.CharField(widget=forms.PasswordInput)

    def __init__(self, request=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.request = request
        self.user_cache = None

    def clean(self):
        cleaned_data = super().clean()
        identity = cleaned_data.get("identity")
        password = cleaned_data.get("password")

        if identity and password:
            username = identity
            if "@" in identity:
                try:
                    username = User.objects.get(email__iexact=identity).username
                except User.DoesNotExist:
                    pass

            self.user_cache = authenticate(self.request, username=username, password=password)
            if self.user_cache is None:
                raise forms.ValidationError("Invalid username/email or password.")

        return cleaned_data

    def get_user(self):
        return self.user_cache
