from django.contrib.auth.views import LogoutView
from django.urls import path

from . import views

urlpatterns = [
    path("", views.home, name="home"),
    path("upload/", views.upload_image, name="upload"),
    path("dashboard/", views.dashboard, name="dashboard"),
    path("about/", views.about, name="about"),
    path("contact/", views.contact, name="contact"),
    path("history/", views.history, name="history"),
    path("profile/", views.profile, name="profile"),
    path("login/", views.login_view, name="login"),
    path("signup/", views.signup_view, name="signup"),
    path("logout/", LogoutView.as_view(), name="logout"),
]
