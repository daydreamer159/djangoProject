
from django.contrib import admin
from django.urls import path
from imgdetec import views


urlpatterns = [
    path('imgtest/', views.imgSave),
]
