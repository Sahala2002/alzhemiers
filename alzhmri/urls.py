
from django.urls import path
from . import views



urlpatterns = [

    # Home / MRI Prediction
    path(
        "",
        views.home,
        name="home"
    ),

    # About page
    path(
        "about/",
        views.about,
        name="about"
    ),

    # Prediction history
    path(
        "history/",
        views.history,
        name="history"
    ),
   path(
        "result/",
        views.result,
        name="result"
    ),
]

