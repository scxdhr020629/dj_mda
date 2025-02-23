from django.urls import path
from dj_email import views
urlpatterns = [
    path('hello/',views.hello)
]