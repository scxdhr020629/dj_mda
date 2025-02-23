from django.urls import path

from dj_map import views

urlpatterns = [
    path('hello/',views.hello),

]