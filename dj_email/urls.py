from django.urls import path
from dj_email import views
urlpatterns = [
    path('hello/',views.hello),
    path('send/',views.send_email),
    path('test_openai/',views.test_openai),
]