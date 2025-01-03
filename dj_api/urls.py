from django.urls import path

from dj_api import views

urlpatterns = [
    path('hello/',views.hello),
    path('get_user/',views.get_user),
    path('add_user/',views.add_user),
    path('get_rnas/',views.get_rnas),
    path('test-redis/', views.test_redis_cache, name='test_redis_cache'),
]