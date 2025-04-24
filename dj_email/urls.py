from xml.etree.ElementInclude import include

from django.urls import path
from dj_email import views
from rest_framework.routers import DefaultRouter
from .views import KnowledgeGraphViewSet, hello, send_email, test_openai

router = DefaultRouter()
router.register(r'knowledge-graph', KnowledgeGraphViewSet, basename='knowledge-graph')
urlpatterns = [
    path('hello/',views.hello),
    path('send/',views.send_email),
    path('test_openai/',views.test_openai),

]+ router.urls