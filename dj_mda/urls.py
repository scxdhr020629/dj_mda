"""
URL configuration for dj_mda project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/4.2/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""

from django.contrib import admin
from django.urls import path, include

from dj_api import views

urlpatterns = [
    path("admin/", admin.site.urls),
    # path("hello", views.hello),
    path("dj_api/", include("dj_api.urls")),
    path("dj_api/email/", include("dj_email.urls")),

]


from django.conf import settings
from django.conf.urls.static import static


# 添加以下代码以提供媒体文件的访问
if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
