from django.urls import path, include
from django.conf import settings
from django.conf.urls.static import static

from . import views


app_name = 'transcriber'

urlpatterns = [
    # two paths: with or without given image
    path('', views.index, name='index'),
    path('download/', views.download_file),
    path('download_zip/', views.download_zip)
] + static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)