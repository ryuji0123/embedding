from api import views
from django.conf.urls import url, include
from django.urls import path
from rest_framework import routers

from app.server.api.views import EmbeddingAPIView

app_name = 'api'

urlpatterns = [
    path('', views.index, name='index'),
    path('csv', EmbeddingAPIView.as_view(), name='csv')
]
