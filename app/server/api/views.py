import plotly.figure_factory as ff

from django.shortcuts import render
from django.http import HttpResponse
from plotly.offline import plot
from rest_framework.views import APIView
from rest_framework.settings import api_settings
from rest_framework_csv import renderers as r
from django.http.response import JsonResponse

from app.server.api.forms import ChoiceForm, label_texts_before_choices
from embedding.api import getEmbedding

# Create your views here.
def index(request):
    """
    View function for index page,
    """

    data = request.GET.get(key="data", default="pokemon")
    embedder = request.GET.get(key="embedder", default="isomap")
    reducer = request.GET.get(key="reducer", default="ica")

    embedding = getEmbedding(
           data_key=data,
           embedder_key=embedder
           )
    
    return JsonResponse(embedding.to_dict())


class EmbeddingAPIView(APIView):
    renderer_classes = (r.CSVRenderer, ) + tuple(api_settings.DEFAULT_RENDERER_CLASSES)

