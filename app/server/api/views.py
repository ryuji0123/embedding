# -*- coding: utf-8 -*-
"""Django views"""

import plotly.figure_factory as ff
from django.shortcuts import render
from django.http import HttpResponse
from plotly.offline import plot
from rest_framework.views import APIView
from rest_framework.settings import api_settings
from rest_framework_csv import renderers as r
from django.http.response import JsonResponse
from django.core.handlers.wsgi import WSGIRequest

from app.server.api.forms import ChoiceForm, label_texts_before_choices
from embedding.api import get_embedding


def index(request: WSGIRequest) -> JsonResponse:
    """Index view
    
    View function for index page.

    Args:
        request (WSGIRequest): API request.
    Returns:
        JsonResponse: Embedding results. DataFrame-like dictionary.
            ex. {"col0": {"0": -89.04664188087918, "1": -5.223673137616546, ...}, {"col1": "0": 4.444561997081812, "1": 1.1147116170115332, ...}}
    """

    data = request.GET.get(key="data", default="pokemon")
    embedder = request.GET.get(key="embedder", default="isomap")
    reducer = request.GET.get(key="reducer", default="ica")

    embedding = get_embedding(
           data_key=data,
           embedder_key=embedder
           )

    return JsonResponse(embedding.to_dict())


class EmbeddingAPIView(APIView):
    renderer_classes = (r.CSVRenderer, ) + tuple(api_settings.DEFAULT_RENDERER_CLASSES)

