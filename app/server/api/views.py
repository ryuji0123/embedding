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

# TODO: Remove code for GUI
# for GUI
from embedding.api import getFigure

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

    # GUI: start
    # fig = getFigure(
    #        data_key=request.POST.get("data_choice", "pokemon"),
    #        embedder_key=request.POST.get("embedder_choice", "isomap"),
    #        reducer_key=request.POST.get("reducer_choice", "ica"),
    #        )

    # forms = ChoiceForm(initial={
    #     "data_choice": request.POST.get("data_choice", ""),
    #     "embedder_choice": request.POST.get("embedder_choice", ""),
    #     "reducer_choice": request.POST.get("reducer_choice", ""),
    #     })
    

    # plot_fig = plot(fig, output_type='div', include_plotlyjs=False)
    # context = {
    #         "zipped_label_texts_and_forms": zip(label_texts_before_choices, forms),
    #         "plot_fig": plot_fig,
    #         }

    # return render(request, 'api/index.html', context)
    # GUI: end


class EmbeddingAPIView(APIView):
    renderer_classes = (r.CSVRenderer, ) + tuple(api_settings.DEFAULT_RENDERER_CLASSES)

