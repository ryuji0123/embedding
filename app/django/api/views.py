import plotly.figure_factory as ff

from django.shortcuts import render
from django.http import HttpResponse
from plotly.offline import plot

from app.django.api.forms import ChoiceForm, label_texts_before_choices
from embedding.api import getFigure

# Create your views here.
def index(request):
    """
    View function for index page,
    """
    fig = getFigure(
           data_key=request.POST.get("data_choice", "pokemon"),
           embedder_key=request.POST.get("embedder_choice", "isomap"),
           reducer_key=request.POST.get("reducer_choice", "ica"),
           )

    forms = ChoiceForm(initial={
        "data_choice": request.POST.get("data_choice", ""),
        "embedder_choice": request.POST.get("embedder_choice", ""),
        "reducer_choice": request.POST.get("reducer_choice", ""),
        })
    

    plot_fig = plot(fig, output_type='div', include_plotlyjs=False)
    context = {
            "zipped_label_texts_and_forms": zip(label_texts_before_choices, forms),
            "plot_fig": plot_fig,
            }

    return render(request, 'api/index.html', context)
