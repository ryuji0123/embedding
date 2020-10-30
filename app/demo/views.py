import plotly.figure_factory as ff

from django.shortcuts import render
from django.http import HttpResponse
from plotly.offline import plot

from app.demo.forms import ChoiceForm, label_texts_before_choices

# Create your views here.
def index(request):
    """
    View function for idnex page,
    """
    df = [
            dict(Task="Job A", Start='2009-01-01', Finish='2009-02-28'),
            dict(Task="Job B", Start='2009-03-05', Finish='2009-04-15'),
            dict(Task="Job C", Start='2009-02-20', Finish='2009-05-30')
            ]

    fig = ff.create_gantt(df)
    plot_fig = plot(fig, output_type='div', include_plotlyjs=False)
    forms = ChoiceForm()
    context = {
            "forms": forms,
            "label_texts": label_texts_before_choices,
            "zipped_label_texts_and_forms": zip(label_texts_before_choices, forms),
            "plot_fig": plot_fig,
            }

    return render(request, 'demo/index.html', context)
