from django import forms

from embedding.data import DATA_REF
from embedding.embedder import EMBEDDERS_REF
from embedding.reducer import REDUCERS_REF

label_texts_before_choices = [
        "Data",
	"Embedder",
	"Reducer",
	]

class ChoiceForm(forms.Form):
    data_choice = forms.fields.ChoiceField(
        choices = (
            (k, k) for k in DATA_REF.keys()
        ),
        required=True,
        widget=forms.widgets.Select(attrs={"class": "form-control"}) 
    )

    embedder_choice = forms.fields.ChoiceField(
        choices = (
            (k, k) for k in EMBEDDERS_REF.keys()
        ),
        required=True,
        widget=forms.widgets.Select(attrs={"class": "form-control"}) 
    )

    reducer_choice = forms.fields.ChoiceField(
        choices = (
            (k, k) for k in REDUCERS_REF.keys()
        ),
        required=True,
        widget=forms.widgets.Select(attrs={"class": "form-control"}) 
    )
