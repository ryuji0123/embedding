from django import forms

from embedding.embedder import EMBEDDERS_REF
from embedding.reducer import REDUCERS_REF

label_texts_before_choices = [
	"Embedder",
	"Reducer",
	]

class ChoiceForm(forms.Form):
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
