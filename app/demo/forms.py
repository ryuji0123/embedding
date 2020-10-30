from django import forms

from embedding.embedder import embedders_ref
from embedding.reducer import reducers_ref

label_texts_before_choices = [
	"Embedder",
	"Reducer",
	]

class ChoiceForm(forms.Form):
    embedder_choice = forms.fields.ChoiceField(
        choices = (
            (k, k) for k in embedders_ref.keys()
        ),
        required=True,
        widget=forms.widgets.Select(attrs={"class": "form-control"}) 
    )

    reducer_choice = forms.fields.ChoiceField(
        choices = (
            (k, k) for k in reducers_ref.keys()
        ),
        required=True,
        widget=forms.widgets.Select(attrs={"class": "form-control"}) 
    )
