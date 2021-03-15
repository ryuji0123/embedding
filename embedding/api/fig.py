import plotly.express as px

from embedding.data import chooseData
from embedding.embedder import chooseEmbedder
from embedding.reducer import chooseReducer


def getFigure(data_key, embedder_key, reducer_key):
    sc_data = chooseData(data_key)
    embedder = chooseEmbedder(embedder_key, sc_data)
    embedder.embed()
    reducer = chooseReducer(reducer_key, sc_data, embedder)
    reducer.reduce()
    return px.scatter(reducer.rd, x='col0', y='col1', color=sc_data.color, title=f'{data_key} | {reducer.class_key}')
