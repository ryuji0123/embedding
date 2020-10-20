import plotly.express as px

from embedding.data import chooseData
from embedding.embedder import *
from embedding.reducer import *


def main():
    which_data = "pokemon"
    tsne_embedder = TSNEEmbedder(chooseData(which_data))
    tsne_embedder.embed(dim=3)
    pca_reducer = PCAReducer(chooseData(which_data), tsne_embedder)
    pca_reducer.reduce(dim=2)

    fig = px.scatter_3d(
        tsne_embedder.em, x='0', y='1', z='2',
        labels={'0': 'dim 1', '1': 'dim 2', '2': 'dim 3'}
    )
    fig.update_traces(marker=dict(size=2, line=dict(width=2, color='DarkSlateGrey')),
                       selector=dict(mode='markers')
                       )
    fig.show()

    fig2 = px.scatter(
        tsne_embedder.em, x='0', y='1',
        labels={'0': 'component 1', '1': 'component 2'}
    )
    fig2.update_traces(marker=dict(size=2, line=dict(width=2, color='DarkSlateGrey')),
                       selector=dict(mode='markers')
                       )
    fig2.show()


if __name__ == "__main__":
    main()
