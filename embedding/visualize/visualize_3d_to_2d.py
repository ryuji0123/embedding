import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import pandas as pd

from embedding.data import chooseData
from embedding.embedder import *
from embedding.reducer import *

import os

# plane equation f(x, y, z) = ax + by + cz = d
# set to output z = (d - ax - by) / c
def plane_z(n_vec, n_vec_src, X, Y):
    a = n_vec[0]
    b = n_vec[1]
    c = n_vec[2]
    d = n_vec @ n_vec_src
    return (d - a * X - b * Y) / c


def visualize_3d_to_2d_projection(embedder, reducer):
    fig3d = px.scatter_3d(
        embedder.em,
        x="0",
        y="1",
        z="2",
        labels={"0": "dim 1", "1": "dim 2", "2": "dim 3"},
    )

    X = np.outer(
        np.linspace(min(embedder.em["0"]), max(embedder.em["0"]), 2),
        np.ones(2),
    )
    Y = np.outer(
        np.linspace(min(embedder.em["1"]), max(embedder.em["1"]), 2),
        np.ones(2),
    ).T

    Z = plane_z(reducer.n_vec, reducer.n_vec_src, X, Y)

    fig3d.add_trace(go.Surface(x=X, y=Y, z=Z, opacity=0.5))

    fig3d.update_traces(
        marker=dict(size=2, line=dict(width=2, color="DarkSlateGrey")),
        selector=dict(mode="markers"),
    )
    fig3d.update_layout(
        title_text='post embedding'
    )
    fig3d.show()

    fig2d = px.scatter(
        reducer.rd, x=0, y=1, labels={"0": "component 1", "1": "component 2"}
    )

    fig2d.update_traces(
        marker=dict(size=2, line=dict(width=2, color="DarkSlateGrey")),
        selector=dict(mode="markers"),
    )
    fig2d.update_layout(
        title_text='post diminsionality reduction'
    )
    fig2d.show()


def main():

    # /workspace/result/ が既ににあると計算されない
    os.remove("/workspace/results/t-sne_embedder_and_pca_reducer_pokemon.csv")

    which_data = "pokemon"
    tsne_embedder = TSNEEmbedder(chooseData(which_data))
    tsne_embedder.embed(dim=3)
    pca_reducer = PCAReducer(chooseData(which_data), tsne_embedder)
    pca_reducer.reduce(dim=2)

    tsne_embedder.em = pd.DataFrame(data=tsne_embedder.em)
    pca_reducer.rd = pd.DataFrame(data=pca_reducer.rd)

    visualize_3d_to_2d_projection(tsne_embedder, pca_reducer)


if __name__ == "__main__":
    main()
