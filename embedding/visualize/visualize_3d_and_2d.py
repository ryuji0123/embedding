import plotly.express as px
import plotly.graph_objects as go
import numpy as np

from embedding.data import chooseData
from embedding.embedder import TSNEEmbedder
from embedding.reducer import PCAReducer


# Plane equation f(x, y, z) = ax + by + cz = d
# Set to output z = (d - ax - by) / c
def plane_z(n_vec, n_vec_src, X, Y):
    a = n_vec[0]
    b = n_vec[1]
    c = n_vec[2]
    d = n_vec @ n_vec_src
    return (d - a * X - b * Y) / c


def visualize_3d(dataframe, fig3d=None):
    # Plot data points
    fig = px.scatter_3d(
        dataframe,
        x="0",
        y="1",
        z="2",
        labels={"0": "dim 1", "1": "dim 2", "2": "dim 3"},
        opacity=1,
    )
    if fig3d is None:
        fig3d = fig
    else:
        fig3d.add_trace(fig)
    return fig3d


def add_projection_plane_in_3d(embedder, reducer, fig3d=None):
    max_em_0 = max(abs(embedder.em["0"]))
    max_em_1 = max(abs(embedder.em["1"]))
    max_em_2 = max(abs(embedder.em["2"]))
    # Plot Projection Plane
    X = np.outer(
        np.linspace(-max_em_0, max_em_0, 2),
        np.ones(2),
    )
    Y = np.outer(
        np.linspace(-max_em_1, max_em_1, 2),
        np.ones(2),
    ).T
    Z = plane_z(reducer.n_vec, reducer.n_vec_src, X, Y)

    fig = go.Figure(data=[go.Surface(x=X, y=Y, z=Z, opacity=0.5)])

    # Plot cone representiing Basis of Projection Plane
    fig.add_trace(
        go.Cone(
            x=reducer.cmp[:, 0] * max_em_0 / 20,
            y=reducer.cmp[:, 1] * max_em_0 / 20,
            z=reducer.cmp[:, 2] * max_em_0 / 20,
            u=reducer.cmp[:, 0] * max_em_0 / 10,
            v=reducer.cmp[:, 1] * max_em_1 / 10,
            w=reducer.cmp[:, 2] * max_em_2 / 10,
        )
    )
    if fig3d is None:
        fig3d = fig
    else:
        for goel in fig.data:
            fig3d.add_trace(goel)
    return fig3d


def visualize_2d(dataframe, fig2d=None):
    # Create 2d figure After Dinmensionality Reduction
    fig = px.scatter(
        dataframe, x="0", y="1", labels={"0": "component 1", "1": "component 2"}
    )
    fig.update_traces(
        marker=dict(size=2, line=dict(width=2, color="DarkSlateGrey")),
        selector=dict(mode="markers"),
    )
    if fig2d is None:
        fig2d = fig
    else:
        fig2d.add_trace(fig)
    return fig2d


# http://comprna.upf.edu/courses/Master_MAT/3_Optimization/U9_Hyperplanes.pdf
def visualize_3d_to_2d_projection(embedder, reducer):
    # Plot 3d figure of embedded data points AND Projection Plane
    fig3d = visualize_3d(embedder.em)
    fig3d = add_projection_plane_in_3d(embedder, reducer, fig3d=fig3d)
    fig3d.update_traces(
        marker=dict(size=2, line=dict(width=2, color="DarkSlateGrey")),
        selector=dict(mode="markers"),
    )
    fig3d.update_layout(title_text="post embedding")
    fig3d.show()

    # Create 2d figure After Dinmensionality Reduction
    fig2d = visualize_2d(reducer.rd)
    fig2d.update_layout(title_text="After Diminsionality Reduction")
    fig2d.show()


if __name__ == "__main__":

    which_data = "artificial"
    # which_data = "pokemon"
    tsne_embedder = TSNEEmbedder(chooseData(which_data))
    tsne_embedder.embed(dim=3, use_cache=True)
    pca_reducer = PCAReducer(chooseData(which_data), tsne_embedder)
    pca_reducer.reduce(dim=2)

    indices = tsne_embedder.em.query('`0` < `1`').index
    print(indices)
    tsne_embedder.filter(indices)
    print(tsne_embedder.fem)
    

    visualize_3d_to_2d_projection(tsne_embedder, pca_reducer)
