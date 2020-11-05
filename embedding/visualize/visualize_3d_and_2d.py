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
        x="col0",
        y="col1",
        z="col2",
        labels={"col0": "dim 1", "col1": "dim 2", "col2": "dim 3"},
        opacity=1,
    )
    if fig3d is None:
        fig3d = fig
    else:
        fig3d.add_trace(fig)
    return fig3d


def add_projection_plane_in_3d(embedder, reducer, fig3d=None):
    max_em_0 = max(abs(embedder.em["col0"]))
    max_em_1 = max(abs(embedder.em["col1"]))
    max_em_2 = max(abs(embedder.em["col2"]))
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
            x=[0],
            y=[0],
            z=[0],
            u=[reducer.n_vec[0] * max_em_0 / 10],
            v=[reducer.n_vec[1] * max_em_1 / 10],
            w=[reducer.n_vec[2] * max_em_2 / 10],
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
        dataframe,
        x="col0",
        y="col1",
        labels={"col0": "component 1", "col1": "component 2"},
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

    indices = tsne_embedder.em.query("`col0` < `col1`").index
    print(indices)
    fem = tsne_embedder.em.loc[indices, :]
    print(fem)

    pca_reducer.calcFilteredRds(fem, 10)

    visualize_3d_to_2d_projection(tsne_embedder, pca_reducer)
    new_pca_reducer = pca_reducer
    new_pca_reducer.rd = pca_reducer.rds[1]
    new_pca_reducer.n_vec = pca_reducer.n_vecs[1]
    new_pca_reducer.cmp = pca_reducer.cmps_oth[1]
    visualize_3d_to_2d_projection(tsne_embedder, new_pca_reducer)
