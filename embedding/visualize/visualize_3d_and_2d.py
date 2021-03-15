import plotly.express as px
import plotly.graph_objects as go
import numpy as np

from embedding.data import chooseData
from embedding.embedder import chooseEmbedder
from embedding.reducer import chooseReducer


# Plane equation f(x, y, z) = ax + by + cz = d
# Set to output z = (d - ax - by) / c
def plane_z(normal_vectors, initial_points_of_normal_vectors, X, Y):
    a = normal_vectors[0]
    b = normal_vectors[1]
    c = normal_vectors[2]
    d = normal_vectors @ initial_points_of_normal_vectors
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
    Z = plane_z(reducer.normal_vectors, reducer.initial_points_of_normal_vectors, X, Y)

    fig = go.Figure(data=[go.Surface(x=X, y=Y, z=Z, opacity=0.5)])

    # Plot cone representiing Basis of Projection Plane
    fig.add_trace(
        go.Cone(
            x=[0],
            y=[0],
            z=[0],
            u=[reducer.normal_vectors[0] * max_em_0 / 10],
            v=[reducer.normal_vectors[1] * max_em_1 / 10],
            w=[reducer.normal_vectors[2] * max_em_2 / 10],
        )
    )

    normal_vectors_line = (
        np.array([max_em_0, max_em_1, max_em_2]) * reducer.normal_vectors
    )
    print(normal_vectors_line)
    fig.add_trace(
        go.Scatter3d(
            x=[-normal_vectors_line[0], normal_vectors_line[0]],
            y=[-normal_vectors_line[1], normal_vectors_line[1]],
            z=[-normal_vectors_line[2], normal_vectors_line[2]],
            mode="lines",
            line=dict(color="darkblue", width=2),
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
    fig3d.update_layout(
        title_text="post embedding",
        scene=dict(
            xaxis=dict(
                nticks=4,
                range=[-20, 20],
                # range=[min(embedder.em["col0"]), max(embedder.em["col0"])],
            ),
            yaxis=dict(
                nticks=4,
                range=[-20, 20],
                # range=[min(embedder.em["col1"]), max(embedder.em["col1"])],
            ),
            zaxis=dict(
                nticks=4,
                range=[-20, 20],
                # range=[min(embedder.em["col2"]), max(embedder.em["col2"])],
            ),
        ),
        width=700,
        margin=dict(r=20, l=10, b=10, t=10),
    )
    fig3d.show()

    # Create 2d figure After Dinmensionality Reduction
    fig2d = visualize_2d(reducer.rd)
    fig2d.update_layout(title_text="After Diminsionality Reduction")
    fig2d.show()


if __name__ == "__main__":

    # Get 3d animations
    # Run these in Jupyter or something
    which_data = "basic_cluster"
    embedder = chooseEmbedder("t_sne", (chooseData(which_data)))
    embedder.embed(dim=3, use_cache=True)
    reducer = chooseReducer("pca", chooseData(which_data), embedder)
    reducer.reduce(dim=2, save_rd=False)

    # visualize_3d_to_2d_projection(embedder, reducer)

    # Double filter
    query1 = "col1<0 & col2>0 & col2>=0.5"
    query0 = "*"

    # For Dimensionality Reduction to 2D
    reducer.setRds(query0=query0, query1=query1)
    print(reducer.getRdsDf())
    fig2d = px.scatter(
        reducer.getRdsDf(),
        x="col0",
        y="col1",
        labels={"col0": "dim 1", "col1": "dim 2"},
        animation_frame="t",
    )

    fig2d.show()

    # For Dimensionality Reduction to 3D
    reducer.setRds(query0=query0, query1=query1, dim=3)
    print(reducer.getRdsDf())
    fig3d = px.scatter_3d(
        reducer.getRdsDf(),
        x="col0",
        y="col1",
        z="col2",
        labels={"col0": "dim 1", "col1": "dim 2", "col2": "dim 3"},
        animation_frame="t",
        color="query0",
    )
    fig3d.update_traces(
        marker=dict(size=2),
    )

    fig3d.update_layout(
        title_text="post embedding",
        scene=dict(
            xaxis=dict(
                nticks=4,
                range=[-20, 20],
                # range=[min(embedder.em["col0"]), max(embedder.em["col0"])],
            ),
            yaxis=dict(
                nticks=4,
                range=[-20, 20],
                # range=[min(embedder.em["col1"]), max(embedder.em["col1"])],
            ),
            zaxis=dict(
                nticks=4,
                range=[-20, 20],
                # range=[min(embedder.em["col2"]), max(embedder.em["col2"])],
            ),
        ),
        width=700,
        margin=dict(r=20, l=10, b=10, t=10),
    )
    fig3d.show()
