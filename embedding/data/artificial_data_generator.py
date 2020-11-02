import numpy as np
import plotly.express as px
import math


def generate_cluster(n_dim, n_points):
    # create covariance matrix as positive semi-definite matrix
    covs = (np.random.rand(n_dim)) * 15
    means = (np.random.rand(n_dim) - 0.5) * 100
    points = []

    for i, (mean, cov) in enumerate(zip(means, covs)):
        points += [[np.random.normal(mean, cov) for _ in range(n_points)]]

    points = np.array(points).T

    a = np.random.randn(n_dim, n_dim) - 0.5
    # use QR decomposition to rotate points using randoml rotation matrix
    q, r = np.linalg.qr(a)
    points = points @ q
    return points


def generate_data(n_dim, n_cluster, n_points):
    # Number of points for each cluster
    n_points_per_cluster = [math.ceil(n_points / n_cluster)] * n_cluster
    if n_points % n_cluster != 0:
        n_points_per_cluster[n_cluster - 1] = n_points % n_points_per_cluster[0]

    all_points = []
    clusters = []
    colors = []

    # generate each cluster
    for i in range(n_cluster):
        cluster = generate_cluster(n_dim, n_points_per_cluster[i])
        clusters += [cluster]
        colors += [i] * n_points_per_cluster[i]

    all_points = np.concatenate(clusters, axis=0)

    # Column names for DataFrame
    cols = ["col{}".format(i) for i in range(n_dim)]
    return all_points, cols, colors


if __name__ == "__main__":
    all_points, cols = generate_data(3, 4, 399)
    print("wow")

    fig2 = px.scatter_3d(
        all_points, x=0, y=1, z=2, labels={"0": "PC 1", "1": "PC 2", "2": "PC 3"}
    )
    fig2.update_traces(
        marker=dict(size=2, line=dict(width=2, color="DarkSlateGrey")),
        selector=dict(mode="markers"),
    )
    fig2.show()
