import math
import numpy as np


class ParentSynthesizer:

    def __init__(self):
        pass

    def normalize(self, points):
        for d in range(points.shape[1]):
            mid_point = (points[:, d].min() + points[:, d].max()) / 2
            points[:, d] -= mid_point
            points[:, d] = points[:, d] / np.abs(points[:, d]).max()

        return points

    def getBasicClusters(self, n_dim, n_points):
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

    def synthesizeData(self, n_dim, n_cluster, n_points):
        n_points_per_cluster = [math.ceil(n_points / n_cluster)] * n_cluster
        if n_points % n_cluster != 0:
            n_points_per_cluster[n_cluster - 1] = n_points % n_points_per_cluster[0]

        all_points = []
        clusters = []
        colors = []

        # generate each cluster
        for i in range(n_cluster):
            cluster = self.getBasicClusters(n_dim, n_points_per_cluster[i])
            clusters += [cluster]
            colors += [i] * n_points_per_cluster[i]

        all_points = np.concatenate(clusters, axis=0)

        # Column names for DataFrame
        cols = ["col{}".format(i) for i in range(n_dim)]

        return all_points, cols, colors
