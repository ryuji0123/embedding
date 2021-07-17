import numpy as np

from scipy.spatial import distance


def measure_kruskal_stress(actual_points, fitted_points):
    # make lower triangular distance matrices
    actual_distances = np.tril(distance.cdist(actual_points, actual_points, metric='euclidean'))
    fitted_distances = np.tril(distance.cdist(fitted_points, fitted_points, metric='euclidean'))

    # calculation stress
    stress = np.sqrt(np.sum((actual_distances - fitted_distances) ** 2) / np.sum(actual_distances ** 2))

    return stress
