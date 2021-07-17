import numpy as np

from scipy.spatial import distance
from scipy.stats import rankdata


def measure_ranking_stress_k_km(actual_points, fitted_points, n_representative_points, representative_indexes,
                                n_neighbors, is_z_value, rank_key):
    # actual_dataでのk個の近傍点と、それがfitted_dataで全て含まれるまでの近傍点(k個とk+m個の比較)の順位相関
    if rank_key == 'local':
        # make a distance matrix between each data points and each representative points in actual data space
        actual_distances_matrix_from_representative_points = distance.cdist(
            actual_points[representative_indexes], actual_points, metric='euclidean')

        # make a distance ranking matrix between each data points in fitted data space
        fitted_all_point_distance_matrix = distance.cdist(fitted_points, fitted_points, metric='euclidean')
        fitted_all_rank_matrix = rankdata(fitted_all_point_distance_matrix, axis=1)
    elif rank_key == 'middle':
        # make distance between all representative points matrices
        actual_distances_matrix_from_representative_points = distance.cdist(
            actual_points[representative_indexes], actual_points[representative_indexes], metric='euclidean')
        fitted_all_representative_points_distances = distance.cdist(
            fitted_points[representative_indexes], fitted_points[representative_indexes], metric='euclidean')
        fitted_all_rank_matrix = rankdata(fitted_all_representative_points_distances, axis=1)
    else:
        raise NameError(f'{rank_key} is not supported.')

    # get the representative_point and k_neighbors indexes for each representative points in actual data space
    actual_neighbor_indexes = np.argsort(actual_distances_matrix_from_representative_points, axis=1)[:, :n_neighbors + 1]

    # calc local ranking stress about each representative points
    representative_stresses = np.zeros(n_representative_points)
    all_max_rank_values = np.zeros((n_representative_points, n_neighbors + 1))
    for i in range(n_representative_points):

        # extract 1 representative point and its k-neighbors(these are called target points)
        actual_target_points_per_representative_point = actual_points[actual_neighbor_indexes[i]]

        # make argsort_matrix from distance matrix between all target points
        actual_target_point_distance_matrix = distance.cdist(actual_target_points_per_representative_point,
                                                             actual_target_points_per_representative_point,
                                                             metric='euclidean')
        actual_target_sorted_indexes = np.argsort(actual_target_point_distance_matrix, axis=1)

        # make rank matrix about the points corresponding target points in fitted data space
        fitted_target_rank_matrix = np.zeros((n_neighbors + 1, n_neighbors + 1))
        for j, target_point_index in enumerate(actual_neighbor_indexes[i]):
            fitted_target_rank_matrix[j] = fitted_all_rank_matrix[
                target_point_index, actual_neighbor_indexes[i, actual_target_sorted_indexes[j]]]

        fitted_target_rank_matrix -= 1.0
        all_max_rank_values[i] = fitted_target_rank_matrix.max(axis=1)

        # calc rank losses about each target points, and the average value is the final stress value about target points
        if is_z_value:
            representative_stresses[i] = (
                np.abs(
                    fitted_target_rank_matrix - np.arange(0, n_neighbors + 1).reshape(1, -1).repeat(n_neighbors + 1,
                                                                                                    axis=0)
                ).sum(axis=1)
            ).mean(axis=0)
        else:
            worst_values = worst_value_calculation(n_neighbors, all_max_rank_values[i])
            representative_stresses[i] = ((
                                              np.abs(
                                                  fitted_target_rank_matrix - np.arange(0, n_neighbors + 1).reshape(1,
                                                                                                                    -1).repeat(
                                                      n_neighbors + 1,
                                                      axis=0)
                                              ).sum(axis=1)) / worst_values).mean(axis=0)

    # return the median value of stresses
    return np.median(np.array(representative_stresses)), all_max_rank_values


def measure_ranking_stress_k_k(actual_points, fitted_points, n_representative_points, representative_indexes,
                               n_neighbors, is_z_value, rank_key):
    # Dでのk個の近傍点だけ抜き取り、D'もそのkこだけでの順位を考える(k個とk個)
    # make distance matrices
    if rank_key == 'local':
        actual_distances_from_representative_points = distance.cdist(actual_points[representative_indexes],
                                                                     actual_points,
                                                                     metric='euclidean')
    elif rank_key == 'middle':
        actual_distances_from_representative_points = distance.cdist(actual_points[representative_indexes],
                                                                     actual_points[representative_indexes],
                                                                     metric='euclidean')
    else:
        raise NameError(f'{rank_key} is not supported.')

    # get k_neighbors for each representative points
    actual_neighbor_indexes = np.argsort(actual_distances_from_representative_points, axis=1)[:, :n_neighbors + 1]

    representative_stresses = np.zeros(n_representative_points)
    for i in range(n_representative_points):
        actual_target_points_per_representative_point = actual_points[actual_neighbor_indexes[i]]
        fitted_target_points_per_representative_point = fitted_points[actual_neighbor_indexes[i]]

        actual_target_point_distances = distance.cdist(actual_target_points_per_representative_point,
                                                       actual_target_points_per_representative_point,
                                                       metric='euclidean')
        fitted_target_point_distances = distance.cdist(fitted_target_points_per_representative_point,
                                                       fitted_target_points_per_representative_point,
                                                       metric='euclidean')

        # make rank matrix about the points corresponding target points in actual_data and fitted_data space
        actual_target_point_ranking_matrix = rankdata(actual_target_point_distances, axis=1) - 1.0
        fitted_target_point_ranking_matrix = rankdata(fitted_target_point_distances, axis=1) - 1.0

        # calc rank losses about each target points, and the average value is the final stress value about target points
        representative_stresses[i] = (
                (actual_target_point_ranking_matrix - fitted_target_point_ranking_matrix) ** 2).sum(
            axis=1).mean(axis=0)
        if not is_z_value:
            worst_value = worst_value_calculation(n_neighbors)
            representative_stresses[i] = representative_stresses[i] / worst_value

    if is_z_value:
        return np.mean(np.array(representative_stresses))
    else:
        return np.median(np.array(representative_stresses))


def measure_ranking_stress_intersectional(actual_points, fitted_points, n_representative_points,
                                          representative_indexes, n_neighbors, is_z_value, rank_key, weight=1.0):
    # DとD'におけるあるK（ある程度大きな数）までの共通する近傍点k個(k個とk個)
    # make distance matrices
    if rank_key == 'local':
        actual_distances_from_representative_points = distance.cdist(actual_points[representative_indexes],
                                                                     actual_points,
                                                                     metric='euclidean')
        fitted_distances_from_representative_points = distance.cdist(fitted_points[representative_indexes],
                                                                     fitted_points,
                                                                     metric='euclidean')
    elif rank_key == 'middle':
        actual_distances_from_representative_points = distance.cdist(actual_points[representative_indexes],
                                                                     actual_points[representative_indexes],
                                                                     metric='euclidean')
        fitted_distances_from_representative_points = distance.cdist(fitted_points[representative_indexes],
                                                                     fitted_points[representative_indexes],
                                                                     metric='euclidean')
    else:
        raise NameError(f'{rank_key} is not supported.')

    actual_ranking_matrix_based_representative_points = rankdata(actual_distances_from_representative_points, axis=1)
    fitted_ranking_matrix_based_representative_points = rankdata(fitted_distances_from_representative_points, axis=1)

    intersectional_neighbor_indexes = np.argsort(
        weight * actual_ranking_matrix_based_representative_points + fitted_ranking_matrix_based_representative_points,
        axis=1)[:, :1 + n_neighbors]

    representative_stresses = np.zeros(n_representative_points)
    for i in range(n_representative_points):
        actual_target_points_per_representative_point = actual_points[intersectional_neighbor_indexes[i]]
        fitted_target_points_per_representative_point = fitted_points[intersectional_neighbor_indexes[i]]

        actual_target_point_distances = distance.cdist(actual_target_points_per_representative_point,
                                                       actual_target_points_per_representative_point,
                                                       metric='euclidean')
        fitted_target_point_distances = distance.cdist(fitted_target_points_per_representative_point,
                                                       fitted_target_points_per_representative_point,
                                                       metric='euclidean')

        actual_target_point_ranking_matrix = rankdata(actual_target_point_distances, axis=1) - 1.0
        fitted_target_point_ranking_matrix = rankdata(fitted_target_point_distances, axis=1) - 1.0

        # calc rank losses about each target points, and the final stress value is the average value about target points
        representative_stresses[i] = (
                    (actual_target_point_ranking_matrix - fitted_target_point_ranking_matrix) ** 2).sum(axis=1).mean(
            axis=0)
        if not is_z_value:
            worst_value = worst_value_calculation(n_neighbors)
            representative_stresses[i] = representative_stresses[i] / worst_value

    if is_z_value:
        return np.mean(np.array(representative_stresses))
    else:
        return np.median(np.array(representative_stresses))


def measure_global_ranking_stress(actual_points, fitted_points, n_representative_points, representative_indexes, is_z_value):
    # actual_dataでのn個の代表点同士のストレス

    # make distance between all representative points matrices
    actual_all_representative_points_distances = distance.cdist(
        actual_points[representative_indexes], actual_points[representative_indexes], metric='euclidean')
    fitted_all_representative_points_distances = distance.cdist(
        fitted_points[representative_indexes], fitted_points[representative_indexes], metric='euclidean')
    actual_all_rank_matrix = rankdata(actual_all_representative_points_distances, axis=1)
    fitted_all_rank_matrix = rankdata(fitted_all_representative_points_distances, axis=1)

    # calc rank losses about each target points, and the average value is the final stress value about target points
    representative_stresses = ((actual_all_rank_matrix - fitted_all_rank_matrix) ** 2).sum(axis=0)
    if is_z_value:
        representative_stresses = representative_stresses / n_representative_points
    else:
        worst_value = worst_value_calculation(n_representative_points)
        representative_stresses = representative_stresses / worst_value

    if is_z_value:
        return np.mean(np.array(representative_stresses))
    else:
        return np.median(np.array(representative_stresses))


def worst_value_calculation(n_neighbors, n_extra_ranks=[]):
    if len(n_extra_ranks) == 0:
        worst_value = n_neighbors * (n_neighbors + 1) * (n_neighbors - 1) / 3
    else:
        worst_value = np.zeros(len(n_extra_ranks))
        for i, n_extra_rank in enumerate(n_extra_ranks):
            if n_extra_rank == n_neighbors:
                worst_value[i] = n_neighbors * (n_neighbors - 1) / 2
            elif n_extra_rank > 2 * n_neighbors - 2:
                if (n_neighbors - n_extra_rank) % 2 == 0:
                    worst_value[i] = (n_neighbors ** 2 + n_extra_rank ** 2) / 2
                else:
                    worst_value[i] = (n_neighbors ** 2 + (n_extra_rank + 1) * (n_extra_rank - 1)) / 2
            else:
                change_point_value = int((2 * n_neighbors + n_extra_rank + 2) / 4)
                worst_value[i] = (2 * n_neighbors + n_extra_rank - change_point_value) * change_point_value

    return worst_value
