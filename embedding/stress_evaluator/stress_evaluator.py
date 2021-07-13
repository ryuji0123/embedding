import numpy as np
import pandas as pd
import os

from embedding.stress_evaluator.kruskal_stress_evaluator import measure_kruskal_stress
from embedding.stress_evaluator.ranking_based_stress import measure_ranking_stress_k_k, \
    measure_ranking_stress_k_km, measure_ranking_stress_intersectional


class StressEvaluator:

    def __init__(self, actual_df, n_representative_points=100):
        self.actual_points = convert_dataframe_to_numpy(actual_df)
        self.local_extra_rank_values = []
        self.middle_extra_rank_values = []

        # representative points assertion
        assert self.actual_points.shape[0] > n_representative_points, \
            'The number of representative points should be smaller than that of data.'

        self.n_representative_points = n_representative_points if n_representative_points \
            else min(100, self.actual_points.shape[0])

        self.representative_indexes = choice_representative_indexes(self.actual_points, self.n_representative_points)

    def kruskal(self, fitted_df):
        # shape assertion
        assert self.actual_points.shape[0] == fitted_df.shape[0], \
            f'The number of points should be same. actual_data: {self.actual_points.shape}, fitted_data: {fitted_df.shape}'

        return measure_kruskal_stress(self.actual_points, convert_dataframe_to_numpy(fitted_df))

    def local_ranking(self, fitted_df, n_neighbors, neighbor_key='k-k+m', weight=1.0, is_z_value=True):
        # shape assertion
        assert self.actual_points.shape[0] == fitted_df.shape[0], \
            f'The number of points should be same. raw_data: {self.actual_points.shape}, fitted_data: {fitted_df.shape}'

        stress = 0
        if neighbor_key == 'k-k+m':
            stress, self.local_extra_rank_values = measure_ranking_stress_k_km(self.actual_points,
                                                                               convert_dataframe_to_numpy(
                                                                                   fitted_df),
                                                                               self.n_representative_points,
                                                                               self.representative_indexes,
                                                                               n_neighbors, is_z_value, rank_key='local')

        if neighbor_key == 'k-k':
            stress = measure_ranking_stress_k_k(self.actual_points, convert_dataframe_to_numpy(fitted_df),
                                                self.n_representative_points, self.representative_indexes,
                                                n_neighbors, is_z_value, rank_key='local')

        if neighbor_key == 'intersectional':
            stress = measure_ranking_stress_intersectional(self.actual_points,
                                                           convert_dataframe_to_numpy(fitted_df),
                                                           self.n_representative_points,
                                                           self.representative_indexes,
                                                           n_neighbors, is_z_value, rank_key='local', weight=weight)
        if is_z_value:
            if neighbor_key == 'k-k+m':
                local_k_km_random_rank_mean, local_k_km_random_rank_var = self.extract_random_rank_distribution_params(
                    n_neighbors, self.local_extra_rank_values)
                z_stress = stress_standardize(stress, [local_k_km_random_rank_mean, local_k_km_random_rank_var])
            else:
                local_k_k_random_rank_mean, local_k_k_random_rank_var = self.extract_random_rank_distribution_params(
                    n_neighbors)
                z_stress = stress_standardize(stress, [local_k_k_random_rank_mean, local_k_k_random_rank_var])

            return z_stress

        return stress

    def middle_ranking(self, fitted_df, n_neighbors, neighbor_key='k-k', weight=1.0, is_z_value=True):
        # shape assertion
        assert self.actual_points.shape[0] == fitted_df.shape[0], \
            f'The number of points should be same. raw_data: {self.actual_points.shape}, fitted_data: {fitted_df.shape}'
        # n_neighbors assertion
        assert self.n_representative_points >= n_neighbors, \
            f'The number of neighbors points should be smaller than number of representative points.' \
            f'n_representative_points: {self.n_representative_points}, n_neighbors: {n_neighbors}'

        stress = 0
        if neighbor_key == 'k-k+m':
            stress, self.middle_extra_rank_values = measure_ranking_stress_k_km(self.actual_points,
                                                                                convert_dataframe_to_numpy(
                                                                                    fitted_df),
                                                                                self.n_representative_points,
                                                                                self.representative_indexes,
                                                                                n_neighbors, is_z_value,
                                                                                rank_key='middle')

        if neighbor_key == 'k-k':
            stress = measure_ranking_stress_k_k(self.actual_points, convert_dataframe_to_numpy(fitted_df),
                                                self.n_representative_points, self.representative_indexes,
                                                n_neighbors, is_z_value, rank_key='middle')
        if neighbor_key == 'intersectional':
            stress = measure_ranking_stress_intersectional(self.actual_points,
                                                           convert_dataframe_to_numpy(fitted_df),
                                                           self.n_representative_points,
                                                           self.representative_indexes,
                                                           n_neighbors, is_z_value, rank_key='middle', weight=weight)

        if is_z_value:
            if neighbor_key == 'k-k+m':
                middle_k_km_random_rank_mean, middle_k_km_random_rank_var = self.extract_random_rank_distribution_params(
                    n_neighbors, self.middle_extra_rank_values)
                z_stress = stress_standardize(stress, [middle_k_km_random_rank_mean, middle_k_km_random_rank_var])
            else:
                middle_k_k_random_rank_mean, middle_k_k_random_rank_var = self.extract_random_rank_distribution_params(
                    n_neighbors)
                z_stress = stress_standardize(stress, [middle_k_k_random_rank_mean, middle_k_k_random_rank_var])

            return z_stress

        return stress

    def extract_random_rank_distribution_params(self, n_neighbors, n_extra_ranks=[]):
        param_cache_path = os.path.join(os.path.sep, os.getcwd(), 'embedding', 'stress_evaluator', "cache")
        if len(n_extra_ranks) == 0:
            file_name = f'{param_cache_path}/{self.n_representative_points}-representative_random_rank_distribution_params.csv'
            try:
                params = pd.read_csv(file_name, index_col=0)
                if float(n_neighbors) in params.index.to_list():

                    return params.loc[n_neighbors]['mean'], params.loc[n_neighbors]['var']
                else:
                    mean, var = self.calc_random_rank_distribution_params(n_neighbors)
                    params.loc[n_neighbors] = [mean, var]
                    params.to_csv(file_name)
            except FileNotFoundError:
                mean, var = self.calc_random_rank_distribution_params(n_neighbors)
                params = pd.Series([n_neighbors, mean, var], index=['n_neighbors', 'mean', 'var'])
                params = params.to_frame().T
                params.to_csv(file_name, index=False)

            return mean, var
        else:
            return self.calc_random_rank_distribution_params(n_neighbors, n_extra_ranks)

    def calc_random_rank_distribution_params(self, n_neighbors, n_extra_ranks=[]):
        n_iter = 1000
        all_state = np.zeros(n_iter)
        if len(n_extra_ranks) == 0:
            for iteration in range(n_iter):
                one_represent_state = np.zeros(self.n_representative_points)
                for i in range(self.n_representative_points):
                    one_represent_state[i] = np.sum(
                        (np.arange(1, n_neighbors + 1) - np.random.permutation(
                            np.arange(1, n_neighbors + 1))) ** 2).mean()
                all_state[iteration] = np.mean(one_represent_state)
        else:
            for iteration in range(n_iter):
                one_represent_state = np.zeros(self.n_representative_points)
                for i in range(self.n_representative_points):
                    for j in range(n_neighbors):
                        one_represent_state[i] += np.sum(
                            np.abs(np.arange(1, n_neighbors + 1) - np.random.permutation(
                                np.arange(1, n_neighbors + n_extra_ranks[i, j] + 1))[:n_neighbors]))
                    one_represent_state[i] = one_represent_state[i] / n_neighbors
                all_state[iteration] = np.median(one_represent_state)

        return all_state.mean(), np.var(all_state)


def choice_representative_indexes(data_points, n_representative_points):
    n_points = data_points.shape[0]
    representative_indexes = []

    # select k-points according to a weighted conditional probability
    probability = np.repeat(1 / n_points, n_points)
    for i in range(n_representative_points):
        representative_indexes.append(np.random.choice(np.arange(n_points), 1, p=probability)[0])
        d = calc_min_distance_from_representative_points(data_points, representative_indexes)
        probability = d / np.sum(d)

    return representative_indexes


def calc_min_distance_from_representative_points(data_points, representative_indexes):
    d = np.zeros((len(representative_indexes), data_points.shape[0]))
    for i, index in enumerate(representative_indexes):
        d[i] = np.sum((data_points - data_points[index]) ** 2, axis=1)

    return np.min(d, axis=0)


def convert_dataframe_to_numpy(data_points):
    if isinstance(data_points, pd.DataFrame):
        return np.array(data_points)

    return data_points


def stress_standardize(stress, params):
    return (stress - params[0]) / np.sqrt(params[1])
