import numpy as np
import pandas as pd

from notebooks.embedding.stress_evaluator.kruskal_stress_evaluator import measure_kruskal_stress


class StressEvaluator:

    def __init__(self, actual_df, n_representative_points=100):
        self.actual_points = convert_dataframe_to_numpy(actual_df)

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
