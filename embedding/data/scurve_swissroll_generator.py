import numpy as np


def normalize(points):
    for d in range(points.shape[1]):
        mid_point = (points[:, d].min() + points[:, d].max()) / 2
        points[:, d] -= mid_point
        points[:, d] = points[:, d] / np.abs(points[:, d].max())
    return points


def generate_scurve(points, thick=0.3):
    points = normalize(points)
    if thick >= 1.0:
        raise('thick must be smaller than 1')
    t = np.arcsin(points[:, 0]) * 3
    if points.shape[1] == 2:
        x = np.sin(t)
        z = np.sign(t) * (np.cos(t) - 1)
    elif points.shape[1] == 3:
        x = np.sin(t) + (points[:, 2] * np.sign(t) * np.sin(t)) * thick
        z = np.sign(t) * (np.cos(t) - 1) + points[:, 2] * np.cos(t) * thick
    else:
        raise('the dimention of points must be smaller than 4.')
    y = points[:, 1]
    X = np.stack((x, y, z), axis=1)
    return X


def generate_swissroll(points, thick):
    points = normalize(points)
    t = 1.5 * (1 + 2 * np.arccos(points[:, 0]))
    if points.shape[1] == 2:
        x = t * np.cos(t)
        z = t * np.sin(t)
    elif points.shape[1] == 3:
        x = t * np.cos(t) - points[:, 2] * (np.sin(t) + t * np.cos(t)) * thick / np.sqrt(1 + np.power(t, 2))
        z = t * np.sin(t) + points[:, 2] * (np.cos(t) - t * np.sin(t)) * thick / np.sqrt(1 + np.power(t, 2))
    else:
        raise('the dimention of points must be smaller than 4.')
    y = points[:, 1] * 10
    X = np.stack((x, y, z), axis=1)
    return X
