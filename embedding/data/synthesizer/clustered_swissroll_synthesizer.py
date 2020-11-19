import numpy as np

from embedding.data.synthesizer.parent_synthesizer import ParentSynthesizer


class ClusteredSwissrollSynthesizer(ParentSynthesizer):

    def __init__(self):
        super(ClusteredSwissrollSynthesizer, self).__init__()

    def synthesizeData(self, thick=1.5, **kwargs):
        points, cols, color = super(ClusteredSwissrollSynthesizer, self).synthesizeData(**kwargs)
        points = self.normalize(points)
        t = 1.5 * (1 + 2 * np.arccos(points[:, 0]))
        if 4.0 <= thick:
            raise('the dimention of points must be smaller than 4.')

        if points.shape[1] == 2:
            x = t * np.cos(t)
            z = t * np.sin(t)
        elif points.shape[1] == 3:
            x = t * np.cos(t) - points[:, 2] * (np.sin(t) + t * np.cos(t)) * thick / np.sqrt(1 + np.power(t, 2))
            z = t * np.sin(t) + points[:, 2] * (np.cos(t) - t * np.sin(t)) * thick / np.sqrt(1 + np.power(t, 2))
        y = points[:, 1] * 10
        X = np.stack((x, y, z), axis=1)

        return X, color
