import numpy as np

from embedding.data.synthesizer.parent_synthesizer import ParentSynthesizer


class ClusteredScurveSynthesizer(ParentSynthesizer):

    def __init__(self):
        super(ClusteredScurveSynthesizer, self).__init__()

    def synthesizeData(self, thick=0.3, **kwargs):
        points, cols, color = super(ClusteredScurveSynthesizer, self).synthesizeData(**kwargs)
        points = self.normalize(points)
        if 1.0 <= thick:
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

        return np.stack((x, points[:, 1], z), axis=1), color
