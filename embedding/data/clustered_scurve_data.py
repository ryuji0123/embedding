import pandas as pd
import numpy as np

from embedding.data.parent_data import ParentData
from embedding.data.synthesizer import ClusteredScurveSynthesizer


class ClusteredScurveData(ParentData):
    def __init__(self, *args):
        super(ClusteredScurveData, self).__init__(*args)
        self.setDataFrameAndColor()
        self.data_key = "clustered_scurve"

    def setDataFrameAndColor(self):
        synthesizer = ClusteredScurveSynthesizer()
        data, self.color = synthesizer.synthesizeData(n_dim=3, n_cluster=5, n_points=1000)
        data += abs(np.min(data))
        self.df = pd.DataFrame(data=data, columns=['col0', 'col1', 'col2'])
