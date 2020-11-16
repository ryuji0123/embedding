import pandas as pd
import numpy as np

from embedding.data.parent_data import ParentData

from embedding.data.artificial_data_generator import generate_data

from embedding.data.scurve_swissroll_generator import generate_swissroll


class ClusteredSwissrollData(ParentData):
    def __init__(self, *args):
        super(ClusteredSwissrollData, self).__init__(*args)
        self.setDataFrameAndColor()
        self.data_key = "clustered-swissroll"

    def setDataFrameAndColor(self):
        points, cols, self.color = generate_data(n_dim=3, n_cluster=5, n_points=10000)
        data = generate_swissroll(points=points, thick=1.5)
        data += abs(np.min(data))
        self.df = pd.DataFrame(data=data, columns=['col0', 'col1', 'col2'])
