import pandas as pd

from embedding.data.parent_data import ParentData

from embedding.data.synthesizer import ParentSynthesizer


class BasicClusterData(ParentData):
    def __init__(self, *args):
        super(BasicClusterData, self).__init__(*args)
        self.setDataFrameAndColor()
        self.data_key = "artificial"

    def setDataFrameAndColor(self):
        synthesizer = ParentSynthesizer()
        data, cols, self.color = synthesizer.synthesizeData(n_dim=6, n_cluster=4, n_points=801)
        self.df = pd.DataFrame(data, columns=cols)
