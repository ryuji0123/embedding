import pandas as pd

from embedding.data import ParentData

from embedding.data.artificial_data_generator import generate_data


class ArtificialData(ParentData):
    def __init__(self, *args):
        super(ArtificialData, self).__init__(*args)
        self.df = self.getDataFrame(self.data_path)
        self.data_key = "artificial"

    def getDataFrame(self, root):
        data, cols = generate_data(n_dim=6, n_cluster=4, n_points=801)
        return pd.DataFrame(data, columns=cols)
