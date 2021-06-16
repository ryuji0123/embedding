import pandas as pd
import numpy as np

from sklearn import datasets

from embedding.data.parent_data import ParentData


class SwrollData(ParentData):
    def __init__(self, *args):
        super(SwrollData, self).__init__(*args)
        self.set_dataframe_and_color()
        self.data_key = "swissroll"

    def set_dataframe_and_color(self):
        data, self.color = datasets.make_swiss_roll(n_samples=1000, random_state=0)
        data += abs(np.min(data))
        self.df = pd.DataFrame(data=data, columns=['col0', 'col1', 'col2'])
