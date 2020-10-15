from sklearn.decomposition import FastICA
import numpy as np
from embedding.reducer.parent_reducer import ParentReducer


class ICAReducer(ParentReducer):
    def __init__(self, *args):
        super(ICAReducer, self).__init__(*args)
        self.class_key += "ica_reducer"

    def execReduce(self, dim=2, axis=1):
        #  Fast ICA need to normalize by myself.
        data = self.df - np.mean(np.array(self.df), axis=axis)[:, np.newaxis]
        self.rd = FastICA(n_components=dim).fit_transform(data)
