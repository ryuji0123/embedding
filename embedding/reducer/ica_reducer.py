from sklearn.decomposition import FastICA
import numpy as np
from embedding.reducer.parent_reducer import ParentReducer


class ICAReducer(ParentReducer):

    def __init__(self, *args):
        super(ICAReducer, self).__init__(*args)
        self.class_key += "ica_reducer"


    def execReduce(self, query, dim=2, axis=1):
        #  Fast ICA requires normalizing on user's side
        data = (
            self.getDF(query)
            - np.mean(np.array(self.getDF(query)), axis=axis)[:, np.newaxis]
        )
        ica = FastICA(n_components=dim)
        self.rd = ica.fit_transform(data)
        self.cmp = ica.components_

