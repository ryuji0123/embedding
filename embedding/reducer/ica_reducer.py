from sklearn.decomposition import FastICA
import numpy as np
from embedding.reducer.parent_reducer import ParentReducer


class ICAReducer(ParentReducer):

    def __init__(self, *args):
        super(ICAReducer, self).__init__(*args)
        self.class_key += "ica_reducer"

    def execReduce(self, query, dim=2, axis=1):
        # Fast ICA requires normalizing by user
        data = (
            self.getDF(query)
            - np.mean(np.array(self.getDF(query)), axis=axis)[:, np.newaxis]
        )
        transformer = FastICA(n_components=dim).fit(data)
        self.cmp = transformer.components_

        # self.rd = transformer.fit_transform(data)

        # Just use simple projection for simplicity (temporarily)
        self.rd = np.empty((len(self.getDF(query)), dim))
        for i, c in enumerate(self.cmp):
            self.rd[:, i] = self.getDF(query).to_numpy() @ c / (c @ c)
