from sklearn.decomposition import PCA

from embedding.reducer.parent_reducer import ParentReducer
import numpy as np


class PCAReducer(ParentReducer):
    def __init__(self, *args):
        super(PCAReducer, self).__init__(*args)
        self.class_key += "pca_reducer"

    def execReduce(self, query, dim=2):
        transformer = PCA(n_components=dim).fit(self.getDF(query))
        # Store principal components

        self.cmp = transformer.components_

        # self.rd = transformer.transform(self.getDF(query))

        # Just use simple projection for simplicity (temporarily)
        self.rd = np.empty((len(self.getDF(query)), dim))
        for i, c in enumerate(self.cmp):
            self.rd[:, i] = self.getDF(query).to_numpy() @ c / (c @ c)
