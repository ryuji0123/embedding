from sklearn.decomposition import NMF
from embedding.reducer.parent_reducer import ParentReducer
import numpy as np


# modify input to be positive to eliminate error
def makePositive(data, embedder=None):
    data.df = data.df + abs(data.df.min().min())
    if embedder is not None:
        embedder.em = embedder.em + abs(embedder.em.min().min())
    return data, embedder


class NMFReducer(ParentReducer):

    def __init__(self, *args):
        args = makePositive(*args)
        super(NMFReducer, self).__init__(*args)
        self.class_key += "nmf_reducer"

    def execReduce(self, query, dim=2):
        transformer = NMF(
            n_components=dim, init="random", random_state=1, max_iter=1000
        ).fit(self.getDF(query))
        self.components = transformer.components_
        # self.rd = transformer.fit_transform(self.getDF(query))

        # Just use simple projection for simplicity (temporarily)

        self.rd = np.empty((len(self.getDF(query)), dim))
        for i, c in enumerate(self.components):
            self.rd[:, i] = self.getDF(query).to_numpy() @ c / (c @ c)
