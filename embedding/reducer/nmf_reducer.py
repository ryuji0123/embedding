from sklearn.decomposition import NMF
from embedding.reducer.parent_reducer import ParentReducer


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
        nmf = NMF(n_components=dim, init="random", random_state=1, max_iter=1000)
        self.rd = nmf.fit_transform(self.getDF(query))
        self.cmp = nmf.components_


