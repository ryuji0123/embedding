from sklearn.decomposition import NMF
from embedding.reducer import ParentReducer


class NMFReducer(ParentReducer):
    def __init__(self, *args):
        super(NMFReducer, self).__init__(*args)
        self.class_key = "nmf_reducer"

    def execReduce(self, dim=2):
        self.rd = NMF(n_components=dim, init='random', random_state=0).fit_transform(self.df)
