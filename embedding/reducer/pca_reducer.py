from sklearn.manifold import MDS
from sklearn.decomposition import PCA

from embedding.reducer import ParentReducer

class PCAReducer(ParentReducer):
    def __init__(self, *args):
        super(PCAReducer, self).__init__(*args)
        self.class_key = "pca_reducer"

    def execReduce(self, dim=2):
        self.em = PCA(n_components=dim).fit_transform(self.df)

class MDSData(Data):
    def reduction(self, dim=2):
        embedding = MDS(n_components=dim)
        self.em = embedding.fit_transform(self.df)
