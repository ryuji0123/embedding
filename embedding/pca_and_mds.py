from embedding import Data
from sklearn.manifold import MDS
from sklearn.decomposition import PCA

class PCAData(Data):
    def reduction(self, dim=2):
        pca = PCA(n_components=dim)
        self.em = pca.fit_transform(self.df)

class MDSData(Data):
    def reduction(self, dim=2):
        embedding = MDS(n_components=dim)
        self.em = embedding.fit_transform(self.df)
