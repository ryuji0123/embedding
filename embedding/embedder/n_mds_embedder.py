from sklearn.manifold import MDS
from embedding.embedder import ParentEmbedder


class N_MDSEmbedder(ParentEmbedder):
    def embed(self, dim=2, metric=None):
        self.em = MDS(n_components=dim, metric=metric).fit_transform(self.df)
