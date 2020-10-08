from sklearn.manifold import Isomap
from embedding.embedder import ParentEmbedder


class IsomapEmbedder(ParentEmbedder):
    def embed(self, dim=2, neighbor=5):
        self.em = Isomap(n_components=dim, n_neighbors=neighbor).fit_transform(self.df)