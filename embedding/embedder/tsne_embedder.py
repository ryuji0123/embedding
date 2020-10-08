from sklearn.manifold import TSNE

from embedding.embedder import ParentEmbedder


class TSNEEmbedder(ParentEmbedder):
    def embed(self, dim=2):
        self.em = TSNE(n_components=2).fit_transform(self.df)
