from sklearn.manifold import SpectralEmbedding
from embedding.embedder import ParentEmbedder


class Laplacian_EigenmapsEmbedder(ParentEmbedder):
    def __init__(self, *args):
        super(Laplacian_EigenmapsEmbedder, self).__init__(*args)
        self.class_key = "lap-e_embedder"

    def execEmbed(self, dim=2, neighbor=10):
        self.em = SpectralEmbedding(n_components=dim, n_neighbors=neighbor).fit_transform(self.df)
