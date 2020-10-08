from sklearn.manifold import Isomap
from embedding.embedder.parent_embedder import ParentEmbedder


class IsomapEmbedder(ParentEmbedder):
    def __init__(self, *args):
        super(IsomapEmbedder, self).__init__(*args)
        self.class_key = "isomap_embedder"

    def execEmbed(self, dim=2, neighbor=5):
        self.em = Isomap(n_components=dim, n_neighbors=neighbor).fit_transform(self.df)
