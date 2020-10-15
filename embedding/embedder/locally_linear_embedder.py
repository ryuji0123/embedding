from sklearn.manifold import LocallyLinearEmbedding

from embedding.embedder.parent_embedder import ParentEmbedder


class LocallyLinearEmbedder(ParentEmbedder):
    def __init__(self, *args):
        super(LocallyLinearEmbedder, self).__init__(*args)
        self.class_key = "locally_linear_embedder"

    def execEmbed(self, dim=2):
        self.em = LocallyLinearEmbedding(n_components=dim).fit_transform(self.df)
