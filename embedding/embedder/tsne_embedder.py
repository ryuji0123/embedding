from sklearn.manifold import TSNE

from embedding.embedder.parent_embedder import ParentEmbedder


class TSNEEmbedder(ParentEmbedder):

    def __init__(self, *args):
        super(TSNEEmbedder, self).__init__(*args)
        self.class_key = "t_sne_embedder"

    def execEmbed(self, dim=3):
        self.em = TSNE(n_components=dim).fit_transform(self.df)
