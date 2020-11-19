from sklearn.manifold import MDS
from embedding.embedder.parent_embedder import ParentEmbedder


class N_MDSEmbedder(ParentEmbedder):

    def __init__(self, *args):
        super(N_MDSEmbedder, self).__init__(*args)
        self.class_key = "n-mds_embedder"

    def execEmbed(self, dim=2, metric=None):
        self.em = MDS(n_components=dim, metric=metric).fit_transform(self.df)
