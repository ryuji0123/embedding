from sklearn.manifold import TSNE
import pandas as pd

from embedding.embedder.parent_embedder import ParentEmbedder


class TSNEEmbedder(ParentEmbedder):
    def __init__(self, *args):
        super(TSNEEmbedder, self).__init__(*args)
        self.class_key = "t-sne_embedder"

    def execEmbed(self, dim=3):
        embedded = TSNE(n_components=dim).fit_transform(self.df)
        self.em = pd.DataFrame(
            data=embedded,
            columns=["{}".format(i) for i in range(embedded.shape[1])],
        )
