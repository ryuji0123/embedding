from sklearn.manifold import TSNE

from embedding import Data


class TSNEData(Data):
    def reduction(self, dim=2):
        self.em = TSNE(n_components=2).fit_transform(self.df)
