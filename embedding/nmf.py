from embedding import Data
from sklearn.decomposition import NMF

class NMFData(Data):
    def reduction(self, dim=2):
        model = NMF(n_components=dim, init='random', random_state=0)
        self.em = model.fit_transform(self.df)
