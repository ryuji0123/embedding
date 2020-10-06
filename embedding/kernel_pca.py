from embedding import Data
from sklearn.decomposition import KernelPCA

class KernelPCAData(Data):
    def reduction(self, dim=2):
        # Separated by 'transformer' variable for readability
        transformer = KernelPCA(n_components=dim, kernel='linear')
        self.em = transformer.fit_transform(self.df)
        # Need not assert in definition??
        assert self.em.shape[1] == dim


