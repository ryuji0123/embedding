from sklearn.decomposition import PCA

from embedding.reducer.parent_reducer import ParentReducer


class PCAReducer(ParentReducer):
    def __init__(self, *args):
        super(PCAReducer, self).__init__(*args)
        self.class_key += "pca_reducer"

    def execReduce(self, dim=2):
        pca = PCA(n_components=dim)
        # Store principal components
        self.cmp = pca.fit(self.df).components_
        self.rd = pca.fit_transform(self.df)
