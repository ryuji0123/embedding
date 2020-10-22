from sklearn.decomposition import PCA
import numpy as np
import pandas as pd

from embedding.reducer.parent_reducer import ParentReducer


class PCAReducer(ParentReducer):
    def __init__(self, *args):
        super(PCAReducer, self).__init__(*args)
        self.class_key += "pca_reducer"

    def execReduce(self, dim=2):
        pca = PCA(n_components=dim)
        # Store principal components
        self.cmp = pca.fit(self.df).components_
        self.set_normal_vector()
        self.n_vec_src = np.zeros_like(self.n_vec)

        reduced = pca.fit_transform(self.df)
        self.rd = pd.DataFrame(
            data=reduced,
            columns=["{}".format(i) for i in range(reduced.shape[1])],
        )
