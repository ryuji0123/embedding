import pandas as pd
import numpy as np

from abc import ABCMeta, abstractmethod

from embedding.data import ParentData
from embedding.embedder import ParentEmbedder


class ParentReducer(metaclass=ABCMeta):
    def __init__(self, data, embedder=None):
        if not isinstance(data, ParentData):
            raise ValueError(f"{type(data)} should inherit {ParentData}")
        if embedder is None:
            self.data = data
            self.df = data.df
            self.class_key = ""
        elif isinstance(embedder, ParentEmbedder):
            self.data = data
            self.df = embedder.em
            self.class_key = f"{embedder.class_key}_and_"
        else:
            raise ValueError(
                f"{type(embedder)} should be None or {type(ParentEmbedder)}"
            )

    def reduce(self, use_cache=False, **kwargs):
        if self.data.exists(self.class_key) and use_cache:
            self.rd = self.data.getResult(self.class_key)
        else:
            self.execReduce(**kwargs)
            # Set normal vector(normal hyperplane)
            self.setNormalVector()
            self.n_vecs = [self.n_vec]
            # Set Origin
            self.n_vec_src = np.zeros_like(self.n_vec)
            self.rd = pd.DataFrame(
                data=self.rd,
                columns=[str(i) for i in range(self.rd.shape[1])],
            )
            self.data.save(self.class_key, self.rd)

    def reduceFiltered(self, em_filtered, n_segments=10, **kwargs):
        # Swap dataframe temporarily, and hold original em
        df_temp = self.df.copy()
        self.df = em_filtered

        self.execReduce(**kwargs)
        self.setNormalVector()

        cmp_end = self.cmp
        n_vec_end = self.n_vec
        n_vec_src_end = self.n_vec_src
        rd_end = self.rd

        # Swap back to original em
        self.df = df_temp

        self.execReduce(**kwargs)
        self.setNormalVector()

        cmp_start = self.cmp
        n_vec_start = self.n_vec
        n_vec_src_start = self.n_vec_src
        rd_start = self.rd

        # Devide into segments
        ...

    # Store normal vector representing plane formed by principal components
    # When dim>2, they are called: normal space/affine subspace/normal hyperplane
    # May need error check!

    def setNormalVector(self):
        A = np.vstack((self.cmp.copy(), np.ones_like(self.cmp[0, :])))
        y = np.zeros_like(A[:, 0])
        y[len(y) - 1] = 1
        # Solve for
        n_vec = np.linalg.lstsq(A, y, rcond=None)[0]
        # Normalize (maybe don't have to)
        self.n_vec = n_vec / np.linalg.norm(n_vec)

    @abstractmethod
    def execReduce(self):
        pass
