import pandas as pd
import numpy as np

from abc import ABCMeta, abstractmethod

from embedding.data import ParentData
from embedding.embedder import ParentEmbedder

# Create Orthogonal Basis
def doGramSchmidt(a):
    u = []
    for i, c in enumerate(a):
        v = c - sum([(c @ u[j]) * u[j] for j in range(i)])
        u.append(v / np.linalg.norm(v))
    return u


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

    def calcFilteredRds(self, em_filtered, n_segments=10, **kwargs):
        # Swap dataframe temporarily, and hold original em
        df_temp = self.df.copy()
        self.df = em_filtered

        self.execReduce(**kwargs)
        self.setNormalVector()

        n_vecs = [self.n_vec]
        n_vec_srcs = [self.n_vec_src]

        # Swap back to original em
        self.df = df_temp

        self.execReduce(**kwargs)
        self.setNormalVector()

        n_vecs.insert(0, self.n_vec)
        n_vec_srcs.insert(0, self.n_vec_src)

        # Devide into segments
        self.calcSegments(n_vecs, n_vec_srcs, n_segments)

        self.rds = []
        for cmp_ in self.cmps_oth:
            self.rds.append([ np.multiply(self.df.to_numpy() @ c, np.tile(c, (len(self.df), 1)).T) for c in cmp_])


    def calcSegments(self, n_vecs, n_vec_srcs, n_segments, option="linear"):
        if option is "linear":
            self.n_vecs = np.linspace(
                n_vecs[0],
                n_vecs[1],
                num=n_segments,
                endpoint=True,
            )
            self.n_vec_srcs = np.zeros_like(self.n_vecs)

        # Calculate components based on Normal Vector Movement
        cmps = [(n_vec - n_vecs[0]) + self.cmp for n_vec in self.n_vecs]

        # Store orthogonal components
        self.cmps_oth = [doGramSchmidt(cmp_) for cmp_ in cmps]

    # Store normal vector representing plane formed by principal components
    # When dim>2, they are called: normal space/affine subspace/normal hyperplane
    # May need error check!

    def setNormalVector(self):
        random_coefficients = np.random.rand(len(self.cmp[0, :]))
        A = np.vstack((self.cmp.copy(), random_coefficients))
        y = np.zeros_like(A[:, 0])
        y[-1] = np.random.rand(1)

        # Solve for
        n_vec = np.linalg.lstsq(A, y, rcond=None)[0]
        # Normalize (maybe don't have to)
        self.n_vec = n_vec / np.linalg.norm(n_vec)

    @abstractmethod
    def execReduce(self):
        pass


# %%
