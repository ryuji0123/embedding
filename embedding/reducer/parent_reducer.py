import pandas as pd
import numpy as np

from abc import ABCMeta, abstractmethod

from embedding.data import ParentData
from embedding.embedder import ParentEmbedder


# Create Orthogonal Bases
def doGramSchmidt(bases):
    u = []
    for i, c in enumerate(bases):
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

    def getDF(self, query="*"):
        if query == "*":
            return self.df
        return self.df.query(query)

    @abstractmethod
    def execReduce(self, query="*"):
        pass

    def reduce(self, use_cache=False, query="*", save_rd=True, **kwargs):
        if self.data.exists(self.class_key) and use_cache:
            self.rd = self.data.getResult(self.class_key)
        else:
            self.execReduce(query=query, **kwargs)
            # Set normal vector(normal hyperplane)
            self.setNormalVector()
            # Set Origin
            self.n_vec_src = np.zeros_like(self.n_vec)
            self.rd = pd.DataFrame(
                data=self.rd,
                columns=["col{}".format(i) for i in range(self.rd.shape[1])],
            )
            if save_rd:
                self.data.save(self.class_key, self.rd)

    def setRds(
        self,
        query1,
        query0="*",
        animation_option="linear",
        n_segments=10,
        **kwargs,
    ):
        self.query0 = query0
        self.query1 = query1

        # Get reduction on last-of-animation state
        self.reduce(query=query1, save_rd=False, **kwargs)
        n_vecs = [self.n_vec]
        n_vec_srcs = [self.n_vec_src]

        # Get reduction on beginning-of-animation state
        self.reduce(query=query0, save_rd=False, **kwargs)
        n_vecs.insert(0, self.n_vec)
        n_vec_srcs.insert(0, self.n_vec_src)

        # Devide into segments
        self.setSegments(n_vecs, n_vec_srcs, n_segments, animation_option)

        self.rds = []
        # Fo each basis in animation
        for cmp_ in self.cmps_oth:
            # For each basis vector of basises
            rd = np.zeros_like(self.rd)
            for i, c in enumerate(cmp_):
                # Project data to each basis
                rd[:, i] = self.getDF(query0).to_numpy() @ c
            rd = pd.DataFrame(
                data=rd,
                columns=["col{}".format(i) for i in range(self.rd.shape[1])],
            )
            self.rds.append(rd)

    def setSegments(self, n_vecs, n_vec_srcs, n_segments, animation_option="linear"):
        if animation_option == "linear":
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
        for cmp_ in cmps:
            doGramSchmidt(cmp_)
        self.cmps_oth = [np.array(doGramSchmidt(cmp_)) for cmp_ in cmps]

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
