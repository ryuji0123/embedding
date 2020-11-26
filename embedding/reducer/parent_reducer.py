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

    # Obtain DataFrame filtered by input query.
    # Input '*' to get whole DataFrame
    def getDF(self, query="*"):
        if query == "*":
            return self.df
        return self.df.query(query)

    # Obtain indices of filtered-data within whole DataFrame
    def getDFIndices(self, query="*"):
        if query == "*":
            return self.df.index
        return self.df.query(query).index

    # Obtain concatenated rds as DataFrame.
    # MUST RUN .setRds FIRST
    def getRdsDf(self):
        try:
            return pd.concat(self.rds)
        except ValueError:
            print("rds is not defined. run .setRds() first.")
        return None

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
            self.initial_points_of_normal_vectors = np.zeros_like(self.normal_vectors)
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
        n_animation_frame=10,
        **kwargs,
    ):
        self.query0 = query0
        self.query1 = query1
        self.n_animation_frame = n_animation_frame

        # Get reduction on last-of-animation state
        self.reduce(query=query1, save_rd=False, **kwargs)
        all_temporal_normal_vectors = [self.normal_vectors]
        all_temporal_initial_points_of_normal_vectors = [
            self.initial_points_of_normal_vectors
        ]

        # Get reduction on beginning-of-animation state
        self.reduce(query=query0, save_rd=False, **kwargs)
        all_temporal_normal_vectors.insert(0, self.normal_vectors)
        all_temporal_initial_points_of_normal_vectors.insert(
            0, self.initial_points_of_normal_vectors
        )

        # Devide into segments
        self.setIntervals(
            all_temporal_normal_vectors,
            all_temporal_initial_points_of_normal_vectors,
            n_animation_frame,
            animation_option,
        )

        self.rds = []
        # Fo each basis in animation
        for t, components in enumerate(self.all_temporal_components):
            # For each basis vector of basises
            rd = np.zeros_like(self.rd)
            for i, c in enumerate(components):
                # Project data to each basis
                rd[:, i] = self.getDF(query0).to_numpy() @ c
            rd = pd.DataFrame(
                data=rd,
                columns=["col{}".format(i) for i in range(self.rd.shape[1])],
            )
            # Add column describing time stamp
            rd["t"] = t
            # Add column describing whether datapoint belongs to filters 0 and 1
            rd["query0"] = np.where(
                rd.index.isin(self.getDFIndices(self.query0)), True, False
            )
            rd["query1"] = np.where(
                rd.index.isin(self.getDFIndices(self.query1)), True, False
            )
            self.rds.append(rd)

    def setIntervals(
        self,
        all_temporal_normal_vectors,
        all_temporal_initial_points_of_normal_vectors,
        n_segments,
        animation_option="linear",
    ):
        if animation_option == "linear":
            self.all_temporal_normal_vectors = np.linspace(
                all_temporal_normal_vectors[0],
                all_temporal_normal_vectors[1],
                num=n_segments,
                endpoint=True,
            )
            self.all_temporal_initial_points_of_normal_vectors = np.zeros_like(
                self.all_temporal_normal_vectors
            )

        # Calculate components based on Normal Vector Movement
        all_temporal_components = [
            (normal_vectors - all_temporal_normal_vectors[0]) + self.components
            for normal_vectors in self.all_temporal_normal_vectors
        ]

        # Store components as orthogonal bases
        self.all_temporal_components = [
            np.array(doGramSchmidt(components))
            for components in all_temporal_components
        ]

    # Store normal vector representing plane formed by principal components
    # When dim>2, they are called: normal space/affine subspace/normal hyperplane
    # May need error check!
    def setNormalVector(self):
        random_coefficients = np.random.rand(len(self.components[0, :]))
        A = np.vstack((self.components.copy(), random_coefficients))
        y = np.zeros_like(A[:, 0])
        y[-1] = np.random.rand(1)

        # Solve for
        normal_vectors = np.linalg.lstsq(A, y, rcond=None)[0]
        # Normalize (maybe don't have to)
        self.normal_vectors = normal_vectors / np.linalg.norm(normal_vectors)
