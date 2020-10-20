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
            self.rd = pd.DataFrame(
                    data=self.rd,
                    columns=["{}".format(i) for i in range(self.rd.shape[1])],
                    )
            self.data.save(self.class_key, self.rd)

    # Store normal vector representing plane formed by principal components
    # When dim>2, they are called: normal space/affine subspace
    # May need error check!
    def set_normal_vector(self):
        n_vec = self.cmp[0, :]
        for i in range(1, self.cmp.shape[0]):
            n_vec = np.cross(n_vec, self.cmp[i, :])
        # Normalize (maybe don't have to)
        self.n_vec = n_vec / np.linalg.norm(n_vec)

    @abstractmethod
    def execReduce(self):
        pass
