import pandas as pd

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
            raise ValueError(f"{type(embedder)} should be None or {type(ParentEmbedder)}")

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

    @abstractmethod
    def execReduce(self):
        pass
