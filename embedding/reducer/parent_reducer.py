import numpy as np
import pandas as pd

from abc import ABCMeta, abstractmethod

from embedding.data import ParentData


class ParentReducer(metaclass=ABCMeta):
    def __init__(self, data, df=None):
        if not isinstance(data, ParentData):
            raise ValueError(f"{type(data)} should inherit {ParentData}")
        if df is None:
            self.data = data
            self.df = data.df
        elif isinstance(df, np.ndarray):
            self.data = data
            self.df = pd.DataFrame(df)
        else:
            raise ValueError(f"{type(df)} should be None or {type(np.ndarray)}")

    def reduce(self):
        if self.data.exists(self.class_key):
            self.rd = self.data.getResult(self.class_key)
        else:
            self.execReduce()
            self.data.save(self.class_key, self.rd)

    @abstractmethod
    def execReduce(self):
        pass
