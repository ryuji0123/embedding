import os
import pandas as pd

from abc import ABCMeta, abstractmethod
from os import path
from os.path import join


class ParentData(metaclass=ABCMeta):
    def __init__(self, data_path, cache_path):
        self.data_path = data_path
        self.cache_path = cache_path
        if not path.exists(cache_path):
            os.makedirs(cache_path)

    def getResult(self, class_key):
        return pd.read_csv(join(self.cache_path, f"{class_key}_{self.data_key}.csv"))

    def exists(self, class_key):
        return path.exists(join(self.cache_path, f"{class_key}_{self.data_key}.csv"))

    def save(self, class_key, em):
        em.to_csv(join(self.cache_path, f"{class_key}_{self.data_key}.csv"), index=False)

    @abstractmethod
    def setDataFrameAndColor(self):
        pass
