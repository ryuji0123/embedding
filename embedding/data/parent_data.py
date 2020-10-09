import os
import pandas as pd

from abc import ABCMeta, abstractmethod
from os import path
from os.path import join


class ParentData(metaclass=ABCMeta):
    def __init__(self, data_path, result_path):
        self.data_path = data_path
        self.result_path = result_path
        if not path.exists(result_path):
            os.makedirs(result_path)

    def get(self, class_key):
        return pd.read_csv(join(self.result_path, f"{class_key}_{self.data_key}.csv"))

    def exists(self, class_key):
        return path.exists(join(self.result_path, f"{class_key}_{self.data_key}.csv"))

    def save(self, class_key, em):
        df = pd.DataFrame(em)
        df.to_csv(join(self.result_path, f"{class_key}_{self.data_key}.csv"), index=False)

    @abstractmethod
    def getDataFrame():
        pass
