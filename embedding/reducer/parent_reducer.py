import pandas as pd
from abc import ABCMeta, abstractmethod


class ParentReducer(metaclass=ABCMeta):
    def __init__(self, file_path, cols, file_sep=','):
        self.df = pd.read_csv(file_path, usecols=cols, sep=file_sep)

    @abstractmethod
    def reduce(self):
        pass
