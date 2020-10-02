import pandas as pd
from abc import ABCMeta, abstractmethod
from os import path
from os.path import join

DATA_PATH = join(path.sep, 'workspace', 'data')


class Data(metaclass=ABCMeta):
    def __init__(self, file_path, cols, file_sep=','):
        self.df = pd.read_csv(file_path, usecols=cols, sep=file_sep)

    @abstractmethod
    def reduction(self):
        pass
