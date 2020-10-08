from abc import ABCMeta, abstractmethod


class ParentEmbedder(metaclass=ABCMeta):
    def __init__(self, df):
        self.df = df

    @abstractmethod
    def embed(self):
        pass
