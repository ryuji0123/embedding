from abc import ABCMeta, abstractmethod


class ParentData(metaclass=ABCMeta):
    @abstractmethod
    def getDataFrame():
        pass
