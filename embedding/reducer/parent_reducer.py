from abc import ABCMeta, abstractmethod


class ParentReducer(metaclass=ABCMeta):
    def __init__(self, data):
        self.data = data
        self.df = data.df

    def reduce(self):
        if self.data.exists(self.class_key):
            self.rd = self.data.getResult(self.class_key)
        else:
            self.execReduce()
            self.data.save(self.class_key, self.rd)

    @abstractmethod
    def execReduce(self):
        pass
