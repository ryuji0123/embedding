from abc import ABCMeta, abstractmethod


class ParentEmbedder(metaclass=ABCMeta):
    def __init__(self, data):
        self.data = data
        self.df = data.df

    def embed(self):
        if self.data.exists(self.class_key):
            print(f'{self.class_key} result exists')
            self.em = self.data.get(self.class_key)
        else:
            self.execEmbed()
            self.data.save(self.class_key, self.em)

    @abstractmethod
    def execEmbed(self):
        pass
