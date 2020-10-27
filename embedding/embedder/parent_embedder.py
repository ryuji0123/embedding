import pandas as pd

from abc import ABCMeta, abstractmethod


class ParentEmbedder(metaclass=ABCMeta):

    def __init__(self, data):
        self.data = data
        self.df = data.df

    def embed(self, use_cache=False, **kwargs):
        if self.data.exists(self.class_key) and use_cache:
            self.em = self.data.getResult(self.class_key)
        else:
            self.execEmbed(**kwargs)
            self.em = pd.DataFrame(
                    data=self.em,
                    columns=[str(i) for i in range(self.em.shape[1])],
                    )
            self.data.save(self.class_key, self.em)

    @abstractmethod
    def execEmbed(self):
        pass
