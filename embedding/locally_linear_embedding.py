from embedding import Data
from sklearn.manifold import LocallyLinearEmbedding

class LLEData(Data):
    def reduction(self, dim=2):
        # variable name 'embedding' in accordance with source code example
        embedding = LocallyLinearEmbedding(n_components=2)
        self.em = embedding.fit_transform(self.df)