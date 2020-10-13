from sklearn.manifold import MDS

from embedding.reducer import ParentReducer

class MDSReducer(ParentReducer):
    def __init__(self, *args):
        super(MDSReducer, self).__init__(*args)
        self.class_key = "mds_reducer"

    def execReduce(self, dim=2):
        self.rd = MDS(n_components=dim).fit_transform(self.df)
