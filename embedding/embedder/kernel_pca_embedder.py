from sklearn.decomposition import KernelPCA

from embedding.embedder.parent_embedder import ParentEmbedder


class KernelPCAEmbedder(ParentEmbedder):
    def __init__(self, *args):
        super(KernelPCAEmbedder, self).__init__(*args)
        self.class_key = "kernel_pca_embedder"

    def execEmbed(self, dim=2):
        self.em = KernelPCA(n_components=dim, kernel='linear').fit_transform(self.df)
