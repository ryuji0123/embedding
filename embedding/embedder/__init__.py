from embedding.embedder.isomap_embedder import IsomapEmbedder
from embedding.embedder.kernel_pca_embedder import KernelPCAEmbedder
from embedding.embedder.laplacian_eigenmaps_embedder import Laplacian_EigenmapsEmbedder
from embedding.embedder.locally_linear_embedder import LocallyLinearEmbedder
from embedding.embedder.n_mds_embedder import N_MDSEmbedder
from embedding.embedder.parent_embedder import ParentEmbedder
from embedding.embedder.tsne_embedder import TSNEEmbedder


embedders_ref = {
        "isomap": IsomapEmbedder,
        "kernel_pca": KernelPCAEmbedder,
        "laplacian_eigenmaps": Laplacian_EigenmapsEmbedder,
        "locally_linear": LocallyLinearEmbedder,
        "n_mds": N_MDSEmbedder,
        "t_sne": TSNEEmbedder,
        }

__all__ = [
        "IsomapEmbedder", "KernelPCAEmbedder", "Laplacian_EigenmapsEmbedder", "LocallyLinearEmbedder",
        "N_MDSEmbedder", "ParentEmbedder", "TSNEEmbedder", "embedders_ref",
        ]
