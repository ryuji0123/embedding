from embedding.data import chooseData
from embedding.embedder import KernelPCAEmbedder


def test_kernel_pca():
    kernel_pca = KernelPCAEmbedder(chooseData("pokemon"))
    kernel_pca.embed()
    assert kernel_pca.em.shape == (801, 2)


if __name__ == "__main__":
    test_kernel_pca()
