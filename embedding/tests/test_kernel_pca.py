import logging

from embedding.data import chooseData
from embedding.embedder import KernelPCAEmbedder


def test_kernel_pca():
    kernel_pca = KernelPCAEmbedder(chooseData("pokemon"))
    kernel_pca.embed(use_cache=True)
    assert kernel_pca.em.shape == (801, 2)


def test_kernel_pca_basic_cluster():
    logging.info("start kernel_pca test using basic_cluster data")
    logging.info("start generating data")
    kernel_pca = KernelPCAEmbedder(chooseData("basic_cluster"))
    logging.info("start embedding")
    kernel_pca.embed(use_cache=True)
    logging.info("finish all process")
    assert kernel_pca.em.shape == (801, 2)


def test_kernel_pca_clustered_scurve():
    logging.info("start kernel_pca test using clustered-scurve data")
    logging.info("start generating data")
    kernel_pca = KernelPCAEmbedder(chooseData("clustered_scurve"))
    logging.info("start embedding")
    kernel_pca.embed(use_cache=True)
    logging.info("finish all process")
    assert kernel_pca.em.shape == (1000, 2)


def test_kernel_pca_clustered_swissroll():
    logging.info("start kernel_pca test using clustered-swissroll data")
    logging.info("start generating data")
    kernel_pca = KernelPCAEmbedder(chooseData("clustered_swissroll"))
    logging.info("start embedding")
    kernel_pca.embed(use_cache=True)
    logging.info("finish all process")
    assert kernel_pca.em.shape == (1000, 2)


if __name__ == "__main__":
    test_kernel_pca()
    test_kernel_pca_basic_cluster()
    test_kernel_pca_clustered_scurve()
    test_kernel_pca_clustered_swissroll()
