import logging

from embedding.embedder import TSNEEmbedder
from embedding.data import chooseData


def test_t_sne_pokemon():
    tsne = TSNEEmbedder(chooseData("pokemon"))
    tsne.embed(use_cache=True)
    assert tsne.em.shape == (801, 2)


def test_t_sne_artificial():
    logging.info("start t-sne test using artificial data")
    logging.info("start generating data")
    tsne = TSNEEmbedder(chooseData("artificial"))
    logging.info("start embedding")
    tsne.embed(use_cache=True)
    logging.info("finish all process")
    assert tsne.em.shape == (801, 2)


if __name__ == "__main__":
    test_t_sne_pokemon()
    test_t_sne_artificial()
