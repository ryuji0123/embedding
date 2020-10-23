from embedding.embedder import N_MDSEmbedder
from embedding.data import chooseData


def test_n_mds():
    n_mds = N_MDSEmbedder(chooseData("pokemon"))
    n_mds.embed(use_cache=True)
    assert n_mds.em.shape == (801, 2)


if __name__ == "__main__":
    test_n_mds()
