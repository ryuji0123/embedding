from embedding.embedder import Laplacian_EigenmapsEmbedder
from embedding.data import chooseData


def test_l_em():
    l_em = Laplacian_EigenmapsEmbedder(chooseData("pokemon"))
    l_em.embed(use_cache=True)
    assert l_em.em.shape == (801, 2)


if __name__ == "__main__":
    test_l_em()
