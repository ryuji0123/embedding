from embedding.embedder import IsomapEmbedder
from embedding.data import chooseData


def test_isomap():
    isomap = IsomapEmbedder(chooseData("pokemon"))
    isomap.embed()
    assert isomap.em.shape == (801, 2)


if __name__ == "__main__":
    test_isomap()
