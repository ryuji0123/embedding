from embedding.embedder import LocallyLinearEmbedder
from embedding.data import chooseData


def test_locally_linear():
    locally_linear = LocallyLinearEmbedder(chooseData("pokemon"))
    locally_linear.embed()
    assert locally_linear.em.shape == (801, 2)


if __name__ == "__main__":
    test_locally_linear()
