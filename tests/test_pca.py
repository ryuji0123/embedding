from embedding.reducer import PCAReducer
from embedding.data import chooseData


def test_pca():
    pca = PCAReducer(chooseData("pokemon"))
    pca.reduce()
    assert pca.rd.shape == (801, 2)


if __name__ == "__main__":
    test_pca()
