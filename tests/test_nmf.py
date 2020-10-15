from embedding.reducer import NMFReducer
from embedding.data import chooseData


def test_nmf():
    nmf = NMFReducer(chooseData("pokemon"))
    nmf.reduce()
    assert nmf.rd.shape == (801, 2)


if __name__ == "__main__":
    test_nmf()
