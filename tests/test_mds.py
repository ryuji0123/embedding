from embedding.reducer import MDSReducer
from embedding.data import chooseData


def test_mds():
    mds = MDSReducer(chooseData("pokemon"))
    mds.reduce()
    assert mds.rd.shape == (801, 2)


if __name__ == "__main__":
    test_mds()
