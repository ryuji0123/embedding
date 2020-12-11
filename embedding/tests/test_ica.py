from embedding.reducer import ICAReducer
from embedding.data import chooseData


def test_ica():
    ica = ICAReducer(chooseData("pokemon"))
    ica.reduce(use_cache=True)
    assert ica.rd.shape == (801, 2)


if __name__ == "__main__":
    test_ica()
