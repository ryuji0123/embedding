from embedding.embedder import IsomapEmbedder
from embedding.reducer import MDSReducer
from embedding.data import chooseData


def test_isomap_and_mds():
    isomap = IsomapEmbedder(chooseData("pokemon"))
    isomap.embed(dim=2)
    assert isomap.em.shape == (801, 2)

    mds = MDSReducer(isomap.data, isomap)
    assert mds.df.shape == (801, 2)
    mds.reduce(dim=2)
    assert mds.rd.shape == (801, 2)


if __name__ == "__main__":
    test_isomap_and_mds()
