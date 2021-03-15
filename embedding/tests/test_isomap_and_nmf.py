from embedding.embedder import IsomapEmbedder
from embedding.reducer import NMFReducer
from embedding.data import chooseData


def test_isomap_and_nmf():
    isomap = IsomapEmbedder(chooseData("pokemon"))
    isomap.embed(dim=2, use_cache=True)
    assert isomap.em.shape == (801, 2)

    nmf = NMFReducer(isomap.data, isomap)
    assert nmf.df.shape == (801, 2)
    nmf.reduce(dim=2, use_cache=True)
    assert nmf.rd.shape == (801, 2)


if __name__ == "__main__":
    test_isomap_and_nmf()
