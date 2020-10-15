from embedding.embedder import TSNEEmbedder
from embedding.data import chooseData


def test_t_sne_artificial():
    tsne = TSNEEmbedder(chooseData("artificial"))
    print(tsne.df)
    tsne.embed()
    print(tsne.em)
    assert tsne.em.shape == (801, 2)


if __name__ == "__main__":
    test_t_sne_artificial()
