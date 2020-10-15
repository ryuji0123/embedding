from embedding.embedder import TSNEEmbedder
from embedding.data import chooseData


def test_t_sne():
    tsne = TSNEEmbedder(chooseData("artificial"))
    tsne2 = TSNEEmbedder(chooseData("pokemon"))
    print(tsne.df)
    print(tsne2.df)
    tsne.embed()
    tsne2.embed()
    print(tsne.em)
    print(tsne2.em)
    assert tsne.em.shape == (801, 2)


if __name__ == "__main__":
    test_t_sne()
