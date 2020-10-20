from embedding.embedder import TSNEEmbedder
from embedding.data import chooseData


def test_t_sne_pokemon():
    tsne = TSNEEmbedder(chooseData("pokemon"))
    tsne.embed()
    assert tsne.em.shape == (801, 2)


def test_t_sne_artificial():
    tsne = TSNEEmbedder(chooseData("artificial"))
    tsne.embed()
    assert tsne.em.shape == (801, 2)


if __name__ == "__main__":
    test_t_sne_pokemon()
    test_t_sne_artificial()
