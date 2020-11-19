from embedding.embedder import TSNEEmbedder
from embedding.data import chooseData


def test_t_sne_pokemon():
    tsne = TSNEEmbedder(chooseData("pokemon"))
    tsne.embed(use_cache=True)
    assert tsne.em.shape == (801, 2)


if __name__ == "__main__":
    test_t_sne_pokemon()
