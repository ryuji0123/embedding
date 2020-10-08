from os.path import join

from embedding import DATA_PATH, KernelPCAData


def test_kernel_pca():
    tsne = KernelPCAData(join(DATA_PATH, 'pokemon.csv.gz'), cols=['hp', 'attack', 'defense', 'sp_attack', 'sp_defense', 'speed'])
    tsne.reduction()
    assert tsne.em.shape == (801, 2)


if __name__ == "__main__":
    test_kernel_pca()
