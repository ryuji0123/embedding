from embedding.data import chooseData
from embedding.embedder import chooseEmbedder


def getEmbedding(data_key, embedder_key):
    sc_data = chooseData(data_key)
    embedder = chooseEmbedder(embedder_key, sc_data)
    embedder.embed()

    return embedder.em