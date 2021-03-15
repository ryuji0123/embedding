from embedding.embedder.const import EMBEDDERS_REF


def chooseEmbedder(key, data):
    if key not in EMBEDDERS_REF:
        raise NotImplementedError(f"{key} is not supported")
    return EMBEDDERS_REF[key](data)
