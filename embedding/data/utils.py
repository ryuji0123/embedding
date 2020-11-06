from embedding.data.const import CACHE_PATH, DATA_PATH, DATA_REF


def chooseData(key):
    if key not in DATA_REF:
        raise NotImplementedError(f"{key} is not supported")
    return DATA_REF[key](DATA_PATH, CACHE_PATH)
