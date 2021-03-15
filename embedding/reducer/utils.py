from embedding.reducer.const import REDUCERS_REF


def chooseReducer(key, data, embedder=None):
    if key not in REDUCERS_REF:
        raise NotImplementedError(f"{key} is not supported")
    return REDUCERS_REF[key](data, embedder)
