from embedding.data import DATA_PATH, PokemonData

ref = {
        "pokemon": PokemonData
        }


def chooseData(key):
    if key not in ref:
        raise NotImplementedError(f"{key} is not supported")
    return ref[key].getDataFrame(DATA_PATH)
