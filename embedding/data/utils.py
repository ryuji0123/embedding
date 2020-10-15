from embedding.data import DATA_PATH, RESULT_PATH, PokemonData

ref = {
        "pokemon": PokemonData
        }


def chooseData(key):
    if key not in ref:
        raise NotImplementedError(f"{key} is not supported")
    return ref[key](DATA_PATH, RESULT_PATH)
