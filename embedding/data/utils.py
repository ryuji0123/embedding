from embedding.data import DATA_PATH, CACHE_PATH, PokemonData, ArtificialData, ScurveData, SwrollData

ref = {
        "pokemon": PokemonData,
        "artificial": ArtificialData,
        "scurve": ScurveData,
        "swissroll": SwrollData
        }


def chooseData(key):
    if key not in ref:
        raise NotImplementedError(f"{key} is not supported")
    return ref[key](DATA_PATH, CACHE_PATH)
