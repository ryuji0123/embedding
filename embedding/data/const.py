from os import path
from os.path import join

from embedding.data.pokemon_data import PokemonData
from embedding.data.artificial_data import ArtificialData
from embedding.data.scurve_data import ScurveData
from embedding.data.swissroll_data import SwrollData


DATA_PATH = join(path.sep, 'workspace', 'data')
CACHE_PATH = join(path.sep, 'workspace', 'cache')
DATA_REF = {
        "pokemon": PokemonData,
        "artificial": ArtificialData,
        "scurve": ScurveData,
        "swissroll": SwrollData
        }
