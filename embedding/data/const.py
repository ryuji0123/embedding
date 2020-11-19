from os import path
from os.path import join

from embedding.data.pokemon_data import PokemonData
from embedding.data.basic_cluster_data import BasicClusterData
from embedding.data.scurve_data import ScurveData
from embedding.data.swissroll_data import SwrollData
from embedding.data.clustered_swissroll_data import ClusteredSwissrollData
from embedding.data.clustered_scurve_data import ClusteredScurveData


DATA_PATH = join(path.sep, 'workspace', 'data')
CACHE_PATH = join(path.sep, 'workspace', 'cache')
DATA_REF = {
        "pokemon": PokemonData,
        "basic_cluster": BasicClusterData,
        "scurve": ScurveData,
        "swissroll": SwrollData,
        "clustered_scurve": ClusteredScurveData,
        "clustered_swissroll": ClusteredSwissrollData
        }
