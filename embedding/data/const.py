from os import path
from os.path import join

from embedding.data.pokemon_data import PokemonData
from embedding.data.basic_cluster_data import BasicClusterData
from embedding.data.scurve_data import ScurveData
from embedding.data.swissroll_data import SwrollData
from embedding.data.clustered_swissroll_data import ClusteredSwissrollData
from embedding.data.clustered_scurve_data import ClusteredScurveData
from embedding.data.json_document_data import JsonDocumentData
from embedding.data.json_document_data_w2v import JsonDocumentDataW2V
from embedding.data.json_document_data_longformer import JsonDocumentDataLf
from embedding.data.json_document_data_bert import JsonDocumentDataBERT
from embedding.data.wikipedia_data import WikipediaData
from embedding.data.wikipedia_tfidf_data import WikipediaTFIDFData
from embedding.data.wikipedia_okapi_data import WikipediaOkapiData
from embedding.data.json_bipartite_graph_bow_data import JsonBipartiteGraphBowData
from embedding.data.json_bipartite_graph_tfidf_data import JsonBipartiteGraphTFIDFData
from embedding.data.json_bipartite_graph_okapi_data import JsonBipartiteGraphOkapiData

DATA_PATH = join(path.sep, "workspace", "embedding", "data", "files")
CACHE_PATH = join(path.sep, "workspace", "cache")
DATA_REF = {
        "pokemon": PokemonData,
        "basic_cluster": BasicClusterData,
        "scurve": ScurveData,
        "swissroll": SwrollData,
        "clustered_scurve": ClusteredScurveData,
        "clustered_swissroll": ClusteredSwissrollData,
        "json_document_BoW": JsonDocumentData,
        "json_document_word2vec": JsonDocumentDataW2V,
        "json_document_longformer": JsonDocumentDataLf,
        "json_document_BERT": JsonDocumentDataBERT,
        "wikipedia": WikipediaData,
        "wikipedia_tfidf": WikipediaTFIDFData,
        "wikipedia_okapi": WikipediaOkapiData,
        "json_bipartite_graph_bow": JsonBipartiteGraphBowData,
        "json_bipartite_graph_tfidf": JsonBipartiteGraphTFIDFData,
        "json_bipartite_graph_okapi": JsonBipartiteGraphOkapiData,
        }
