from embedding.reducer.ica_reducer import ICAReducer
from embedding.reducer.mds_reducer import MDSReducer
from embedding.reducer.nmf_reducer import NMFReducer
from embedding.reducer.pca_reducer import PCAReducer


REDUCERS_REF = {
        "ica": ICAReducer,
        "mds": MDSReducer,
        "nmf": NMFReducer,
        "pca": PCAReducer,
        }
