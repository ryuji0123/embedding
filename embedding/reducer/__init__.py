from embedding.reducer.ica_reducer import ICAReducer
from embedding.reducer.mds_reducer import MDSReducer
from embedding.reducer.nmf_reducer import NMFReducer
from embedding.reducer.pca_reducer import PCAReducer
from embedding.reducer.parent_reducer import ParentReducer

reducers_ref = {
        "ica": ICAReducer,
        "mds": MDSReducer,
        "nmf": NMFReducer,
        "pca": PCAReducer,
        }

__all__ = [
        "ICAReducer", "MDSReducer", "NMFReducer", "ParentReducer", "PCAReducer", "reducers_ref"
        ]
