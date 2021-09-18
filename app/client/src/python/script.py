#!/usr/bin/python
# -*- coding: utf-8 -*-
""" Script for pyodide

This script is run by pyodide

"""

import pandas as pd
from sklearn.manifold import MDS
from sklearn.decomposition import PCA
import numpy as np


embedding = 'EMBEDDING' # embedding data passed by frontend
reducer_name = 'REDUCER' # reducer's name passed by frontend 


def main():
    """ Main
    
    returns:
        str: Dimensionally reduced data in JSON string format

    """

    df = pd.read_json(embedding)

    # reducer
    if reducer_name == "mds":
        rd = MDS(n_components=2).fit_transform(df)
    else: # pca (default)
        transformer = PCA(n_components=2).fit(df)
        components = transformer.components_
        rd = np.empty((len(df), 2))
        for i, c in enumerate(components):
            rd[:, i] = df.to_numpy() @ c / (c @ c)
    
    rd = pd.DataFrame(
        data=rd,
        columns=["col{}".format(i) for i in range(rd.shape[1])],
    )

    return rd.to_json()


main()