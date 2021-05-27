# -*- coding: utf-8 -*-
"""Utils
Utility functions for embedding.
"""

from pandas import DataFrame

from embedding.data import chooseData
from embedding.embedder import chooseEmbedder


def get_embedding(data_key: str, embedder_key: str) -> DataFrame:
    """Get embedding
    Args:
        data_key (str): Name of dataset.
        embedder_key (str): Name of embedder.
    Returns:
        DataFrame: Embedding results.
    """

    sc_data = chooseData(data_key)
    embedder = chooseEmbedder(embedder_key, sc_data)
    embedder.embed()

    return embedder.em