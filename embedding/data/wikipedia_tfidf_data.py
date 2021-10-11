#!/usr/bin/python
# -*- coding: utf-8 -*-
""" Json Document Data

Create distance matrix from documents in json format

"""

from os import path
from os.path import join
import logging
import json
import math

import pandas as pd
import numpy as np
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer

from embedding.data.parent_data import ParentData


log = logging.getLogger(__name__)


class WikipediaTFIDFData(ParentData):
    """Wikipedia tf-idf Data

    Distance matrix from documents in json format

    Attributes:
        data_key (str): An identifying name to distinguish this data from other data.
        df (DataFrame): M*M Distance matrix. M = Number of documents.
        color (ndarray): Color information for each object.
    """

    def __init__(self, *args, docs_num_threshold=1000, words_freq_threshold_bottom=100, words_freq_threshold_top=150) -> None:
        """ Initialize """

        super().__init__(*args)
        self.words_freq_threshold_bottom = words_freq_threshold_bottom
        self.words_freq_threshold_top = words_freq_threshold_top
        self.docs_num_threshold = docs_num_threshold
        self.set_dataframe_and_color(self.data_path)
        self.data_key = "wikipedia_tfidf"

    def set_dataframe_and_color(self, root: str) -> None:
        """ Set DataFrame and Color
        Args:
            root (str): Root directory for dataset.
        """

        # if not path.exists(join(self.cache_path, "wikipedia_tfidf.csv")):
        if True:
            data_root = join(root, "private/wikics")
            self.make_dataset(data_root)

        self.df = pd.read_csv(
                join(self.cache_path, "wikipedia_tfidf.csv"),
                )

        self.color = np.array([1] * self.df.shape[0])

    def make_dataset(self, data_root: str) -> None:
        """ Make Dataset
        Make dataset from json files and save it as csv.
        
        Args:
            data_root: Root directory for document json files.
        """


        with open(f"{data_root}/metadata.json") as f:
            json_obj = json.load(f)
            nodes = json_obj["nodes"]

        docs = []

        for node in nodes:
            docs.append(" ".join(node["tokens"]))
            if len(docs) == self.docs_num_threshold:
                break

        # nltk settings
        nltk.download('punkt')
        stemmer = PorterStemmer()
        cv = CountVectorizer()
        texts = []  # A list of tokenized texts separated by half-width characters

        for doc in docs:
            tokenized = word_tokenize(doc)

            for i in range(len(tokenized)):
                tokenized[i] = stemmer.stem(tokenized[i])

            text = " ".join(tokenized)
            texts.append(text)

        # Vectorize
        bows = cv.fit_transform(texts).toarray()

        # Filtering by word frequency
        words_freq = np.sum(bows, axis=0)
        out_of_range_words_indices = np.argwhere(
                (words_freq < self.words_freq_threshold_bottom) | (self.words_freq_threshold_top <= words_freq)
            )
        bows = np.delete(bows, np.ravel(out_of_range_words_indices), 1)
        zero_indices = np.argwhere(np.all(bows == 0, axis=1))
        bows = np.delete(bows, np.ravel(zero_indices), 0)

        # Weighting by tf-idf
        tf = bows / np.repeat(np.sum(bows, axis=1).reshape(-1, 1), bows.shape[1], axis=1)

        num_documents = bows.shape[0]
        idf = np.sum(np.where(bows > 0, 1, 0), axis=0)
        idf = np.array(list(map(lambda x: math.log(num_documents/(x+1)), idf)))

        weighted_bows = tf * np.repeat(idf.reshape(1, -1), bows.shape[0], axis=0)

        df = pd.DataFrame(weighted_bows)
        df.to_csv(join(self.cache_path, "wikipedia_tfidf.csv"), index=False)
