#!/usr/bin/python
# -*- coding: utf-8 -*-
""" Json Document Data

Create distance matrix from documents in json format

"""

from os import path
from os.path import join
import logging
import glob
import json
import math

import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform
import tensorflow_datasets as tfds

from embedding.data.parent_data import ParentData


log = logging.getLogger(__name__)


class WikipediaData(ParentData):
    """Wikipedia Data

    Distance matrix from documents in json format

    Attributes:
        data_key (str): An identifying name to distinguish this data from other data.
        df (DataFrame): M*M Distance matrix. M = Number of documents.
        color (ndarray): Color information for each object.
    """

    def __init__(self, *args) -> None:
        """ Initialize """

        super().__init__(*args)
        self.set_dataframe_and_color(self.data_path)
        self.data_key = "wikipedia"

    def set_dataframe_and_color(self, root: str) -> None:
        """ Set DataFrame and Color
        Args:
            root (str): Root directory for dataset.
        """

        if not path.exists(join(self.cache_path, "wikipedia.csv")):
            self.make_dataset()

        self.df = pd.read_csv(
                join(self.cache_path, "wikipedia.csv"),
                )

        self.color = np.array([1] * self.df.shape[0])

    def make_dataset(self) -> None:
        """ Make Dataset
        Make dataset from json files and save it as csv.
        """

        dataset = tfds.load('wiki40b/en', split='test')

        docs = []
        start_paragraph = False
        docs_num = 1000
        for wiki in dataset.as_numpy_iterator():
            doc = ""
            for text in wiki['text'].decode().split('\n'):
                if start_paragraph:
                    doc += text.replace('_NEWLINE_', '') # _NEWLINE_は削除
                    start_paragraph = False
                if text == '_START_PARAGRAPH_': # _START_PARAGRAPH_のみ取得
                    start_paragraph = True
            docs.append(doc)
            if len(docs) == docs_num:
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
        words_freq_threshold = docs_num * 0.1
        words_freq = np.sum(bows, axis=0)
        frequent_words_indices = np.argwhere(words_freq >= words_freq_threshold)
        bows = np.delete(bows, np.ravel(frequent_words_indices), 1)

        # Weighting by tf-idf
        tf = bows / np.repeat(np.sum(bows, axis=1).reshape(-1, 1), bows.shape[1], axis=1)

        num_documents = len(docs)
        idf = np.sum(np.where(bows > 0, 1, 0), axis=0)
        idf = np.array(list(map(lambda x: math.log(num_documents/x), idf)))

        weighted_bows = tf * np.repeat(idf.reshape(1, -1), bows.shape[0], axis=0)

        df = pd.DataFrame(weighted_bows)
        df.to_csv(join(self.cache_path, "wikipedia.csv"), index=False)
