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

from embedding.data.parent_data import ParentData


log = logging.getLogger(__name__)


class JsonDocumentData(ParentData):
    """Json Document Data

    Distance matrix from documents in json format

    Attributes:
        data_key (str): An identifying name to distinguish this data from other data.
        df (DataFrame): M*M Distance matrix. M = Number of documents. 
        color (ndarray): Color information for each object.
    """

    def __init__(self, *args) -> None:
        """ Initialize """

        super().__init__(*args)
        self.setDataFrameAndColor(self.data_path)
        self.data_key = "json_document"


    def setDataFrameAndColor(self, root: str) -> None:
        """ Set DataFrame and Color
        Args:
            root (str): Root directory for dataset. 
        """

        if not path.exists(join(self.cache_path, "json_document.csv")):
            data_root = join(root, "private/blog.barracuda.com_en_2021-03-11_0")
            self.make_dataset(data_root)

        self.df = pd.read_csv(
                join(self.cache_path, "json_document.csv"),
                )

        self.color = np.array([1] * self.df.shape[0])


    def make_dataset(self, data_root: str) -> None:
        """ Make Dataset
        Make dataset from json files and save it as csv.

        Args:
            data_root: Root directory for document json files.
        """

        json_paths = glob.glob(f"{data_root}/**/*.json")

        # nltk settings
        nltk.download('punkt')
        stemmer = PorterStemmer()
        cv = CountVectorizer()
        texts = [] # A list of tokenized texts separated by half-width characters

        for json_path in json_paths:
            with open(json_path) as f:
                json_obj = json.load(f)
                body = json_obj["body"]

                soup = BeautifulSoup(body, "html.parser")
                for script in soup(["script", "style"]):
                    script.decompose()
                text = soup.get_text()
                tokenized = word_tokenize(text)

                for i in range(len(tokenized)):
                    tokenized[i] = stemmer.stem(tokenized[i])

                text = " ".join(tokenized)
                texts.append(text)

        # Vectorize
        bows = cv.fit_transform(texts).toarray()

        # Filtering by word frequency
        words_freq_threshold = 1000
        words_freq = np.sum(bows, axis=0)
        frequent_words_indices = np.argwhere(words_freq >= words_freq_threshold )
        bows = np.delete(bows, np.ravel(frequent_words_indices), 1)

        # Weighting by tf-idf
        tf = bows / np.repeat(np.sum(bows, axis=1).reshape(-1, 1), bows.shape[1], axis=1)

        num_documents = len(json_paths)
        idf = np.sum(np.where(bows > 0, 1, 0), axis=0)
        idf = np.array(list(map(lambda x: math.log(num_documents/x), idf)))

        weighted_bows = tf * np.repeat(idf.reshape(1, -1), bows.shape[0], axis=0)

        # Calculate distance matrix
        dist_mat = squareform(pdist(weighted_bows, metric='cosine'))

        df = pd.DataFrame(dist_mat)
        df.to_csv(join(self.cache_path, "json_document.csv"), index=False)


