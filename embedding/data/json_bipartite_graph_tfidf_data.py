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
import glob

import pandas as pd
import numpy as np
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer as WNL
from sklearn.feature_extraction.text import CountVectorizer
import gensim.downloader
from bs4 import BeautifulSoup

from embedding.data.parent_data import ParentData


log = logging.getLogger(__name__)


class JsonBipartiteGraphTFIDFData(ParentData):
    """Wikipedia tf-idf Data

    Distance matrix from documents in json format

    Attributes:
        data_key (str): An identifying name to distinguish this data from other data.
        df (DataFrame): M*M Distance matrix. M = Number of documents.
        color (ndarray): Color information for each object.
    """

    def __init__(self, *args, docs_num_threshold: int = 100000, words_min_document_freq: float = 0.25, words_filter_target_word: str = "computer", words_filter_topn: int = 1000) -> None:
        """ Initialize 
        
        Args:
            docs_num_threshold (int): Number of documents to embed
            words_min_document_freq (float): The minimum document frequency of a word
            words_filter_target_word (str): The word to use as the basis for word filtering. Words in the neighborhood of this word will be whitelisted.
            words_filter_topn (int): Number of the neighborhood of the target word to be searched for by word filtering
        
        """

        super().__init__(*args)
        self.words_min_document_freq = words_min_document_freq
        self.docs_num_threshold = docs_num_threshold
        self.words_filter_target_word = words_filter_target_word
        self.words_filter_topn = words_filter_topn
        self.set_dataframe_and_color(self.data_path)
        self.data_key = "json_bipartite_graph_tfidf"

    def set_dataframe_and_color(self, root: str) -> None:
        """ Set DataFrame and Color
        Args:
            root (str): Root directory for dataset.
        """

        if not path.exists(join(self.cache_path, "json_bipartite_graph_tfidf.csv")):
            data_root = join(root, "private/json_docs_en")
            self.make_dataset(data_root)

        self.df = pd.read_csv(
                join(self.cache_path, "json_bipartite_graph_tfidf.csv"),
                )

        self.color = np.array([1] * self.df.shape[0])

    def make_dataset(self, data_root: str) -> None:
        """ Make Dataset
        Make dataset from json files and save it as csv.
        
        Args:
            data_root: Root directory for document json files.
        """


        json_paths = glob.glob(f"{data_root}/**/*.json", recursive=True)
        json_paths = json_paths[:self.docs_num_threshold]
        log.info(f"num of docs: {len(json_paths)}")

        # nltk settings
        nltk.download('punkt')
        nltk.download('stopwords')
        nltk.download('wordnet')
        stop_words = nltk.corpus.stopwords.words('english')
        symbols = ["'", '"', ':', ';', '.', ',', '-', '!', '?', "'s"]
        stop_words += symbols

        glove_vectors = gensim.downloader.load('glove-wiki-gigaword-100')
        white_list = [t[0] for t in glove_vectors.most_similar(self.words_filter_target_word, topn=self.words_filter_topn)]

        stemmer = WNL()
        texts = []  # A list of tokenized texts separated by half-width characters

        for json_path in json_paths:
            with open(json_path) as f:
                json_obj = json.load(f)
                body = json_obj["body"]

                soup = BeautifulSoup(body, "html.parser")
                for script in soup(["script", "style"]):
                    script.decompose()
                doc = soup.get_text()
                tokenized = [stemmer.lemmatize(word) for word in word_tokenize(doc) if word in white_list and word not in stop_words]

                text = " ".join(tokenized)
                texts.append(text)

        # Vectorize
        cv = CountVectorizer(min_df=self.words_min_document_freq)
        bows = cv.fit_transform(texts).toarray()

        # Remove zero vectors
        zero_indices = np.argwhere(np.all(bows == 0, axis=1))
        bows = np.delete(bows, np.ravel(zero_indices), 0)

        # Weighting by tf-idf
        tf = bows / np.repeat(np.sum(bows, axis=1).reshape(-1, 1), bows.shape[1], axis=1)

        num_documents = bows.shape[0]
        idf = np.sum(np.where(bows > 0, 1, 0), axis=0)
        idf = np.array(list(map(lambda x: math.log(num_documents/(x+1)), idf)))

        weighted_bows = tf * np.repeat(idf.reshape(1, -1), bows.shape[0], axis=0)

        df = pd.DataFrame(weighted_bows)
        log.info(df.shape)
        df.columns = cv.get_feature_names()
        df.to_csv(join(self.cache_path, "json_bipartite_graph_tfidf.csv"), index=False)
