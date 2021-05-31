from os import path
from os.path import join
import logging
import glob
import json

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
    def __init__(self, *args) -> None:
        super().__init__(*args)
        self.setDataFrameAndColor(self.data_path)
        self.data_key = "json_document"


    def setDataFrameAndColor(self, root: str) -> None:
        """
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
        """
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

        # Calculate distance matrix
        dist_mat = squareform(pdist(bows, metric='cosine'))

        df = pd.DataFrame(dist_mat)
        df.to_csv(join(self.cache_path, "json_document.csv"), index=False)


