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
from tqdm import tqdm

import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform
import gensim.downloader
import torch
from transformers import LongformerTokenizer, LongformerModel

from embedding.data.parent_data import ParentData


log = logging.getLogger(__name__)


class JsonDocumentDataLf(ParentData):
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
        self.set_dataframe_and_color(self.data_path)
        self.data_key = "json_document_longformer"


    def set_dataframe_and_color(self, root: str) -> None:
        """ Set DataFrame and Color
        Args:
            root (str): Root directory for dataset. 
        """

        # if not path.exists(join(self.cache_path, "json_document.csv")):
        if True:
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

        log.info(f"Making dataset...")
        json_paths = glob.glob(f"{data_root}/**/*.json")

        # nltk settings
        nltk.download('punkt')
        stemmer = PorterStemmer()
        cv = CountVectorizer()
        texts = [] # A list of tokenized texts separated by half-width characters

        # BERT
        feature_matrix = []
        device = torch.device('cuda')
        tokenizer = LongformerTokenizer.from_pretrained('allenai/longformer-base-4096')
        model = LongformerModel.from_pretrained('allenai/longformer-base-4096').to(device)
        for json_path in tqdm(json_paths):
            with open(json_path) as f:
                json_obj = json.load(f)
                body = json_obj["body"]

                soup = BeautifulSoup(body, "html.parser")
                for script in soup(["script", "style"]):
                    script.decompose()
                text = soup.get_text()

                with torch.no_grad():
                    input_ids = torch.tensor(tokenizer.encode(text)).unsqueeze(0).to(device)
                    attention_mask = torch.ones(input_ids.shape, dtype=torch.long, device=input_ids.device).to(device)
                    global_attention_mask = torch.zeros(input_ids.shape, dtype=torch.long, device=input_ids.device).to(device)
                    outputs = model(input_ids, attention_mask=attention_mask, global_attention_mask=global_attention_mask)

                    vec = outputs.last_hidden_state[0].cpu().detach().clone().numpy().mean(0)
                # np.append(feature_matrix, vec)
                feature_matrix.append(list(vec))
                # log.info(f"Done: {len(feature_matrix)}")

                
        feature_matrix = np.array(feature_matrix)
        log.info(f"Longformer: {feature_matrix.shape}")

        # Calculate distance matrix
        dist_mat = squareform(pdist(feature_matrix, metric='cosine'))

        df = pd.DataFrame(dist_mat)
        df.to_csv(join(self.cache_path, "json_document.csv"), index=False)
        log.info(f"Successfully made dataset.")

