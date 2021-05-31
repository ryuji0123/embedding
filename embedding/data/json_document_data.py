from os import path
from os.path import join
import logging
import glob
import json

import pandas as pd
import numpy as np
from bs4 import BeautifulSoup

from embedding.data.parent_data import ParentData


log = logging.getLogger(__name__)


class JsonDocumentData(ParentData):
    def __init__(self, *args) -> None:
        super().__init__(*args)
        self.setDataFrameAndColor(self.data_path)
        self.data_key = "json_document"

    def setDataFrameAndColor(self, root: str) -> None:
        if not path.exists(join(self.cache_path, "json_document.csv")):
            data_root = join(root, "private/blog.barracuda.com_en_2021-03-11_0")
            self.make_dataset(data_root)

        self.df = pd.read_csv(
                join(self.cache_path, "json_document.csv"),
                )

        self.color = np.squeeze(pd.read_csv(
                join(root, "pokemon.csv.gz"),
                usecols=["is_legendary"]).values)

    def make_dataset(self, data_root: str) -> None:
        """
        Make dataset from json files and save it as csv.
        """

        log.info("####### JSON ##########################################################")
        
        l = glob.glob(f"{data_root}/**/*.json")
        log.info(f"Num of json: {len(l)}")

        
        with open(l[0]) as f:
            json_obj = json.load(f)
            body = json_obj["body"]

            soup = BeautifulSoup(body, "html.parser")
            for script in soup(["script", "style"]):
                script.decompose()
            text = soup.get_text()
            log.info(f"Sample: {text}")
