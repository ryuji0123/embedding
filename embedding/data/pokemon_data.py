import pandas as pd
import numpy as np
from os.path import join

from embedding.data.parent_data import ParentData


class PokemonData(ParentData):
    def __init__(self, *args):
        super(PokemonData, self).__init__(*args)
        self.setDataFrameAndColor(self.data_path)
        self.data_key = "pokemon"

    def setDataFrameAndColor(self, root):
        self.df = pd.read_csv(
                join(root, "pokemon.csv.gz"),
                usecols=["hp", "attack", "defense", "sp_attack", "sp_defense", "speed"],
                )

        self.color = np.squeeze(pd.read_csv(
                join(root, "pokemon.csv.gz"),
                usecols=["is_legendary"]).values)
