import pandas as pd
import numpy as np
from os.path import join

from embedding.data.parent_data import ParentData


class PokemonData(ParentData):
    def __init__(self, *args):
        super(PokemonData, self).__init__(*args)
<<<<<<< HEAD
        self.set_dataframe_and_color(self.data_path)
        self.data_key = "pokemon"

    def set_dataframe_and_color(self, root):
=======
        self.setDataFrameAndColor(self.data_path)
        self.data_key = "pokemon"

    def setDataFrameAndColor(self, root):
>>>>>>> 171bed1951b805dff50142670868bca9679a11f6
        self.df = pd.read_csv(
                join(root, "pokemon.csv.gz"),
                usecols=["hp", "attack", "defense", "sp_attack", "sp_defense", "speed"],
                )

        self.color = np.squeeze(pd.read_csv(
                join(root, "pokemon.csv.gz"),
                usecols=["is_legendary"]).values)
