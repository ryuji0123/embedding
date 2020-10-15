import pandas as pd

from os.path import join

from embedding.data.parent_data import ParentData


class PokemonData(ParentData):
    def __init__(self, *args):
        super(PokemonData, self).__init__(*args)
        self.df = self.getDataFrame(self.data_path)
        self.data_key = "pokemon"

    def getDataFrame(self, root):
        return pd.read_csv(
                join(root, "pokemon.csv.gz"),
                usecols=["hp", "attack", "defense", "sp_attack", "sp_defense", "speed"],
                )
