import pandas as pd

from os.path import join

from embedding.data import ParentData


class PokemonData(ParentData):
    @staticmethod
    def getDataFrame(root):
        return pd.read_csv(
                join(root, "pokemon.csv.gz"),
                usecols=["hp", "attack", "defense", "sp_attack", "sp_defense", "speed"],
                )
