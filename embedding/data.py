import pandas as pd
from os import path
from os.path import join

DATA_PATH = join(path.sep, 'workspace', 'data')


class Data:
    def __init__(self, file_path, file_sep=','):
        self.df = pd.read_csv(file_path, sep=file_sep)


if __name__ == '__main__':
    file_path = join(DATA_PATH, 'pokemon.csv.gz')
    data = Data(file_path)
