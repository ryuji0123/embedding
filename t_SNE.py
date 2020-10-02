import csv
import numpy as np
from sklearn.manifold import TSNE


def main(n_dim):
    # preprocess
    data = preprocess()

    # t-SNE
    X_reduced = TSNE(n_components=n_dim, random_state=0).fit_transform(data)
    print(f'X_reduced shape: {X_reduced.shape}')


def preprocess():
    data = []  # feature_value
    with open('data/pokemon.csv') as f:
        reader = csv.reader(f)
        for i, row in enumerate(reader):
            if i == 0:
                continue
            data.append([int(row[j]) for j in index_list])
    data = np.array(data)
    print(f'race_value data shape : {data.shape}')
    return data


if __name__ == '__main__':
    # race_val = {'name': 29, 'hp': 28, 'atk': 19, 'def': 25, 's_atk': 33, 's_def': 34, 'spd': 35}
    index_list = [28, 19, 25, 33, 34, 35]
    print(f'X: race_value')

    # check_feature()
    main(2)