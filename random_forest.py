import argparse
import os
import math
import warnings

import numpy as np
import pickle
import sklearn
import tensorflow as tf
import umap

from main import data_loader, PCA

warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()
parser.add_argument('--new', action = 'store_true', help = 'Create a new model')
parser.add_argument('--model_path', type = str, default = './random_forest.pkl', help = 'Store and load the model from this path')
parser.add_argument('--dim', type = int, default = -1, help = 'The target dimension we want to reduce from origin for both train data and test data')
args = parser.parse_args()


def main():
    train_data, train_label, test_data, test_label = data_loader()

    if args.dim != -1:
        if args.dim > train_data.shape[0]:
            raise ValueError(f'You want to reduce the dimension from [{train_data.shape[0]}] to [{args.dim}], which is illegal')

    if not os.path.exists(args.model_path) or args.new:
        if args.dim != -1:
            reducer = umap.UMAP(n_components = args.dim)
            train_data = reducer.fit_transform(train_data.T).T
            print(f'After dimension reduction by umap, train data has dimension {train_data.shape[0]}')

        model = sklearn.ensemble.RandomForestClassifier()
        model.fit(train_data.T, train_label)

        with open(args.model_path, 'wb') as file:
            pickle.dump(model, file)
    else:
        if args.dim != -1:
            reducer = umap.UMAP(n_components = args.dim)
            test_data = reducer.fit_transform(test_data.T).T
            print(f'After dimension reduction by umap, test data has dimension {test_data.shape[0]}')

        with open(args.model_path, 'rb') as file:
            model = pickle.load(file)

    result = model.predict(test_data.T)
    

if __name__ == '__main__':
    main()