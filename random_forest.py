import argparse
import os
import math
import warnings

# Setup the env before import umap, so that it won't jump the warning
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
warnings.filterwarnings("ignore")

import numpy as np
import pickle
import sklearn
# import tensorflow as tf
# import umap

from main import data_loader, PCA


parser = argparse.ArgumentParser()
parser.add_argument('--new', action = 'store_true', help = 'Create a new model')
parser.add_argument('--name', type = str, default = 'random_forest', help = 'The name of model')
parser.add_argument('--dim', type = int, default = -1, help = 'The target dimension we want to reduce from origin for both train data and test data')
parser.add_argument('--period', type = int, default = 24, help = 'The time period of the train and test data')
parser.add_argument('--interval', type = int, default = 1, help = 'The interval of the train and test data')
parser.add_argument('--which', type = int, default = 1, help = 'Choose the unit of kbar, 1 means 1hr, 3 means 15mins. 2 and 4 means find the difference between each two kbar')
args = parser.parse_args()


def main():
    train_data, train_label, test_data, test_label = data_loader(
        args.which,
        args.period,
        args.interval
    )

    reducer = None
    if args.dim != -1:
        if args.dim > train_data.shape[0]:
            raise ValueError(f'You want to reduce the dimension from [{train_data.shape[0]}] to [{args.dim}], which is illegal')
        else:
            # reducer = umap.UMAP(n_components = args.dim)
            # print('=====' * 20)
            # print('Reducing dimension of the test data...')
            # test_data = reducer.fit_transform(test_data.T).T
            # print(f'After dimension reduction by umap, test data has dimension {test_data.shape[0]}')
            print('=====' * 20)

    model_path = args.name + f'_which{args.which}_period{args.period}_interval{args.interval}' + (f'_dim{args.dim}' if args.dim != -1 else '') + '.pkl'
    if not os.path.exists(model_path) or args.new:
        if args.dim != -1:
            # print('Reducing dimension of the train data...')
            # train_data = reducer.fit_transform(train_data.T).T
            # print(f'After dimension reduction by umap, train data has dimension {train_data.shape[0]}')
            print('=====' * 20)

        print('Training...')
        model = sklearn.ensemble.RandomForestClassifier(n_estimators = 100, criterion = 'entropy')
        model.fit(train_data.T, train_label)

        with open(model_path, 'wb') as file:
            pickle.dump(model, file)
    else:
        with open(model_path, 'rb') as file:
            model = pickle.load(file)

    print('Predicting...')
    result = model.predict(test_data.T)
    confusion_matrix = np.array([[0, 0], [0, 0]])

    print('Analyzing...')
    for i in range(len(result)):
        confusion = (bool(result[i]), bool(test_label[i]))
        match confusion:
            case (True, True):
                confusion_matrix[0, 0] += 1
            case (True, False):
                confusion_matrix[0, 1] += 1
            case (False, True):
                confusion_matrix[1, 0] += 1
            case (False, False):
                confusion_matrix[1, 1] += 1
            case _:
                raise ValueError(f'The type of (result, test_label) should be (boolean, boolean), but now they are ({type(confusion[0])}, {type(confusion[1])})')

    accuracy = np.trace(confusion_matrix) / test_label.shape[0]
    precision = confusion_matrix[0, 0] / np.sum(confusion_matrix[0, :])
    recall = confusion_matrix[0, 0] / np.sum(confusion_matrix[:, 0])
    f1_score = 2 * precision * recall / (precision + recall)
    print(f'F1 score: {f1_score}\nAccuracy: {accuracy}')

if __name__ == '__main__':
    main()