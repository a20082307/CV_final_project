import numpy as np
import scipy as sp

import main

class DBSCAN:
    def __init__(self, data):
        self.data = data
        self.clusters = []

    def fit(self, eps, min_points, dimension = -1):
        if dimension > 0:
            self.data = PCA(self.data, dimension)
        pass

    def predict(self, data):

        pass


def PCA(data, dimension):
    _, eigenvectors = sp.sparse.linalg.eigs(data, k = dimension, which = 'LM')
    eigenvectors = eigenvectors.real

    W = eigenvectors[:, : dimension]
    Z = W.T @ (data - np.mean(data, axis = 1).reshape(-1, 1))


if __name__ == '__main__':
    train_data, _, _, _ = main.data_loader()
    print(train_data.shape)