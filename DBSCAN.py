import numpy as np
import scipy as sp

from main import data_loader, PCA

class BallTreeNode:
    def __init__(self, centroid, p1, p2, dimension):
        self.centroid = centroid
        self.p1 = p1
        self.p2 = p2
        self.left = None
        self.right = None
        self.is_leaf = False
        self.points = np.empty((dimension, 0))


class BallTree:
    def __init__(self, data, min_points):
        self.root = self._build(data, min_points)

    def _build(self, data, min_points):
        # Step 1: Find the centroid of the data points
        centroid = np.mean(data, axis = 1)
        print(centroid, centroid.shape)

        # Step 2: Find the points that are farthest from the centroid


class DBSCAN:
    def __init__(self, data):
        self.data = data
        self.clusters = []

    def fit(self, eps, min_points, dimension = -1):
        if dimension > 0:
            self.data = PCA(self.data, dimension)

        self.tree = BallTree(self.data, min_points)

        diff = self.data.T[:, np.newaxis, :] - self.data.T[np.newaxis, :, :]
        dist = np.linalg.norm(diff, axis = -1)
        print(dist.shape)

    def predict(self, data):

        pass

if __name__ == '__main__':
    train_data, train_label, _, _ = data_loader()
    # print(np.sum(train_label == 1), np.sum(train_label == 0), len(train_label))

    model = DBSCAN(train_data)
    model.fit(0.5, 5, 20)