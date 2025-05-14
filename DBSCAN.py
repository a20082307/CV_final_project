import copy
import random

import numpy as np
import scipy as sp
import sklearn
import sklearn.cluster

from main import data_loader, PCA


class DBSCAN:
    def __init__(self, data, label, eps, min_points, dimension):
        self.data = data
        self.label = label
        self.eps = eps
        self.min_points = min_points
        self.dim = dimension
        
        self.index = [i for i in range(data.shape[1])]
        self.clusters = sp.cluster.hierarchy.DisjointSet(self.index)
        self.noise = set()

    def fit(self):
        if self.dim > 0:
            self.data = PCA(self.data, self.dim)

        points_without_cluster = set(self.index)
        select_order = copy.deepcopy(self.index)
        
        random.seed(110062333)
        random.shuffle(select_order)

        print('clustering...')
        while len(points_without_cluster):
            random_index = select_order.pop()
            if not random_index in points_without_cluster:
                continue

            random_point = self.data[:, random_index].reshape(-1, 1)
            points_without_cluster.remove(random_index)

            diff = random_point - self.data
            dist = np.linalg.norm(diff, axis = 0)
            dist = dist[dist > 0]
            # print(np.max(dist), np.min(dist))

            neighbors = np.where(dist < self.eps)[0]
            if neighbors.shape[0] >= self.min_points:
                # print(neighbors.shape, len(points_without_cluster))
                for neighbor in neighbors:
                    if neighbor in points_without_cluster:
                        points_without_cluster.remove(neighbor)
                    elif neighbor in self.noise:
                        self.noise.remove(neighbor)
                    self.clusters.merge(random_index, neighbor)
            else:
                self.noise.add(random_index)
        
        cluster_size = np.array([len(cluster) for cluster in self.clusters.subsets()])
        valid_cluster = np.where(cluster_size > self.min_points)[0]

        print(f'There are {valid_cluster.shape[0]} clusters and {len(self.noise)} noise points')
        for i in range(len(valid_cluster)):
            print(f'Cluster {i} has {len(self.clusters.subsets()[valid_cluster[i]])} elements')

            
    def predict(self, data):



        raise NotImplementedError

if __name__ == '__main__':
    train_data, train_label, _, _ = data_loader(24, 2)
    # print(np.sum(train_label == 1), np.sum(train_label == 0), len(train_label))

    model = DBSCAN(train_data, train_label, 4, 30, -1)
    model.fit()

    db = sklearn.cluster.DBSCAN(eps=4, min_samples=30, metric='euclidean')
    clusters = db.fit_predict(train_data.T)
    print(f"Number of clusters: {len(np.unique(clusters[clusters != -1]))}")
    print(f"Number of noise points: {np.sum(clusters == -1)}")