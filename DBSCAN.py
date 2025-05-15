import random

import numpy as np
import scipy as sp
import sklearn
import sklearn.cluster
import sklearn.decomposition
import tqdm

from main import data_loader, PCA


class DBSCAN:
    def __init__(self, data, label, eps, min_points, dimension):
        if dimension > 0:
            self.data = PCA(data, dimension)
        else:
            self.data = data

        self.label = label
        self.eps = eps
        self.min_points = min_points
        
        self.data_num = self.data.shape[1]
        self.index = [i for i in range(data.shape[1])]
        self.clusters = sp.cluster.hierarchy.DisjointSet(self.index)
        self.cluster_meaning = []
        self.noise = set()

    def fit(self):
        points_without_cluster = set(self.index)
        
        with tqdm.tqdm(total = self.data_num, desc = '[Clustering]') as pbar:
            for cur_point in range(self.data_num):
                pbar.update(1)

                if cur_point not in points_without_cluster:
                    continue

                points_without_cluster.remove(cur_point)

                diff = self.data[:, cur_point].reshape(-1, 1) - self.data
                dist = np.linalg.norm(diff, axis = 0)
                neighbors = np.where(dist <= self.eps)[0]

                if len(neighbors) < self.min_points:
                    self.noise.add(cur_point)
                    continue

                while len(neighbors):
                    neighbor = neighbors[0]
                    neighbors = np.delete(neighbors, 0)

                    if neighbor not in points_without_cluster and neighbor not in self.noise:
                        continue

                    if neighbor in self.noise:
                        self.noise.remove(neighbor)
                    elif neighbor in points_without_cluster:
                        points_without_cluster.remove(neighbor)

                    self.clusters.merge(cur_point, neighbor)
                    
                    n_diff = self.data[:, neighbor].reshape(-1, 1) - self.data
                    n_dist = np.linalg.norm(n_diff, axis = 0)
                    neighbor_neighbors = np.where(n_dist <= self.eps)[0]

                    if len(neighbor_neighbors) < self.min_points:
                        continue

                    for nn in neighbor_neighbors:
                        if nn in neighbors:
                            continue

                        neighbors = np.hstack([neighbors, nn])
                        
        all_clusters = self.clusters.subsets()
        cluster_size = np.array([len(cluster) for cluster in all_clusters])
        valid_cluster = np.where(cluster_size > 1)[0]
        self.clusters = [all_clusters[i] for i in valid_cluster]

        # print(f'There are {valid_cluster.shape[0]} clusters and {len(self.noise)} noise points')
        # for i in range(len(self.clusters)):
        #     print(f'Cluster {i + 1} has {len(self.clusters[i])} elements')

        for index in valid_cluster:
            cluster = all_clusters[index]
            rise_points = [i for i in cluster if self.label[i]]
            rise_prob = len(rise_points) / len(cluster)
            
            if rise_prob > 0.5:
                self.cluster_meaning.append(('Rise', rise_prob))
            elif rise_prob < 0.5:
                self.cluster_meaning.append(('Fall', 1 - rise_prob))
            else:
                self.cluster_meaning.append(('Not Sure', 0.5))

        print(self.cluster_meaning)

    def predict(self, data):
        raise NotImplementedError


if __name__ == '__main__':
    train_data, train_label, _, _ = data_loader(24, 2)
    # print(np.sum(train_label == 1), np.sum(train_label == 0), len(train_label))

    model = DBSCAN(train_data, train_label, 0.05, 20, 2)
    model.fit()

    # train_data = PCA(train_data, 2).T
    # print(train_data.shape)
    # db = sklearn.cluster.DBSCAN(eps = 0.05, min_samples = 20, metric='euclidean')
    # clusters = db.fit_predict(train_data)

    # cluster_num = len(np.unique(clusters[clusters != -1]))
    # print(f"Number of clusters: {cluster_num}")
    # print(f"Number of noise points: {np.sum(clusters == -1)}")
    # for i in range(cluster_num):
    #     print(len(clusters[clusters == i]))