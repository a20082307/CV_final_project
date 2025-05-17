import random

import numpy as np
import scipy as sp
# import sklearn
# import sklearn.cluster
# import sklearn.decomposition
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
        # for i, cluster in enumerate(valid_cluster):
        #     print(f'Cluster {i + 1} has {len(all_clusters[cluster])} elements')

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

        # for i in range(len(self.cluster_meaning)):
        #     print(f'Cluster {i + 1} will predict [{self.cluster_meaning[i][0]}] with confidence [{self.cluster_meaning[i][1]}]')

    def predict(self, data):
        if data.shape[0] != self.data.shape[0]:
            print()
            raise ValueError(f'The dimension of input data isn\'t equal to the train data. The dimension of input data is [{data.shape[0]}] and the dimension of train data is [{self.data.shape[0]}]\n')

        diff = data.reshape(-1 ,1) - self.data
        dist = np.linalg.norm(diff, axis = 0)
        closet_point = np.argsort(dist, kind = 'stable')
        sorted_dist = dist[closet_point]

        candidate = 0
        while sorted_dist[candidate] < self.eps:
            if closet_point[candidate] not in self.noise:
                for i, cluster in enumerate(self.clusters):
                    if closet_point[candidate] in cluster:
                        return self.cluster_meaning[i]
            candidate += 1

        if candidate == 0:  
            centroids = np.array([np.mean(self.data[:, np.array(list(cluster))], axis = 1) for cluster in self.clusters]).T
            dc_diff = data.reshape(-1, 1) - centroids
            dc_dist = np.linalg.norm(dc_diff, axis = 0)

            return self.cluster_meaning[np.argmin(dc_dist)]
        
        else:
            rise_score = 0
            fall_score = 0
            for i, point in enumerate(closet_point):
                if self.label[point]:
                    rise_score += np.exp(-sorted_dist[i] / (2 * (self.eps / 2) ** 2))
                else:
                    fall_score += np.exp(-sorted_dist[i] / (2 * (self.eps / 2) ** 2))

            rise_prob = rise_score / (rise_score + fall_score)
            if rise_prob > 0.5:
                return ('Rise', rise_prob)
            elif rise_prob < 0.5:
                return ('Fall', 1 - rise_prob)
            else:
                return ('Not Sure', 0.5)


if __name__ == '__main__':
    train_data, train_label, test_data, test_label = data_loader(1, 24, 2)
    # print(np.sum(train_label == 1), np.sum(train_label == 0), len(train_label))

    model = DBSCAN(train_data, train_label, 0.05, 20, 2)
    model.fit()

    test_data = PCA(test_data, 2)
    confusion_matrix = np.array([[0, 0], [0, 0]])
    unknown = 0
    
    for i, data in enumerate(tqdm.tqdm(test_data.T, desc = '[Predicting]')):
        rlt = model.predict(data)
        confusion = ((rlt[0], bool(test_label[i])))

        match confusion:
            case ('Rise', True):
                confusion_matrix[0, 0] += 1
            case ('Rise', False):
                confusion_matrix[1, 0] += 1
            case ('Fall', True):
                confusion_matrix[0, 1] += 1
            case ('Fall', False):
                confusion_matrix[1, 1] += 1
            case _:
                unknown += 1

    accuracy = np.trace(confusion_matrix) / test_label.shape[0]
    precision = confusion_matrix[0, 0] / np.sum(confusion_matrix[:, 0])
    recall = confusion_matrix[0, 0] / np.sum(confusion_matrix[0, :])
    f1_score = 2 * precision * recall / (precision + recall)
    print(f'F1 score: {f1_score}\nAccuracy: {accuracy}')
    print(f'Data with No trend have {unknown}, they take about {(unknown / test_label.shape[0] * 100):.2f}%')


# test code:
    ### To see if class DBSCAN behaves like sklearn.cluster.DBSCAN ###
    # train_data = PCA(train_data, 2).T
    # print(train_data.shape)
    # db = sklearn.cluster.DBSCAN(eps = 0.05, min_samples = 20, metric='euclidean')
    # clusters = db.fit_predict(train_data)
    # cluster_num = len(np.unique(clusters[clusters != -1]))
    # print(f"Number of clusters: {cluster_num}")
    # print(f"Number of noise points: {np.sum(clusters == -1)}")
    # for i in range(cluster_num):
    #     print(len(clusters[clusters == i]))
    ### ============================== ###

    ### To see if predict() can handle the input data with different dimension to the train data ###
    # wrong_data = np.array([1, 2, 3, 4, 5])
    # model.predict(wrong_data)
    ### ============================== ###
    