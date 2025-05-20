import argparse
import random
import math
import os

import numpy as np
import pickle
import scipy as sp
import sklearn as sk
# import sklearn.cluster
# import sklearn.decomposition
import tqdm

from main import data_loader, PCA

# class DBSCAN:
#     def __init__(self, data, label, eps, min_points, dimension):
#         if dimension > 0:
#             self.data, self.train_mean, self.W = PCA(data, dimension)
#         else:
#             self.data = data

#         self.label = label
#         self.eps = eps
#         self.min_points = min_points
        
#         self.data_num = self.data.shape[1]
#         self.index = [i for i in range(data.shape[1])]
#         self.clusters = sp.cluster.hierarchy.DisjointSet(self.index)
#         self.cluster_meaning = []
#         self.noise = set()

#     def fit(self):
#         points_without_cluster = set(self.index)
        
#         with tqdm.tqdm(total = self.data_num, desc = '[Clustering]') as pbar:
#             for cur_point in range(self.data_num):
#                 if cur_point not in points_without_cluster:
#                     continue
#                 pbar.update(1)
                
#                 points_without_cluster.remove(cur_point)

#                 diff = self.data[:, cur_point].reshape(-1, 1) - self.data
#                 dist = np.linalg.norm(diff, axis = 0)
#                 neighbors = np.where(dist <= self.eps)[0]

#                 if len(neighbors) < self.min_points:
#                     self.noise.add(cur_point)
#                     continue

#                 while len(neighbors):
#                     neighbor = neighbors[0]
#                     neighbors = np.delete(neighbors, 0)

#                     if neighbor not in points_without_cluster and neighbor not in self.noise:
#                         continue

#                     if neighbor in self.noise:
#                         self.noise.remove(neighbor)
#                     elif neighbor in points_without_cluster:
#                         points_without_cluster.remove(neighbor)
#                         pbar.update(1)

#                     self.clusters.merge(cur_point, neighbor)
                    
#                     n_diff = self.data[:, neighbor].reshape(-1, 1) - self.data
#                     n_dist = np.linalg.norm(n_diff, axis = 0)
#                     neighbor_neighbors = np.where(n_dist <= self.eps)[0]

#                     if len(neighbor_neighbors) < self.min_points:
#                         continue

#                     for nn in neighbor_neighbors:
#                         if nn in neighbors:
#                             continue
#                         if nn not in points_without_cluster and nn not in self.noise:
#                             continue

#                         neighbors = np.hstack([neighbors, nn])
                        
#         all_clusters = self.clusters.subsets()
#         cluster_size = np.array([len(cluster) for cluster in all_clusters])
#         valid_cluster = np.where(cluster_size > 1)[0]
#         self.clusters = [all_clusters[i] for i in valid_cluster]

#         # print(f'There are {valid_cluster.shape[0]} clusters and {len(self.noise)} noise points')
#         # for i, cluster in enumerate(valid_cluster):
#         #     print(f'Cluster {i + 1} has {len(all_clusters[cluster])} elements')

#         for index in valid_cluster:
#             cluster = all_clusters[index]
#             rise_points = [i for i in cluster if self.label[i]]
#             rise_prob = len(rise_points) / len(cluster)
            
#             if rise_prob > 0.5:
#                 self.cluster_meaning.append(('Rise', rise_prob))
#             elif rise_prob < 0.5:
#                 self.cluster_meaning.append(('Fall', 1 - rise_prob))
#             else:
#                 self.cluster_meaning.append(('Not Sure', 0.5))

#         # for i in range(len(self.cluster_meaning)):
#         #     print(f'Cluster {i + 1} will predict [{self.cluster_meaning[i][0]}] with confidence [{self.cluster_meaning[i][1]}]')

#     def predict(self, dataset, label):
#         dataset -= self.train_mean
#         dataset = self.W.T @ dataset

#         confusion_matrix = np.array([[0, 0], [0, 0]])
#         unknown = 0
        
#         for i, data in enumerate(tqdm.tqdm(dataset.T, desc = '[Predicting]')):
#             rlt = model.predict_single_point(data)
#             confusion = ((rlt[0], bool(label[i])))

#             match confusion:
#                 case ('Rise', True):
#                     confusion_matrix[0, 0] += 1
#                 case ('Rise', False):
#                     confusion_matrix[1, 0] += 1
#                 case ('Fall', True):
#                     confusion_matrix[0, 1] += 1
#                 case ('Fall', False):
#                     confusion_matrix[1, 1] += 1
#                 case _:
#                     unknown += 1

#         accuracy = np.trace(confusion_matrix) / label.shape[0]
#         precision = confusion_matrix[0, 0] / np.sum(confusion_matrix[:, 0])
#         recall = confusion_matrix[0, 0] / np.sum(confusion_matrix[0, :])
#         f1_score = 2 * precision * recall / (precision + recall)
#         print(f'F1 score: {f1_score}\nAccuracy: {accuracy}')
#         print(f'Data with No trend have {unknown}, they take about {(unknown / label.shape[0] * 100):.2f}%')

#     def predict_single_point(self, data):
#         if data.shape[0] != self.data.shape[0]:
#             print()
#             raise ValueError(f'The dimension of input data isn\'t equal to the train data. The dimension of input data is [{data.shape[0]}] and the dimension of train data is [{self.data.shape[0]}]\n')

#         diff = data.reshape(-1 ,1) - self.data
#         dist = np.linalg.norm(diff, axis = 0)
#         closet_point = np.argsort(dist, kind = 'stable')
#         sorted_dist = dist[closet_point]

#         candidate = 0
#         while sorted_dist[candidate] < self.eps:
#             if closet_point[candidate] not in self.noise:
#                 for i, cluster in enumerate(self.clusters):
#                     if closet_point[candidate] in cluster:
#                         return self.cluster_meaning[i]
#             candidate += 1

#         if candidate == 0:  
#             centroids = np.array([np.mean(self.data[:, np.array(list(cluster))], axis = 1) for cluster in self.clusters]).T
#             dc_diff = data.reshape(-1, 1) - centroids
#             dc_dist = np.linalg.norm(dc_diff, axis = 0)

#             return self.cluster_meaning[np.argmin(dc_dist)]
        
#         else:
#             rise_score = 0
#             fall_score = 0
#             for i, point in enumerate(closet_point):
#                 if self.label[point]:
#                     rise_score += np.exp(-sorted_dist[i] / (2 * (self.eps / 2) ** 2))
#                 else:
#                     fall_score += np.exp(-sorted_dist[i] / (2 * (self.eps / 2) ** 2))

#             rise_prob = rise_score / (rise_score + fall_score)
#             if rise_prob > 0.5:
#                 return ('Rise', rise_prob)
#             elif rise_prob < 0.5:
#                 return ('Fall', 1 - rise_prob)
#             else:
#                 return ('Not Sure', 0.5)


DIM = 10
EPS = 7.5
MIN_SAMPLE = 10

parser = argparse.ArgumentParser()
parser.add_argument('--new')
args = parser.parse_args()

def predict(clusters, train_data, train_label, test_data):
    cluster_num = len(np.unique(clusters[clusters != -1]))

    cluster_meaning = []
    for i in range(cluster_num):
        points = clusters == i
        rise_points = len(train_label[points] == True)
        fall_points = len(train_label[points] == False)
        rise_prob = (rise_points) / (rise_points + fall_points)
        
        if rise_prob > 0.5:
            cluster_meaning.append(('Rise', rise_prob))
        elif rise_prob < 0.5:
            cluster_meaning.append(('Fall', 1 - rise_prob))
        else:
            cluster_meaning.append(('Not Sure', 0.5))

    result = []

    for data in tqdm.tqdm(test_data.T, desc = '[Predicting]'):
        diff = data.reshape(-1, 1) - train_data
        dist = np.linalg.norm(diff, axis = 0)
        closet_point = np.argsort(dist, kind = 'stable')
        sorted_dist = dist[closet_point]

        candidate = 0
        finish = False
        while sorted_dist[candidate] < EPS:
            cluster_id = clusters[closet_point[candidate]]
            if cluster_id != -1:
                result.append(cluster_meaning[cluster_id])
                finish = True
                break
            
            candidate += 1

        if finish:
            continue

        # No any point in eps
        if candidate == 0:
            centroids = np.array([np.mean(train_data[:, clusters == i], axis = 1) for i in range(cluster_num)]).T
            diff = data.reshape(-1, 1) - centroids
            dist = np.linalg.norm(diff, axis = 0)

            result.append(cluster_meaning[np.argmin(dist)])
            continue
        
        # All the points in eps are noise points
        else:
            rise_score = 0
            fall_score = 0
            for i, point in enumerate(closet_point):
                if train_label[point]:
                    rise_score += np.exp(-sorted_dist[i] / (2 * (EPS / 2) ** 2))
                else:
                    fall_score += np.exp(-sorted_dist[i] / (2 * (EPS / 2) ** 2))

            rise_prob = rise_score / (rise_score + fall_score)
            if rise_prob > 0.5:
                result.append(('Rise', rise_prob))
            elif rise_prob < 0.5:
                result.append(('Fall', 1 - rise_prob))
            else:
                result.append(('Not Sure', 0.5))
            continue
            
    return result

def main():
    train_data, train_label, test_data, test_label = data_loader(2, 32, 1)
    # print(np.sum(train_label == 1), np.sum(train_label == 0), len(train_label))

    # model = DBSCAN(train_data, train_label, 0.5, 20, 4)
    # model.fit()
    # model.predict(test_data, test_label)

    # train_data, train_mean, W = PCA(train_data, DIM)
    if not os.path.exists('DBSCAN.pkl') or args.new:
        print('Clustering...')
        model = sk.cluster.DBSCAN(eps = EPS, min_samples = MIN_SAMPLE)
        clusters = model.fit_predict(train_data.T)
        print('Finish clustering')

        cluster_num = len(np.unique(clusters[clusters != -1]))
        print("==============================")
        print(f"Number of clusters: {cluster_num}")
        print(f"Number of noise points: {np.sum(clusters == -1)}")
        print("==============================")

        if cluster_num == 1:
            print('Since the number of clusters is 1, please select other set of hyperparameters')
            return

        with open('DBSCAN.pkl', 'wb') as file:
            pickle.dump(clusters, file)
    else:
        with open('DBSCAN.pkl', 'rb') as file:
            clusters = pickle.load(file)
        print("Load model successfully")

    # test_data -= train_mean
    # test_data = W.T @ test_data
    result = predict(clusters, train_data, train_label, test_data)

    confusion_matrix = np.array([[0, 0], [0, 0]])
    noise = 0
    for i in range(len(result)):
        confusion = (result[i][0], test_label[i])
        print(confusion)
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
                noise += 1

    accuracy = np.trace(confusion_matrix) / test_label.shape[0]
    precision = confusion_matrix[0, 0] / np.sum(confusion_matrix[:, 0])
    recall = confusion_matrix[0, 0] / np.sum(confusion_matrix[0, :])
    f1_score = 2 * precision * recall / (precision + recall)
    print(f'F1 score: {f1_score}\nAccuracy: {accuracy}')
    print(f'Data with No trend have {noise}, they take about {(noise / test_label.shape[0] * 100):.4f}%')

if __name__ == '__main__':
    main()