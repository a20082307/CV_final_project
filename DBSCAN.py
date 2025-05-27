import argparse
import random
import math
import os

import numpy as np
import pickle
import scipy as sp
import sklearn as sk
import tqdm

from main import data_loader, PCA

EPS = 11
MIN_SAMPLE = 5000

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
    train_data, train_label, test_data, test_label = data_loader(which = 5, time_period = 32, interval = 1)
    print(f'The shape of train data: {train_data.shape}')
    print("==============================")

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