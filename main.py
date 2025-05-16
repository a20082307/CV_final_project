import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy as sp

def data_loader(which = 1, time_period = 24, interval = 1):
    """
    Load the data from the CSV file
    The data contains the price of bitcoin from 2021 to 2024
    The time unit of each K line is 1 hour
    Each data contains "time_period" K lines
    We want to predict the direction of the close price in the next "interval" K lines
    """
    data_path = {
        1: ['train.csv', 'test.csv'],
        2: ['train_15min.csv', 'test_15min.csv']
    }

    if which != 1 and which != 2:
        raise ValueError(f'No such kind of data, We only have [2] kinds of data, but you require [{which}]th kind of data')

    train_data_csv = pd.read_csv(data_path[which][0])
    train_data_csv['Open Time'] = pd.to_datetime(train_data_csv['Open Time'], unit = 'ms')
    tem_train_data = train_data_csv[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
    train_data = data_preprocess(tem_train_data, tem_train_data)
    train_dataset, train_labels = generate_labels(train_data, time_period, interval)

    test_data_csv = pd.read_csv(data_path[which][1])
    test_data_csv['Open Time'] = pd.to_datetime(test_data_csv['Open Time'], unit = 'ms')
    tem_test_data = test_data_csv[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
    test_data = data_preprocess(tem_test_data, tem_train_data)
    test_dataset, test_labels = generate_labels(test_data, time_period, interval)

    return train_dataset, train_labels, test_dataset, test_labels

def data_preprocess(data, train_data):
    """
    Calculate the difference between price[t] and price[t-1] for all features including volume
    Notice that we also need to pass train data into this function
    Since when normalizing the data, we need to use the mean and std of the train data to normalize the target data
    """
    data['Open'] = data['Open'].diff()
    data['High'] = data['High'].diff()
    data['Low'] = data['Low'].diff()
    data['Close'] = data['Close'].diff()
    data['Volume'] = data['Volume'].diff()

    data.dropna(inplace = True)
    data = data[['Open', 'High', 'Low', 'Close', 'Volume']].copy().to_numpy()
    # print(data.shape)
    # print(data[:5, :])
    # print(data[-5:, :])

    train_data['Open'] = train_data['Open'].diff()
    train_data['High'] = train_data['High'].diff()
    train_data['Low'] = train_data['Low'].diff()
    train_data['Close'] = train_data['Close'].diff()
    train_data['Volume'] = train_data['Volume'].diff()

    train_data.dropna(inplace = True)
    train_data = train_data[['Open', 'High', 'Low', 'Close', 'Volume']].copy().to_numpy()

    # Normalize the data
    for i in range(5):
        data[:, i] = (data[:, i] - np.mean(train_data[:, i])) / np.std(train_data[:, i])
        
    return data

def generate_labels(data, time_period = 24, interval = 1):
    """
    Generate the dataset and the labels for the data
    """
    dataset = []
    labels = []
    for i in range(len(data) - time_period - interval):
        dataset.append(data[i : i + time_period])
        labels.append(data[i + time_period + interval - 1][4] > data[i + time_period - 1][4])
    
    dataset = np.array(dataset).T
    dataset = dataset.reshape((-1, dataset.shape[-1]))
    # print(dataset.shape)

    labels = np.array(labels)
    # print(labels.shape)

    return dataset, labels

def PCA(data, dimension):
    if (dimension <= 0):
        raise ValueError("Dimension must be greater than 0")
    
    if (dimension > data.shape[0]):
        raise ValueError("Dimension must be less than or equal to the number of features")
    
    mean = np.mean(data, axis = 1).reshape(-1, 1)
    centered_data = data - mean
    covariance = centered_data @ centered_data.T

    _, eigenvectors = sp.sparse.linalg.eigs(covariance, k = dimension, which = 'LM')
    eigenvectors = eigenvectors.real

    W = eigenvectors[:, : dimension]
    Z = W.T @ centered_data
    # print(Z.shape)

    return Z


if __name__ == "__main__":
    train_data, train_labels, test_data, test_labels = data_loader(which = 1)
    print(f'========== First kind of data ==========')
    print(train_data[:5, :3])
    print(train_data[:5, :3])
    print(test_data[:5, :3])
    print(test_data[:5, :3])
    print()
    

    train_data, train_labels, test_data, test_labels = data_loader(which = 2)
    print(f'========== Second kind of data ==========')
    print(train_data[:5, :3])
    print(train_data[:5, :3])
    print(test_data[:5, :3])
    print(test_data[:5, :3])