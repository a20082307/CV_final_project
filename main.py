import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def data_loader(time_period = 24, interval = 1):
    """
    Load the data from the CSV file
    The data contains the price of bitcoin from 2021 to 2024
    The time unit of each K line is 1 hour
    Each data contains "time_period" K lines
    We want to predict the direction of the close price in the next "interval" K lines
    """
    train_data_csv = pd.read_csv('train.csv')
    train_data_csv['Open Time'] = pd.to_datetime(train_data_csv['Open Time'], unit = 'ms')
    train_data = train_data_csv[['Open Time', 'Open', 'High', 'Low', 'Close', 'Volume']].copy()
    train_data = data_preprocess(train_data)
    train_dataset, train_labels = generate_labels(train_data, time_period, interval)

    test_data_csv = pd.read_csv('test.csv')
    test_data_csv['Open Time'] = pd.to_datetime(test_data_csv['Open Time'], unit = 'ms')
    test_data = test_data_csv[['Open Time', 'Open', 'High', 'Low', 'Close', 'Volume']].copy()
    test_data = data_preprocess(test_data)
    test_dataset, test_labels = generate_labels(test_data, time_period, interval)

    return train_dataset, train_labels, test_dataset, test_labels

def data_preprocess(data):
    """
    Calculate the difference between price[t] and price[t-1] for all features including volume
    """
    data['Open'] = data['Open'].diff()
    data['High'] = data['High'].diff()
    data['Low'] = data['Low'].diff()
    data['Close'] = data['Close'].diff()
    data['Volume'] = data['Volume'].diff()

    data.dropna(inplace = True)
    data = data[['Open Time', 'Open', 'High', 'Low', 'Close', 'Volume']].copy().to_numpy()
    # print(data.shape)
    # print(data[:5, :])
    # print(data[-5:, :])

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
    dataset = np.array(dataset)
    labels = np.array(labels)

    return dataset, labels


if __name__ == "__main__":
    train_data, train_labels, test_data, test_labels = data_loader()