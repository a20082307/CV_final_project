import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def data_loader():
    """
    Load the data from the CSV file
    The data contains the price of bitcoin from 2021 to 2024
    The time period of each K line is 1 hour
    """
    
    data_csv = pd.read_csv('train.csv')
    data_csv['Open Time'] = pd.to_datetime(data_csv['Open Time'], unit = 'ms')
    data = data_csv[['Open Time', 'Open', 'High', 'Low', 'Close', 'Volume']].copy()

    return data_preprocess(data)

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
    return data


if __name__ == "__main__":
    data = data_loader()