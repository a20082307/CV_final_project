import pandas as pd
import numpy as np

def data_loader():
    """
    Load the data from the CSV file
    The data contains the price of bitcoin from 2021 to 2024
    The time period of each K line is 1 hour
    """
    
    data_csv = pd.read_csv('train.csv')
    data_csv['Open Time'] = pd.to_datetime(data_csv['Open Time'], unit = 'ms')
    data = data_csv[['Open Time', 'Open', 'High', 'Low', 'Close', 'Volume']].copy()
    return data

def data_preprocess(data):
    """
    Calculate the difference between price[t] and price[t-1] for all features including volume
    """






if __name__ == "__main__":
    data = data_loader()
    data = data_preprocess(data)