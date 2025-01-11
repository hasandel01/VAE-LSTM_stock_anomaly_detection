import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler

def preprocess_raw_data(stock_data, market_data):
    stock_data.index = pd.to_datetime(stock_data.index).tz_convert("US/Eastern")
    market_data.index = pd.to_datetime(market_data.index).tz_convert("US/Eastern")

    stock_data['Return'] = stock_data['Close'].pct_change() * 100
    market_data['Return'] = market_data['Close'].pct_change() * 100
    stock_data['Relative Return'] = stock_data['Return'] - market_data['Return']
    stock_data['Volume Change'] = stock_data['Volume'].pct_change() * 100
    stock_data['Volatility'] = stock_data['Return'].rolling(window=7).std()

    covariance = stock_data['Return'].rolling(window=35).cov(market_data['Return'])
    market_variance = market_data['Return'].rolling(window=35).var()
    stock_data['Beta'] = covariance / market_variance

    data = pd.DataFrame({
        'stock_return': stock_data['Return'],
        'market_return': market_data['Return'],
        'relative_return': stock_data['Relative Return'],
        'volume_change': stock_data['Volume Change'],
        'volatility': stock_data[  'Volatility'],
        'beta': stock_data['Beta']
    })

    data.replace([np.inf, -np.inf], np.nan, inplace=True)
    data.dropna(inplace=True)

    return data


def scale_data(data):
    """
    Scale data for deep learning models. Using MinMaxScaler to make sure the data is normalized, variables that
    are measured at different scales contributes equally to the model fitting.
    :param data: Preprocessed data.
    :return: Scaled data and scaling factor.
    """
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data)

    return scaled_data, scaler


import numpy as np


def create_sequences(data, lookback=33, target_col=0):
    """
    Create Seq2One sequences and capture date indices for each label (y).

    Parameters:
        data (pd.DataFrame): Must have a DateTimeIndex or some meaningful index.
        lookback (int): Number of time steps in each input sequence.
        target_col (int): Which column to predict.

    Returns:
        X (np.ndarray): shape (num_sequences, lookback, num_features)
        y (np.ndarray): shape (num_sequences,)
        y_dates (pd.Index): Date index for each label in y.
        :param target_col:
    """
    if not isinstance(data, pd.DataFrame):
        raise ValueError("Data must be a pandas DataFrame with a DateTimeIndex.")

    date_index = data.index  # original date index
    data_vals = data.values

    X, y = [], []
    y_dates = []

    for i in range(len(data_vals) - lookback):
        # Past 'lookback' rows as input
        X.append(data_vals[i : i + lookback])
        # The next row’s target_col as the label
        y.append(data_vals[i + lookback, target_col])
        # Store the date that corresponds to the label
        y_dates.append(date_index[i + lookback])

    return np.array(X), np.array(y), pd.Index(y_dates)



def test_train_split(scaled_data, train_ratio=0.8):
    """
    Using manuel train_test_split for deep learning models, this function divides the dataset into train and test sets.
    It holds the sequential nature of the time-series data.
    :param train_ratio: Train ratio
    :param scaled_data: Scaled stock data.
    :return: A tuple (train_data, test_data) where:
             - train_data contains the first `train_ratio` fraction of the dataset.
             - test_data contains the remaining data for testing.
    """
    train_size = int(len(scaled_data) * train_ratio)
    train_data, test_data = scaled_data[:train_size], scaled_data[train_size:]
    return train_data, test_data

def train_test_split(X, y, y_dates, train_ratio=0.8):
    """
    Splits (X, y, y_dates) into train and test sets, preserving temporal order.
    """
    seq_count = len(X)
    train_size = int(seq_count * train_ratio)

    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    dates_train, dates_test = y_dates[:train_size], y_dates[train_size:]

    return X_train, X_test, y_train, y_test, dates_train, dates_test


