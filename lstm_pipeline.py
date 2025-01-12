import numpy as np
from data_preprocessing.data_preprocessing import preprocess_raw_data, scale_data,train_test_split, create_sequences
from plotting.plotting import plot_lstm_results, plot_lstm_training_loss
from models.lstm import train_lstm_model, build_lstm_model
from utils.utils import *
import streamlit as st
import matplotlib.pyplot as plt

def detect_anomalies_lstm(model, X_test, y_test, threshold_percentile=97):
    """
    Detect anomalies using an LSTM for next-step prediction.

    :param model: Trained LSTM model
    :param X_test: Test sequences
    :param y_test: True next-step values (aligned with X_test)
    :param threshold_percentile: Which percentile to choose as anomaly threshold
    :return: (mse, threshold, anomalies, stock_specific_anomalies)
    """
    y_pred = model.predict(X_test)
    y_pred = y_pred.flatten()
    mse = np.mean(np.power(y_test - y_pred, 2), axis=0)

    # So let’s refine that to a per-sample error:
    errors = np.power((y_test - y_pred), 2)  # array of per-sample squared errors
    threshold_value = np.percentile(errors, threshold_percentile)

    anomalies = errors > threshold_value

    return mse, errors, threshold_value, anomalies, y_pred


def run_lstm_detection_pipeline(stock_data, market_data, lookback=35, threshold_percentile=97, epochs=40, batch_size=64):

    data = preprocess_raw_data(stock_data, market_data)
    data_scaled, scaler = scale_data(data)
    # target_col = 0 means we predict the next step of data['stock_return'] (assuming it's the first column after scaling).
    X, y, y_dates = create_sequences(pd.DataFrame(data_scaled, index=data.index), lookback=lookback)
    X_train, X_test, y_train, y_test, dates_train, dates_test = train_test_split(
        X=X, y=y, y_dates=y_dates, train_ratio=0.8
    )
    input_shape = (X_train.shape[1], X_train.shape[2])  # (lookback, num_features)
    lstm_model = build_lstm_model(input_shape)
    history = train_lstm_model(lstm_model, X_train, y_train, epochs=epochs, batch_size=batch_size, patience=5)

    plot_lstm_training_loss(history)

    mse, errors, threshold_value, anomalies, y_pred = detect_anomalies_lstm(lstm_model, X_test, y_test,
                                                                                             threshold_percentile=threshold_percentile)

    plot_lstm_results(data, dates_test, y_test, anomalies, mse)

