import numpy as np
from data_preprocessing.data_preprocessing import preprocess_raw_data, scale_data,train_test_split, create_sequences
from plotting.plotting import plot_lstm_results, plot_lstm_training_loss
from models.lstm import train_lstm_model, build_lstm_model
from utils.utils import *


def detect_anomalies_lstm(model, X_test, y_test, original_data, threshold_percentile=97):
    """
    Detect anomalies using an LSTM for next-step prediction.

    :param model: Trained LSTM model
    :param X_test: Test sequences
    :param y_test: True next-step values (aligned with X_test)
    :param original_data: Pandas DataFrame containing the test portion of the data
                          that aligns with y_test. (e.g. data.iloc[test_start_index_for_labels:])
    :param threshold_percentile: Which percentile to choose as anomaly threshold
    :return: (mse, threshold, anomalies, stock_specific_anomalies)
    """
    # 1) Predict
    y_pred = model.predict(X_test)
    y_pred = y_pred.flatten()

    # 2) Compute MSE
    mse = np.mean(np.power(y_test - y_pred, 2), axis=0)

    # 3) Define threshold
    threshold = np.percentile(mse, threshold_percentile)  # for a single MSE value, you might also do MSE per point
    # In many time-series LSTM approaches, we compute MSE *per test sample*, not aggregated.
    # So let’s refine that to a per-sample error:
    errors = np.power((y_test - y_pred), 2)  # array of per-sample squared errors
    threshold_value = np.percentile(errors, threshold_percentile)

    # 4) Mark anomalies
    anomalies = errors > threshold_value

    # 5) Apply additional "stock-specific" logic
    stock_anomalies = []
    # Make sure original_data length matches the length of `anomalies`.
    # If your sequences have a certain offset, be sure you pass the correctly aligned portion of `original_data`.
    for i, is_anomaly in enumerate(anomalies):
        if is_anomaly:
            row = original_data.iloc[i]  # e.g. row containing relative_return, market_return, ...
            relative_return = row['relative_return']
            market_return = row['market_return']
            volume_change = row['volume_change']
            volatility = row['volatility']

            # Reuse your logic
            if (
                    abs(market_return) < abs(relative_return)
                    and volume_change > np.percentile(original_data['volume_change'], 90)
                    and volatility > np.percentile(original_data['volatility'], 90)
            ):
                stock_anomalies.append(True)
            else:
                stock_anomalies.append(False)
        else:
            stock_anomalies.append(False)

    return mse, errors, threshold_value, anomalies, stock_anomalies, y_pred


def run_lstm_detection_pipeline(stock_data, market_data, lookback=35, target_col=0,
                       threshold_percentile=97, epochs=30, batch_size=64):
    """
    End-to-end LSTM anomaly detection based on next-step prediction.
    """
    # --- 1) Preprocess & Scale ---
    data = preprocess_raw_data(stock_data, market_data)
    data_scaled, scaler = scale_data(data)

    # --- 2) Create Sequences ---
    # target_col = 0 means we predict the next step of data['stock_return'] (assuming it's the first column after scaling).
    X, y, y_dates = create_sequences(pd.DataFrame(data_scaled, index=data.index), lookback=lookback)

    # --- 3) Train/Test Split ---
    X_train, X_test, y_train, y_test, dates_train, dates_test = train_test_split(
        X=X, y=y, y_dates=y_dates, train_ratio=0.8
    )

    # --- 4) Build & Train LSTM ---
    input_shape = (X_train.shape[1], X_train.shape[2])  # (lookback, num_features)
    lstm_model = build_lstm_model(input_shape)
    history = train_lstm_model(lstm_model, X_train, y_train, epochs=epochs, batch_size=batch_size, patience=10)
    plot_lstm_training_loss(history)

    # --- 5) Detect Anomalies ---
    # Slice the original data so it aligns with the test portion’s labels
    test_data_for_anomalies = data.loc[dates_test]
    mse, errors, threshold_value, anomalies, stock_anomalies, y_pred = detect_anomalies_lstm(
        lstm_model,
        X_test,
        y_test,
        original_data=test_data_for_anomalies,
        threshold_percentile=threshold_percentile
    )

    # --- 6) Plot Results: stock + market returns + anomalies + predictions ---
    plot_lstm_results(data, dates_test, y_test, y_pred, anomalies, stock_anomalies, mse)



