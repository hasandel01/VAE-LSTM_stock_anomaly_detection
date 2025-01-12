import numpy as np
import tensorflow as tf
from data_preprocessing.data_preprocessing import preprocess_raw_data, scale_data, create_sequences, train_test_split
from plotting.plotting import plot_lstm_results
import streamlit as st
from models.vae_lstm import VAELSTMHybrid, train_vae_lstm_hybrid

import numpy as np
import tensorflow as tf

import numpy as np
import tensorflow as tf


def detect_anomalies_vae_lstm(model, X_test, y_test, lstm_input, original_data, threshold_percentile=97):
    log_likelihoods = []
    mse_vae = []
    lstm_predictions = []
    anomalies_vae = []
    anomalies_lstm = []

    for i in range(len(X_test)):
        # Pass input through the model
        x_recon, z_mean, z_logvar, lstm_output = model(X_test[i][np.newaxis, :], lstm_input=lstm_input[i][np.newaxis, :])
        x_recon = np.squeeze(x_recon, axis=0)  # Remove batch dimension
        lstm_output = lstm_output.numpy().flatten()[0]  # Predicted next value

        # Compute reconstruction error (MSE)
        recon_error = np.mean(np.square(X_test[i] - x_recon))
        mse_vae.append(recon_error)

        # Compute log likelihood
        variance = tf.exp(z_logvar).numpy()
        likelihood = -0.5 * np.sum(np.log(2 * np.pi * variance) + np.square(X_test[i] - np.squeeze(x_recon)) / variance)
        log_likelihoods.append(likelihood)

        # Store LSTM predictions and compute prediction error
        lstm_predictions.append(lstm_output)
        lstm_error = np.square(y_test[i] - lstm_output)

        # Identify anomalies
        anomalies_vae.append(recon_error)
        anomalies_lstm.append(lstm_error)

    # Threshold calculations
    vae_threshold = np.percentile(mse_vae, threshold_percentile)
    likelihood_threshold = np.percentile(log_likelihoods, 100 - threshold_percentile)
    lstm_threshold = np.percentile(anomalies_lstm, threshold_percentile)

    # Mark anomalies
    vae_anomalies = np.array(mse_vae) > vae_threshold
    likelihood_anomalies = np.array(log_likelihoods) < likelihood_threshold  # Low likelihood is anomalous
    lstm_anomalies = np.array(anomalies_lstm) > lstm_threshold

    combined_anomalies = np.logical_or.reduce((vae_anomalies, likelihood_anomalies, lstm_anomalies))

    print(f"VAE Reconstruction Threshold: {vae_threshold}")
    print(f"Log Likelihood Threshold: {likelihood_threshold}")
    print(f"LSTM Prediction Threshold: {lstm_threshold}")
    print(f"Total VAE Anomalies: {np.sum(vae_anomalies)}")
    print(f"Total Log Likelihood Anomalies: {np.sum(likelihood_anomalies)}")
    print(f"Total LSTM Anomalies: {np.sum(lstm_anomalies)}")
    print(f"Combined Anomalies: {np.sum(combined_anomalies)}")

    return log_likelihoods, mse_vae, lstm_predictions, combined_anomalies


import pandas as pd

def run_vae_lstm_detection_pipeline(stock_data, market_data, lookback=33, latent_dim=2, lstm_units=64, epochs=10):
    """
    End-to-end pipeline for VAE-LSTM hybrid anomaly detection.

    Parameters:
        stock_data (pd.DataFrame): Stock market data.
        market_data (pd.DataFrame): Market index data.
        lookback (int): Number of time steps in each input sequence for LSTM.
        latent_dim (int): Dimension of the latent space in the VAE.
        lstm_units (int): Number of units in the LSTM layer.
        epochs (int): Number of training epochs.

    Returns:
        None. Outputs visualizations in Streamlit.
    """
    # --- 1) Preprocess Data ---
    st.write("Preprocessing data...")
    data = preprocess_raw_data(stock_data, market_data)
    data_scaled, scaler = scale_data(data)

    # Convert scaled data back to a DataFrame with the original index
    data_scaled_df = pd.DataFrame(data_scaled, columns=data.columns, index=data.index)

    # --- 2) Create Sequences ---
    st.write("Creating sequences for LSTM...")
    X, y, y_dates = create_sequences(data_scaled_df, lookback=lookback)  # X shape: (samples, time_steps, features)

    # --- 3) Train/Test Split ---
    X_train, X_test, y_train, y_test, dates_train, dates_test = train_test_split(
        X=X, y=y, y_dates=y_dates, train_ratio=0.8
    )

    # --- 4) Initialize VAE-LSTM Model ---
    st.write("Initializing VAE-LSTM model...")
    input_dim = X_train.shape[2]
    model = VAELSTMHybrid(input_dim=input_dim, latent_dim=latent_dim, lstm_units=lstm_units)

    # --- 5) Train the Model ---
    st.write("Training VAE-LSTM model...")
    train_vae_lstm_hybrid(model, X_train, X_train, epochs=epochs)

    # --- Detect Anomalies ---
    st.write("Detecting anomalies...")
    log_likelihoods, mse_vae, lstm_predictions, combined_anomalies = detect_anomalies_vae_lstm(
        model, X_test, y_test, X_test, data.iloc[len(X_train):], threshold_percentile=97
    )

    # --- Visualize Results ---
    st.write("Visualizing results...")
    plot_lstm_results(data, dates_test, y_test, combined_anomalies, mse_vae)

