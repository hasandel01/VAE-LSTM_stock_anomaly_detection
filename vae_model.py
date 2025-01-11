from models.vae import VAE, train_vae
from plotting.plotting import (plot_vae_results, plot_beta_with_anomalies,
                               plot_volatility_with_anomalies, plot_volumes_with_anomalies, plot_vae_training_loss)
from data_preprocessing.data_preprocessing import preprocess_raw_data, scale_data, test_train_split
import numpy as np
import tensorflow as tf

def detect_anomalies_vae(vae, test_data, original_data, threshold_percentile=97):
    test_data_tf = tf.cast(test_data, tf.float32)
    x_recon , _, _ = vae(test_data_tf)
    x_recon = x_recon.numpy()
    mse = np.mean(np.power(test_data - x_recon, 2), axis=1)

    threshold = np.percentile(mse, threshold_percentile)
    anomalies = mse > threshold

    stock_anomalies = []
    for i, is_anomaly in enumerate(anomalies):
        if is_anomaly:
            relative_return = original_data['relative_return'].iloc[i]
            market_return = original_data['market_return'].iloc[i]
            volume_change = original_data['volume_change'].iloc[i]
            volatility = original_data['volatility'].iloc[i]

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

    return mse, threshold, anomalies, stock_anomalies

def run_vae_detection_pipeline(stock_data, market_data):

    data = preprocess_raw_data(stock_data, market_data)
    data_scaled, scaler = scale_data(data)
    train_data, test_data = test_train_split(data_scaled, train_ratio=0.8)

    input_dim = train_data.shape[1]
    vae = VAE(input_dim=input_dim, latent_dim=2, hidden_dim=32)
    reconstruction_loss, kl_loss, total_loss = train_vae(vae, train_data, batch_size=64, epochs=10, learning_rate=1e-3)

    mse, threshold, anomalies, stock_specific_anomalies = detect_anomalies_vae(vae, test_data, original_data=data.iloc[
                                                                                                             len(train_data):],
                                                                               threshold_percentile=97)

    plot_vae_training_loss(reconstruction_loss, kl_loss, total_loss, epochs=10)
    plot_vae_results(anomalies, stock_specific_anomalies, data, train_size=int(len(data) * 0.8))

    plot_volumes_with_anomalies(data=data, anomalies=anomalies,
                                stock_specific_anomalies=stock_specific_anomalies,train_size=int(len(data) * 0.8))

    plot_volatility_with_anomalies(
        data=data,
        anomalies=anomalies,
        stock_specific_anomalies=stock_specific_anomalies,
        train_size=int(len(data) * 0.8)
    )

    plot_beta_with_anomalies(
        data=data,
        anomalies=anomalies,
        stock_specific_anomalies=stock_specific_anomalies,
        train_size=int(len(data) * 0.8)
    )