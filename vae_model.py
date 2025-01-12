from models.vae import VAE, train_vae
from plotting.plotting import (plot_vae_results,
                               plot_volatility_with_anomalies, plot_volumes_with_anomalies, plot_vae_training_loss,
                               plot_market_relative_volatility_with_anomalies)
from data_preprocessing.data_preprocessing import preprocess_raw_data, scale_data, test_train_split
import numpy as np
import tensorflow as tf

def detect_anomalies_vae(vae, test_data):
    test_data_tf = tf.cast(test_data, tf.float32)
    x_recon , _, _ = vae(test_data_tf)
    x_recon = x_recon.numpy()
    mse = np.mean(np.power(test_data - x_recon, 2), axis=1)

    threshold = np.mean(mse) + 2 * np.std(mse)  # Mean + 2 * standard deviations (%95.4)
    anomalies = mse > threshold

    return mse, threshold, anomalies

def run_vae_detection_pipeline(stock_data, market_data, epochs=40):

    data = preprocess_raw_data(stock_data, market_data)
    data_scaled, scaler = scale_data(data)
    train_data, test_data = test_train_split(data_scaled, train_ratio=0.8)

    input_dim = train_data.shape[1]
    vae = VAE(input_dim=input_dim, latent_dim=2, hidden_dim=32)
    reconstruction_loss, kl_loss, total_loss = train_vae(vae, train_data, batch_size=64, epochs=epochs, learning_rate=1e-4)

    mse, threshold, anomalies = detect_anomalies_vae(vae, test_data)

    plot_vae_training_loss(reconstruction_loss, kl_loss, total_loss, epochs=epochs)
    plot_vae_results(anomalies, data, train_size=int(len(data) * 0.8), mse=mse)
    plot_volumes_with_anomalies(data=data, anomalies=anomalies,train_size=int(len(data) * 0.8))
    plot_volatility_with_anomalies(data=data,anomalies=anomalies,train_size=int(len(data) * 0.8))
    plot_market_relative_volatility_with_anomalies(data=data,anomalies=anomalies,train_size=int(len(data) * 0.8))