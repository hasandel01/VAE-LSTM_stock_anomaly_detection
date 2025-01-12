import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Input

import tensorflow as tf
import tensorflow.keras.layers as layers
import numpy as np


def build_lstm_model(input_shape, units=64, dense_units=32, learning_rate=1e-5):
    """
    Build a simple LSTM model for next-step prediction (Seq2One) with dynamic hyperparameters.
    input_shape: (lookback, num_features)
    units: Number of units in the LSTM layer.
    dense_units: Number of units in the dense layer.
    learning_rate: Learning rate for the Adam optimizer.
    """
    model = tf.keras.Sequential([
        layers.InputLayer(input_shape=input_shape),
        layers.LSTM(units, return_sequences=False),
        layers.Dense(dense_units, activation='relu'),
        layers.Dense(1, activation='linear')  # predicting a single value
    ])

    model.compile(
        loss='mse',
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate)
    )
    return model


from tensorflow.keras.callbacks import EarlyStopping


def train_lstm_model(model, X_train, y_train, epochs=40, batch_size=64, patience=5):
    """
    Trains the LSTM model with EarlyStopping to prevent overfitting.

    Parameters:
        model: Compiled LSTM model.
        X_train: Training input sequences.
        y_train: Training labels.
        epochs: Maximum number of epochs.
        batch_size: Batch size for training.
        patience: Number of epochs with no improvement to wait before stopping.

    Returns:
        history: Training history object containing loss and validation loss per epoch.
    """
    # EarlyStopping callback
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=3,
        restore_best_weights=True,
        verbose=1  # Verbose logs when training stops
    )

    # Train the model
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.2,  # Use 10% of the training data for validation
        callbacks=[early_stopping],  # Add EarlyStopping
        verbose=1
    )

    return history

