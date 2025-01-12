import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import pandas as pd
import mplfinance as mpf

def plot_vae_results(anomalies, data, train_size, mse):
    """
    Plot stock and market returns with anomalies and detailed changes.

    Parameters:
        anomalies (array): Boolean array for all anomalies.
        stock_specific_anomalies (array): Boolean array for stock-specific anomalies.
        data (pd.DataFrame): Data containing 'stock_return' and 'market_return'.
        train_size (int): Number of training samples (used to split test data).
        :param mse:
    """
    # Ensure data is a DataFrame
    if not isinstance(data, pd.DataFrame):
        raise ValueError("`data` must be a Pandas DataFrame with 'stock_return' and 'market_return' columns.")

    # Slice test data
    test_index = data.index[train_size:]
    anomaly_dates = test_index[anomalies]

    # Plot data
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(test_index, data['stock_return'][train_size:], label='Stock Returns', color='blue')
    ax.plot(test_index, data['market_return'][train_size:], label='Market Returns', color='orange')
    ax.scatter(
        anomaly_dates,
        data['stock_return'][anomaly_dates],
        color='red',
        label='Anomalies',
        zorder=5
    )

    ax.set_title("Stock and Market Returns with Stock-Specific Anomalies")
    ax.set_xlabel("Date")
    ax.set_ylabel("Returns (%)")
    ax.legend()
    st.pyplot(fig)

    # Display anomaly details
    st.write("### Anomaly Dates with Detailed Changes")
    for date in anomaly_dates:
        stock_change = data.loc[date, 'stock_return']
        market_change = data.loc[date, 'market_return']
        stock_direction = "Increase" if stock_change > 0 else "Decrease"
        market_direction = "Increase" if market_change > 0 else "Decrease"
        st.write(
            f"{date.strftime('%Y-%m-%d %H:%M:%S')} - Stock: {stock_change:.2f}% ({stock_direction}), "
            f"Market: {market_change:.2f}% ({market_direction})"
        )

    st.subheader("Mean Squared Error (MSE)")
    mean_mse = np.mean(mse)  # Calculate the mean MSE
    st.write(f"The Mean Squared Error (MSE) of the model is: **{mean_mse:.5f}**")


def plot_volumes_with_anomalies(data, anomalies, train_size):
    """
    Plots stock volumes with general and stock-specific anomalies.
    """
    test_index = data.index[train_size:]
    anomaly_dates = test_index[anomalies]

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(test_index, data['volume_change'][train_size:], label='Volume Change (%)', color='blue')

    # Mark general anomalies
    ax.scatter(
        anomaly_dates,
        data['volume_change'][anomaly_dates],
        color='red',
        label='General Anomalies',
        zorder=5
    )

    ax.set_title("Volume Changes with Anomalies")
    ax.set_xlabel("Date")
    ax.set_ylabel("Volume Change (%)")
    ax.legend()
    st.pyplot(fig)

    # Log details of anomalies
    st.write("### Volume Anomalies Details")
    for date in anomaly_dates:
        volume_change = data.loc[date, 'volume_change']
        st.write(f"{date.strftime('%Y-%m-%d %H:%M:%S')} - Volume Change: {volume_change:.2f}%")


def plot_lstm_results(
        data,
        dates_test,
        y_test,
        anomalies,
        mse
):
    """
    Plot both actual stock returns and market returns for the test set,
    overlay the LSTM predictions, and highlight anomalies.

    :param mse:
    :param data: Original DataFrame containing ['stock_return', 'market_return']
                 (indexed by dates_test).
    :param dates_test: pd.Index corresponding to each point in y_test/y_pred.
    :param y_test: Actual next-step target (e.g., stock_return).
    :param anomalies: Boolean array marking general anomalies.
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    # Plot actual stock return
    ax.plot(
        dates_test,
        data.loc[dates_test, 'stock_return'],
        label='Stock Return (Actual)',
        color='blue'
    )

    # Plot market return
    ax.plot(
        dates_test,
        data.loc[dates_test, 'market_return'],
        label='Market Return',
        color='orange'
    )

    # Highlight anomalies
    anomaly_dates = dates_test[anomalies]

    ax.scatter(
        anomaly_dates,
        data.loc[anomaly_dates, 'stock_return'],
        color='red',
        label='Anomalies (All)',
        zorder=5
    )

    ax.set_title("LSTM Predictions with Stock & Market Returns + Anomalies")
    ax.set_xlabel("Date")
    ax.set_ylabel("Returns (%)")
    ax.legend()
    st.pyplot(fig)

    # Print anomaly details in Streamlit
    st.write("### Detected Anomalies")
    for date in anomaly_dates:
        stock_change = data.loc[date, 'stock_return']
        market_change = data.loc[date, 'market_return']
        st.write(
            f"Date: {date}, "
            f"Stock: {stock_change:.2f}%, Market: {market_change:.2f}%"
        )

    st.subheader("Mean Squared Error (MSE)")
    st.write(f"The Mean Squared Error (MSE) of the model is: **{mse:.5f}**")

# Function to plot VAE training loss components
def plot_vae_training_loss(reconstruction_loss, kl_loss, total_loss, epochs):
    """
    Plot VAE training loss components with Streamlit.
    :param reconstruction_loss: List of reconstruction losses over epochs.
    :param kl_loss: List of KL divergence losses over epochs.
    :param total_loss: List of total losses over epochs.
    :param epochs: Total number of epochs.
    """
    plt.figure(figsize=(12, 6))
    plt.plot(range(1, epochs + 1), reconstruction_loss, label='Reconstruction Loss', color='blue')
    plt.plot(range(1, epochs + 1), kl_loss, label='KL Divergence Loss', color='orange')
    plt.plot(range(1, epochs + 1), total_loss, label='Total Loss', linestyle='--', color='green')
    plt.title('VAE Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid()
    st.pyplot(plt)

def plot_lstm_training_loss(history):
    """
    Plot LSTM training and validation loss using Streamlit.
    :param history: The history object returned by the model.fit method.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(history.history['loss'], label='Training Loss', color='blue')
    if 'val_loss' in history.history:
        ax.plot(history.history['val_loss'], label='Validation Loss', linestyle='--', color='orange')
    ax.set_title('LSTM Training Loss')
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Loss (MSE)')
    ax.legend()
    ax.grid()
    st.pyplot(fig)


import matplotlib.pyplot as plt
import streamlit as st

def plot_volatility_with_anomalies(data, anomalies, train_size):
    """
    Plot stock volatility with anomalies and stock-specific anomalies.

    Parameters:
        data (pd.DataFrame): Data containing 'volatility' column.
        anomalies (array): Boolean array for general anomalies.
        stock_specific_anomalies (array): Boolean array for stock-specific anomalies.
        train_size (int): Number of training samples (used to split test data).
    """
    test_index = data.index[train_size:]
    anomaly_dates = test_index[anomalies]

    # Plot volatility
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(test_index, data['volatility'][train_size:], label='Volatility', color='blue')

    # Highlight anomalies
    ax.scatter(anomaly_dates, data['volatility'][anomaly_dates], color='red', label='General Anomalies', zorder=5)
    ax.scatter(anomaly_dates, data['volatility'][anomaly_dates], color='green',
               label='Stock-Specific Anomalies', zorder=5)

    # Customize the plot
    ax.set_title("Stock Volatility with Anomalies")
    ax.set_xlabel("Date")
    ax.set_ylabel("Volatility")
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)

    # Log details of anomalies
    st.write("### Volatility Anomalies Details")
    for date in anomaly_dates:
        volatility_value = data.loc[date, 'volatility']
        st.write(f"{date.strftime('%Y-%m-%d %H:%M:%S')} - Volatility: {volatility_value:.2f}")


import matplotlib.pyplot as plt
import streamlit as st


def plot_market_relative_volatility_with_anomalies(data, anomalies, train_size):
    """
    Plots market relative volatility with anomalies scattered.

    Parameters:
        data (pd.DataFrame): The dataset containing features.
        anomalies (np.ndarray): Boolean array indicating anomaly status.
        train_size (int): Size of the training dataset for indexing.
    """
    # Get the test data index
    test_index = data.index[train_size:]
    anomaly_dates = test_index[anomalies]

    fig, ax = plt.subplots(figsize=(12, 6))

    # Plot market relative volatility
    ax.plot(test_index, data['market_relative_volatility'][train_size:], label='Market Relative Volatility',
            color='blue')

    # Scatter anomalies
    ax.scatter(
        anomaly_dates,
        data['market_relative_volatility'][anomaly_dates],
        color='red',
        label='Anomalies',
        zorder=5
    )

    # Add labels, title, and legend
    ax.set_title("Market Relative Volatility with Anomalies")
    ax.set_xlabel("Date")
    ax.set_ylabel("Market Relative Volatility")
    ax.legend()
    ax.grid(True)

    # Display the plot in Streamlit
    st.pyplot(fig)

    # Log anomaly details
    st.write("### Market Relative Volatility Anomalies Details")
    for date in anomaly_dates:
        market_relative_volatility = data.loc[date, 'market_relative_volatility']
        st.write(f"{date.strftime('%Y-%m-%d %H:%M:%S')} - Market Relative Volatility: {market_relative_volatility:.4f}")




