import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import pandas as pd

def plot_vae_results(anomalies, stock_specific_anomalies, data, train_size):
    """
    Plot stock and market returns with anomalies and detailed changes.

    Parameters:
        anomalies (array): Boolean array for all anomalies.
        stock_specific_anomalies (array): Boolean array for stock-specific anomalies.
        data (pd.DataFrame): Data containing 'stock_return' and 'market_return'.
        train_size (int): Number of training samples (used to split test data).
    """
    # Ensure data is a DataFrame
    if not isinstance(data, pd.DataFrame):
        raise ValueError("`data` must be a Pandas DataFrame with 'stock_return' and 'market_return' columns.")

    # Slice test data
    test_index = data.index[train_size:]
    anomaly_dates = test_index[anomalies]
    stock_specific_dates = test_index[stock_specific_anomalies]

    # Plot data
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(test_index, data['stock_return'][train_size:], label='Stock Returns', color='blue')
    ax.plot(test_index, data['market_return'][train_size:], label='Market Returns', color='orange')
    ax.scatter(
        anomaly_dates,
        data['stock_return'][anomaly_dates],
        color='red',
        label='Anomalies (All)',
        zorder=5
    )
    ax.scatter(
        stock_specific_dates,
        data['stock_return'][stock_specific_dates],
        color='green',
        label='Stock-Specific Anomalies',
        zorder=5
    )

    for date in stock_specific_dates:
        stock_change = data.loc[date, 'stock_return']
        direction = "Increase" if stock_change > 0 else "Decrease"
        ax.annotate(
            f"{direction} {stock_change:.2f}%",
            (date, data.loc[date, 'stock_return']),
            textcoords="offset points",
            xytext=(0, 10),
            ha='center',
            fontsize=9,
            color='green'
        )

    ax.set_title("Stock and Market Returns with Stock-Specific Anomalies")
    ax.set_xlabel("Date")
    ax.set_ylabel("Returns (%)")
    ax.legend()
    st.pyplot(fig)

    # Display anomaly details
    st.write("### Anomaly Dates with Detailed Changes")
    for date in stock_specific_dates:
        stock_change = data.loc[date, 'stock_return']
        market_change = data.loc[date, 'market_return']
        stock_direction = "Increase" if stock_change > 0 else "Decrease"
        market_direction = "Increase" if market_change > 0 else "Decrease"
        st.write(
            f"{date.strftime('%Y-%m-%d %H:%M:%S')} - Stock: {stock_change:.2f}% ({stock_direction}), "
            f"Market: {market_change:.2f}% ({market_direction})"
        )


def plot_volumes_with_anomalies(data, anomalies, stock_specific_anomalies, train_size):
    """
    Plots stock volumes with general and stock-specific anomalies.
    """
    test_index = data.index[train_size:]
    anomaly_dates = test_index[anomalies]
    stock_specific_dates = test_index[stock_specific_anomalies]

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

    # Mark stock-specific anomalies
    ax.scatter(
        stock_specific_dates,
        data['volume_change'][stock_specific_dates],
        color='green',
        label='Stock-Specific Anomalies',
        zorder=5
    )

    ax.set_title("Volume Changes with Anomalies")
    ax.set_xlabel("Date")
    ax.set_ylabel("Volume Change (%)")
    ax.legend()
    st.pyplot(fig)

    # Log details of anomalies
    st.write("### Volume Anomalies Details")
    for date in stock_specific_dates:
        volume_change = data.loc[date, 'volume_change']
        st.write(f"{date.strftime('%Y-%m-%d %H:%M:%S')} - Volume Change: {volume_change:.2f}%")

def plot_lstm_results(
        data,
        dates_test,
        y_test,
        y_pred,
        anomalies,
        stock_specific_anomalies,
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
    :param y_pred: LSTM-predicted next-step target.
    :param anomalies: Boolean array marking general anomalies.
    :param stock_specific_anomalies: Boolean array marking stock-specific anomalies.
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

    # (Optional) Plot the predicted next-step stock return, if you want
    # to visualize how the LSTM tracks the actual returns.
    ax.plot(
        dates_test,
        y_pred,
        label='Predicted Stock Return (LSTM)',
        color='purple',
        linestyle='--'
    )

    # Highlight anomalies
    anomaly_dates = dates_test[anomalies]
    stock_specific_dates = dates_test[stock_specific_anomalies]

    ax.scatter(
        anomaly_dates,
        data.loc[anomaly_dates, 'stock_return'],
        color='red',
        label='Anomalies (All)',
        zorder=5
    )
    ax.scatter(
        stock_specific_dates,
        data.loc[stock_specific_dates, 'stock_return'],
        color='green',
        label='Stock-Specific Anomalies',
        zorder=5
    )

    ax.set_title("LSTM Predictions with Stock & Market Returns + Anomalies")
    ax.set_xlabel("Date")
    ax.set_ylabel("Returns (%)")
    ax.legend()
    st.pyplot(fig)

    # Print anomaly details in Streamlit
    st.write("### Detected Anomalies")
    for date in stock_specific_dates:
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

def plot_volatility_with_anomalies(data, anomalies, stock_specific_anomalies, train_size):
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
    stock_specific_dates = test_index[stock_specific_anomalies]

    # Plot volatility
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(test_index, data['volatility'][train_size:], label='Volatility', color='blue')

    # Highlight anomalies
    ax.scatter(anomaly_dates, data['volatility'][anomaly_dates], color='red', label='General Anomalies', zorder=5)
    ax.scatter(stock_specific_dates, data['volatility'][stock_specific_dates], color='green',
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
    for date in stock_specific_dates:
        volatility_value = data.loc[date, 'volatility']
        st.write(f"{date.strftime('%Y-%m-%d %H:%M:%S')} - Volatility: {volatility_value:.2f}")

def plot_beta_with_anomalies(data, anomalies, stock_specific_anomalies, train_size):
    """
    Plot stock beta values with anomalies and stock-specific anomalies.

    Parameters:
        data (pd.DataFrame): Data containing 'beta' column.
        anomalies (array): Boolean array for general anomalies.
        stock_specific_anomalies (array): Boolean array for stock-specific anomalies.
        train_size (int): Number of training samples (used to split test data).
    """
    test_index = data.index[train_size:]
    anomaly_dates = test_index[anomalies]
    stock_specific_dates = test_index[stock_specific_anomalies]

    # Plot beta values
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(test_index, data['beta'][train_size:], label='Beta', color='blue')

    # Highlight anomalies
    ax.scatter(anomaly_dates, data['beta'][anomaly_dates], color='red', label='General Anomalies', zorder=5)
    ax.scatter(stock_specific_dates, data['beta'][stock_specific_dates], color='green',
               label='Stock-Specific Anomalies', zorder=5)

    # Customize the plot
    ax.set_title("Stock Beta with Anomalies")
    ax.set_xlabel("Date")
    ax.set_ylabel("Beta")
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)

    # Log details of anomalies
    st.write("### Beta Anomalies Details")
    for date in stock_specific_dates:
        beta_value = data.loc[date, 'beta']
        st.write(f"{date.strftime('%Y-%m-%d %H:%M:%S')} - Beta: {beta_value:.2f}")



