import streamlit as st
import os
from lstm_model import run_lstm_detection_pipeline
from utils.utils import fetch_data
from vae_model import run_vae_detection_pipeline
from vae_lstm_model import run_vae_lstm_detection_pipeline
import traceback

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

st.title('Anomaly Detection on Stock Data')

stock_symbol = st.text_input("Enter a stock symbol from NASDAQ (e.g. , TSLA, AAPL)", value="MSFT")
market_symbol = "^IXIC"  ## NASDAQ index symbol
method = st.radio("Anomaly Detection Method:", ["LSTM Model", "Variational Autoencoder (VAE)", "VAE-LSTM Model"])

if st.button("Run Detection"):
    st.write(f"Fetching data from {market_symbol} for stock {stock_symbol}...")

    try:
        stock_data, market_data = fetch_data(stock_symbol, market_symbol)
        st.write(f"{len(stock_data)} data fetched from {market_symbol} for stock {stock_symbol}.")

        if method == "LSTM Model":
            run_lstm_detection_pipeline(stock_data, market_data)
        elif method == "Variational Autoencoder (VAE)":
            run_vae_detection_pipeline(stock_data, market_data)
        elif method == "VAE-LSTM Model":
            run_vae_lstm_detection_pipeline(stock_data, market_data)

    except Exception as e:
        st.error(e)
        traceback.print_exc()






