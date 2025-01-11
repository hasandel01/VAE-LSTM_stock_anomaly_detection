import yfinance as yf
import pandas as pd

def fetch_data(stock_symbol, market_symbol):
    """
    Fetch stock and market data from Yahoo Finance.
    """
    stock = yf.download(stock_symbol, period='2y', interval='1h')
    market = yf.download(market_symbol, period='2y', interval='1h')
    return stock, market

