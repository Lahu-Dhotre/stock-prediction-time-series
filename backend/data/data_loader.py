"""
Data Loader Module
"""
import pandas as pd
import yfinance as yf

def load_stock_data(ticker: str, period: str = "1y") -> pd.DataFrame:
    """
    Fetch daily adjusted close price for a given stock ticker.

    Args:
        ticker (str): Stock ticker symbol.
        period (str): Period for historical data (default: "1y").

    Returns:
        pd.DataFrame: DataFrame with index as dates and a column 'close'.
    """
    try:
        # Explicitly set auto_adjust to False to include 'Adj Close'
        data = yf.download(ticker, period=period, progress=False, auto_adjust=False)
        if data.empty:
            raise ValueError(f"No data found for ticker {ticker}")
        return data[["Adj Close"]].rename(columns={"Adj Close": "close"})
    except Exception as e:
        raise RuntimeError(f"Failed to load data for {ticker}: {e}")