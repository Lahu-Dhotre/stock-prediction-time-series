"""
Data Preprocessor Module

Summary of preprocessing steps:
1. Ensure daily frequency and forward-fill missing values.
2. Remove weekends (Saturday and Sunday).
3. Add lag feature: previous day's closing price (lag_1).
4. Normalize closing price to [0, 1] using MinMaxScaler.
5. Drop rows with NaN values introduced by lagging.
"""

import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.stattools import adfuller


def check_stationarity(series: pd.Series) -> bool:
    """
    Check if a time series is stationary using the Augmented Dickey-Fuller test.
    Prints the p-value and returns True if stationary, False otherwise.
    """
    result = adfuller(series.dropna())
    p_value = result[1]
    print(f"Stationarity test (ADF): p-value = {p_value:.4f}")
    if p_value < 0.05:
        print("Series is likely stationary (reject H0)")
        return True
    else:
        print("Series is likely NOT stationary (fail to reject H0)")
        return False


def preprocess_stock_data(data: pd.DataFrame) -> pd.DataFrame:
    # Ensure daily frequency and fill missing values
    data = data.asfreq('D').ffill()

    # Remove weekends (Saturday = 5, Sunday = 6)
    data = data[data.index.dayofweek < 5]

    # Add lag feature: previous day's closing price
    data['lag_1'] = data['close'].shift(1)

    # Add rolling statistics: 5-day rolling mean and std
    data['rolling_mean'] = data['close'].rolling(window=5).mean()
    data['rolling_std'] = data['close'].rolling(window=5).std()

    # Normalize closing price to [0, 1]
    scaler = MinMaxScaler()
    data['scaled_close'] = scaler.fit_transform(data[['close']])

    # Drop rows with NaN values introduced by lagging/rolling
    data = data.dropna()

    # Check stationarity
    check_stationarity(data['close'])

    return data