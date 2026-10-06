import sys
import os
# Ensure project root is in PYTHONPATH for imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
"""
Test script for loading and evaluating saved time series models.
"""
import joblib
import numpy as np
import pandas as pd
from backend.data.data_loader import load_stock_data
from backend.data.preprocessor import preprocess_stock_data
from backend.utils.evaluate import rmse

import sys
# Prompt user for ticker symbol
ticker = input("Enter stock ticker (e.g., AAPL): ").strip().upper()
horizon = 7
print(f"Loading test data for {ticker}...")
raw_data = load_stock_data(ticker, period="1mo")  # Use a different period for testing
processed_data = preprocess_stock_data(raw_data)

# Prepare true values
true = processed_data['close'].tail(horizon).values

# Load ARIMA model
arima_model = joblib.load('backend/models/arima_model.pkl')
arima_forecast = arima_model.forecast(steps=horizon)
print("ARIMA forecast:", arima_forecast)

# Load Prophet model
try:
    prophet_model = joblib.load('backend/models/prophet_model.pkl')
    # Prophet forecast
    future = prophet_model.make_future_dataframe(periods=horizon)
    prophet_forecast_df = prophet_model.predict(future)
    prophet_forecast = prophet_forecast_df['yhat'].tail(horizon).values
    print("Prophet forecast:", prophet_forecast)
except Exception as e:
    print("Could not load Prophet model or forecast:", e)
    prophet_forecast = np.full(horizon, np.nan)

# Load Random Forest model
rf_model = joblib.load('backend/models/random_forest_model.pkl')
df = processed_data.copy()
df['target'] = df['close'].shift(-horizon)
df = df.dropna()
X = df[['lag_1', 'rolling_mean', 'rolling_std', 'scaled_close']]
rf_forecast = rf_model.predict(X.tail(horizon))
print("Random Forest forecast:", rf_forecast)

# RMSE comparison with NaN masking
def valid_rmse(true_arr, pred_arr):
    true_arr = np.array(true_arr).flatten()
    pred_arr = np.array(pred_arr).flatten()
    mask = (~np.isnan(true_arr)) & (~np.isnan(pred_arr))
    return rmse(true_arr[mask], pred_arr[mask])

print("\nModel Comparison (RMSE):")
arima_rmse = valid_rmse(true, arima_forecast)
prophet_rmse = valid_rmse(true, prophet_forecast)
rf_rmse = valid_rmse(true, rf_forecast)
print(f"ARIMA RMSE: {arima_rmse:.4f}")
print(f"Prophet RMSE: {prophet_rmse:.4f}")
print(f"Random Forest RMSE: {rf_rmse:.4f}")

rmse_dict = {'ARIMA': arima_rmse, 'Prophet': prophet_rmse, 'Random Forest': rf_rmse}
best_model = min(rmse_dict, key=rmse_dict.get)
print(f"\nBest model for this test data is: {best_model} (lowest RMSE)")
