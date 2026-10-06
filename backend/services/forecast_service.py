"""
Forecast Service Module
"""
import pandas as pd
from ..data.data_loader import load_stock_data
from ..data.preprocessor import preprocess_data
from ..models.arima_model import ARIMAModel
from ..utils.volatility import calculate_volatility
from ..utils.metrics import calculate_confidence_score

def run_forecast(ticker: str, horizon_days: int) -> dict:
    """
    Run the forecasting pipeline for a given stock ticker.

    Args:
        ticker (str): Stock ticker symbol.
        horizon_days (int): Forecast horizon (e.g., 7 or 30 days).

    Returns:
        dict: Forecast results including trend, volatility, and confidence score.
    """
    try:
        # Load and preprocess data
        raw_data = load_stock_data(ticker)
        data = preprocess_data(raw_data)

        # Fit ARIMA model
        model = ARIMAModel()
        model.fit(data['close'])

        # Forecast
        forecast = model.forecast(horizon_days)

        # Calculate metrics
        last_close = data['close'].iloc[-1]
        forecast_last = forecast.iloc[-1]
        percent_change = ((forecast_last - last_close) / last_close) * 100

        trend = "UP" if forecast_last > last_close else "DOWN" if forecast_last < last_close else "SIDEWAYS"
        volatility = calculate_volatility(data['close'])
        confidence_score = calculate_confidence_score(model)

        return {
            "ticker": ticker,
            "last_close": last_close,
            "forecast": forecast.tolist(),
            "trend": trend,
            "percent_change": percent_change,
            "volatility": volatility,
            "confidence_score": confidence_score
        }
    except Exception as e:
        raise RuntimeError(f"Forecasting failed for {ticker}: {e}")