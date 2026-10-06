import numpy as np
from backend.models.arima_model import ARIMAModel
from backend.utils.evaluate import rmse
from backend.data.data_loader import load_stock_data
from backend.data.preprocessor import preprocess_stock_data

# Decision logic
def get_decision(trend, percent_change, volatility):
    if trend == "UP" and percent_change > 0 and volatility in ["LOW", "MEDIUM"]:
        return "BUY", "The model predicts an upward trend with manageable volatility."
    elif trend == "DOWN" or volatility == "HIGH":
        return "HIGH RISK", "The model predicts a downward trend or high volatility. Exercise caution."
    else:
        return "WAIT", "The model predicts sideways movement or uncertain conditions. Waiting is advised."

# Volatility level
def get_volatility(preds):
    std = np.std(preds)
    if std < 1:
        return "LOW"
    elif std < 5:
        return "MEDIUM"
    else:
        return "HIGH"

# Trend direction
def get_trend(current, future):
    if future > current * 1.01:
        return "UP"
    elif future < current * 0.99:
        return "DOWN"
    else:
        return "SIDEWAYS"

# Main insight engine
def generate_insights(ticker, periods=(7, 30)):
    raw_data = load_stock_data(ticker, period="1y")
    processed_data = preprocess_stock_data(raw_data)
    current_price = processed_data['close'].iloc[-1]
    arima = ARIMAModel(order=(1, 1, 1))
    arima.fit(processed_data['close'])
    results = {}
    for horizon in periods:
        forecast = arima.forecast(horizon).values
        future = float(forecast[-1])
        percent_change = float((future - current_price) / current_price * 100)
        volatility = get_volatility(forecast)
        trend = get_trend(float(current_price), future)
        decision, explanation = get_decision(str(trend), percent_change, str(volatility))
        true = processed_data['close'].tail(horizon).values
        # Mask NaNs in both arrays
        true_arr = np.array(true).flatten()
        forecast_arr = np.array(forecast).flatten()
        mask = (~np.isnan(true_arr)) & (~np.isnan(forecast_arr))
        confidence = float(1 / (rmse(true_arr[mask], forecast_arr[mask]) + 1e-6))
        def safe_json(val):
            if isinstance(val, float) and (np.isnan(val) or np.isinf(val)):
                return None
            if isinstance(val, list):
                return [safe_json(x) for x in val]
            return val
        results[horizon] = {
            "predicted_prices": safe_json(forecast.tolist()),
            "trend": safe_json(trend),
            "percent_change": safe_json(percent_change),
            "volatility": safe_json(volatility),
            "confidence": safe_json(confidence),
            "decision": safe_json(decision),
            "explanation": safe_json(explanation)
        }
    return current_price, results
"""
Insight Engine Module
"""
def generate_suggestions(percent_change: float, volatility: str, confidence_score: float) -> dict:
    """
    Generate insights and recommendations based on forecast metrics.

    Args:
        percent_change (float): Percent change in forecast vs last close.
        volatility (str): Volatility level (LOW, MEDIUM, HIGH).
        confidence_score (float): Confidence score (0-1).

    Returns:
        dict: Insights including suggestion and plain-English explanation.
    """
    if confidence_score >= 0.70:
        if volatility in ["LOW", "MEDIUM"] and percent_change > 0:
            suggestion = "BUY"
            insight_text = "Stock is expected to rise steadily with moderate volatility. Good for medium-term buyers."
        elif volatility == "HIGH" or percent_change < 0:
            suggestion = "HIGH_RISK"
            insight_text = "Stock shows high volatility or downward trend. Exercise caution."
        else:
            suggestion = "WAIT"
            insight_text = "Stock is stable but lacks strong confidence for buying."
    else:
        suggestion = "WAIT"
        insight_text = "Confidence is too low to make a recommendation."

    return {
        "suggestion": suggestion,
        "insight_text": insight_text
    }