"""
Metrics Utility Module
"""
from statsmodels.tsa.arima.model import ARIMA

def calculate_confidence_score(model: ARIMA) -> float:
    """
    Calculate confidence score based on model's RMSE.

    Args:
        model (ARIMA): Fitted ARIMA model.

    Returns:
        float: Confidence score (0-1).
    """
    try:
        rmse = model.mse_resid ** 0.5
        # Map RMSE to a confidence score (example mapping)
        confidence = max(0, min(1, 1 - rmse / 100))
        return confidence
    except Exception as e:
        raise RuntimeError(f"Failed to calculate confidence score: {e}")