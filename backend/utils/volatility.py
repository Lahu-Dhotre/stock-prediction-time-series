"""
Volatility Utility Module
"""
import pandas as pd

def calculate_volatility(series: pd.Series) -> str:
    """
    Calculate volatility level based on standard deviation.

    Args:
        series (pd.Series): Time series data.

    Returns:
        str: Volatility level (LOW, MEDIUM, HIGH).
    """
    try:
        std_dev = series.pct_change().std() * 100  # Convert to percentage
        if std_dev < 1:
            return "LOW"
        elif std_dev < 2:
            return "MEDIUM"
        else:
            return "HIGH"
    except Exception as e:
        raise RuntimeError(f"Failed to calculate volatility: {e}")