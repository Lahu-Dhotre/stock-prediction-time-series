"""Evaluation Utilities Module"""
from sklearn.metrics import mean_squared_error

def rmse(true, pred):
    """Calculate Root Mean Squared Error between true and predicted values."""
    return mean_squared_error(true, pred) ** 0.5  # Take the square root manually