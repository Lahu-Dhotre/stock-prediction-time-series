"""
ARIMA Model Implementation
"""
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from .base_model import BaseTimeSeriesModel
import pickle

class ARIMAModel(BaseTimeSeriesModel):
    """
    ARIMA model for time series forecasting.
    """

    def __init__(self, order: tuple[int, int, int] = (1, 1, 1)):
        """
        Initialize the ARIMA model with the given order.

        Args:
            order (tuple[int, int, int]): ARIMA (p, d, q) parameters.
        """
        self.order = order
        self.model = None

    def fit(self, series: pd.Series) -> None:
        """
        Fit the ARIMA model to the given time series data.

        Args:
            series (pd.Series): Time series data to fit the model.
        """
        self.model = ARIMA(series, order=self.order).fit()

    def forecast(self, horizon: int) -> pd.Series:
        """
        Forecast future values for the given horizon.

        Args:
            horizon (int): Number of future time steps to forecast.

        Returns:
            pd.Series: Forecasted values with datetime index.
        """
        if self.model is None:
            raise ValueError("Model must be fitted before forecasting.")

        forecast = self.model.get_forecast(steps=horizon)
        forecast_index = pd.date_range(start=self.model.data.dates[-1], periods=horizon + 1, freq='D')[1:]
        return pd.Series(forecast.predicted_mean, index=forecast_index)

    def save_model(self, file_path: str) -> None:
        """
        Save the fitted ARIMA model to a file.

        Args:
            file_path (str): Path to save the serialized model.
        """
        if self.model is None:
            raise ValueError("Model must be fitted before saving.")
        with open(file_path, 'wb') as f:
            pickle.dump(self.model, f)

    @staticmethod
    def load_model(file_path: str) -> 'ARIMAModel':
        """
        Load a serialized ARIMA model from a file.

        Args:
            file_path (str): Path to the serialized model file.

        Returns:
            ARIMAModel: Instance of ARIMAModel with the loaded model.
        """
        with open(file_path, 'rb') as f:
            model = pickle.load(f)
        instance = ARIMAModel()
        instance.model = model
        return instance