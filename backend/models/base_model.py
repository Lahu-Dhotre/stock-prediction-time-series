"""
Base Model Interface
"""
from abc import ABC, abstractmethod
import pandas as pd

class BaseTimeSeriesModel(ABC):
    """
    Abstract base class for time series models.
    """

    @abstractmethod
    def fit(self, series: pd.Series) -> None:
        """
        Fit the model to the given time series data.

        Args:
            series (pd.Series): Time series data to fit the model.
        """
        pass

    @abstractmethod
    def forecast(self, horizon: int) -> pd.Series:
        """
        Forecast future values for the given horizon.

        Args:
            horizon (int): Number of future time steps to forecast.

        Returns:
            pd.Series: Forecasted values with datetime index.
        """
        pass