import pandas as pd
from prophet import Prophet

class ProphetModel:
    def __init__(self):
        self.model = Prophet()

    def fit(self, series: pd.Series):
        df = pd.DataFrame({
            'ds': pd.to_datetime(series.index).to_numpy().reshape(-1),
            'y': series.values.reshape(-1)
        })
        self.model.fit(df)

    def forecast(self, horizon: int):
        future = self.model.make_future_dataframe(periods=horizon)
        forecast = self.model.predict(future)
        return forecast['yhat'][-horizon:].values
