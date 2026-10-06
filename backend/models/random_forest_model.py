from sklearn.ensemble import RandomForestRegressor
import pandas as pd

class RandomForestModel:
    def __init__(self):
        self.model = RandomForestRegressor()

    def fit(self, X, y):
        self.model.fit(X, y)

    def forecast(self, X_future):
        return self.model.predict(X_future)
