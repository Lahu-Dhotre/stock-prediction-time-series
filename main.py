"""
Main entry point for testing the Time Series Forecasting module.
"""
from backend.data.data_loader import load_stock_data
from backend.data.preprocessor import preprocess_stock_data
from backend.models.arima_model import ARIMAModel
from backend.models.prophet_model import ProphetModel
from backend.models.random_forest_model import RandomForestModel
from backend.utils.evaluate import rmse
import numpy as np

def main():
    """
    Main function to test the ARIMA, Prophet, and Random Forest models.
    """
    # Prompt user for ticker symbol
    ticker = input("Enter stock ticker (e.g., AAPL): ").strip().upper()
    horizon = 7  # Forecast horizon (e.g., 7 days)

    try:
        # Step 1: Load stock data
        print(f"\n[Step 1] Loading data for {ticker}...")
        raw_data = load_stock_data(ticker, period="1y")
        print(f"Loaded {len(raw_data)} records. Sample data:\n{raw_data.head()}")

        # Step 2: Preprocess the data
        print("\n[Step 2] Preprocessing data...")
        processed_data = preprocess_stock_data(raw_data)
        print(f"Processed data sample:\n{processed_data.head()}")

        # ARIMA
        # Let's use a smart math tool called ARIMA to guess future stock prices!
        # Technical reference: ARIMA stands for AutoRegressive Integrated Moving Average.
        # It uses past prices, differencing, and moving averages to predict future values.
        print("\n[Step 3] Running ARIMA...")  # Tell us we're starting the ARIMA part
        arima = ARIMAModel(order=(1, 1, 1))  # Make an ARIMA model (like a robot that learns from numbers)
        arima.fit(processed_data['close'])  # Teach our robot using past prices
        arima_forecast = arima.forecast(horizon)  # Ask our robot to guess the next 7 prices
        print("ARIMA forecast:", arima_forecast.values)  # Show the robot's guesses
        # Save ARIMA model
        import joblib
        joblib.dump(arima.model, 'backend/models/arima_model.pkl')
        print("ARIMA model saved as backend/models/arima_model.pkl")

        # Prophet
        # Let's use another smart tool called Prophet to guess future stock prices!
        # Technical reference: Prophet is a forecasting tool developed by Facebook.
        # It is designed to handle seasonality, holidays, and missing data in time series.
        print("\n[Step 4] Running Prophet...")  # Tell us we're starting the Prophet part
        prophet = ProphetModel()  # Make a Prophet model (like a robot that knows about holidays and seasons)
        prophet.fit(processed_data['close'])  # Teach our Prophet robot using past prices
        prophet_forecast = prophet.forecast(horizon)  # Ask Prophet to guess the next 7 prices
        print("Prophet forecast:", prophet_forecast)  # Show Prophet's guesses
        # Save Prophet model
        try:
            prophet.model.save('backend/models/prophet_model')
            print("Prophet model saved as backend/models/prophet_model directory")
        except Exception:
            import joblib
            joblib.dump(prophet.model, 'backend/models/prophet_model.pkl')
            print("Prophet model saved as backend/models/prophet_model.pkl")

        # Random Forest
        # Let's use a smart tool called Random Forest to guess future stock prices!
        # Technical reference: Random Forest is a machine learning model that uses many decision trees.
        # It looks at patterns in past prices and features to make predictions.
        print("\n[Step 5] Running Random Forest...")  # Tell us we're starting the Random Forest part
        rf = RandomForestModel()  # Make a Random Forest model (like a robot that learns from lots of examples)
        # Prepare features for RF
        df = processed_data.copy()
        df['target'] = df['close'].shift(-horizon)
        df = df.dropna()
        X = df[['lag_1', 'rolling_mean', 'rolling_std', 'scaled_close']]
        y = df['target']
        rf.fit(X, y)  # Teach our Random Forest robot using past prices and features
        rf_forecast = rf.forecast(X.tail(horizon))  # Ask Random Forest to guess the next 7 prices
        print("Random Forest forecast:", rf_forecast)  # Show Random Forest's guesses
        # Save Random Forest model
        joblib.dump(rf.model, 'backend/models/random_forest_model.pkl')
        print("Random Forest model saved as backend/models/random_forest_model.pkl")

        # Model comparison
        # Get the true values for the last 'horizon' days, dropping any NaNs
        true = processed_data['close'].tail(horizon).values
        print("\nModel Comparison (RMSE):")
        # Mask NaNs in both true and predicted arrays for each model
        def valid_rmse(true_arr, pred_arr):
            true_arr = np.array(true_arr).flatten()
            pred_arr = np.array(pred_arr).flatten()
            mask = (~np.isnan(true_arr)) & (~np.isnan(pred_arr))
            return rmse(true_arr[mask], pred_arr[mask])

        arima_rmse = valid_rmse(true, arima_forecast.values)
        prophet_rmse = valid_rmse(true, prophet_forecast)
        rf_rmse = valid_rmse(true, rf_forecast)
        print(f"ARIMA RMSE: {arima_rmse:.4f}")
        print(f"Prophet RMSE: {prophet_rmse:.4f}")
        print(f"Random Forest RMSE: {rf_rmse:.4f}")

        # Find the best model
        # Let's see which robot made the best guesses! We look for the lowest RMSE (closest to real prices)
        rmse_dict = {'ARIMA': arima_rmse, 'Prophet': prophet_rmse, 'Random Forest': rf_rmse}
        best_model = min(rmse_dict, key=rmse_dict.get)  # Find the robot with the smallest RMSE
        print(f"\nBest model for this data is: {best_model} (lowest RMSE)")  # Tell us which robot was best
        print("\n[Summary for Beginners]")
        print(f"RMSE means how close the guesses are to the real prices. Lower is better!")
        print(f"In this run, {best_model} made the best guesses for future prices.")

    except Exception as e:
        print(f"\n[Error] An error occurred during the pipeline execution: {e}")

if __name__ == "__main__":
    main()