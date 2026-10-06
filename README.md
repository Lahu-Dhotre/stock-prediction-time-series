# Stock Prediction Time Series

A full-stack stock forecasting application that predicts short-term stock prices and provides a simple decision engine for buy/wait/high-risk recommendations. The project combines a FastAPI backend, a Streamlit frontend, and multiple time-series forecasting models.

## Overview

This repository analyzes stock market data using:

- ARIMA
- Prophet
- Random Forest
- Technical trend analysis
- Forecast confidence and decision scoring

The app allows users to input a stock ticker, view 1-year historical trends, and receive short-term forecasts for 7-day and 30-day horizons.

## Project Structure

```text
stock-prediction-time-series/
├── README.md
├── main.py
├── backend/
│   ├── __init__.py
│   ├── api.py
│   ├── requirements.txt
│   ├── test.py
│   ├── data/
│   │   ├── data_loader.py
│   │   └── preprocessor.py
│   ├── models/
│   │   ├── arima_model.py
│   │   ├── prophet_model.py
│   │   ├── random_forest_model.py
│   │   └── trained models (.pkl or directories)
│   ├── routes/
│   │   └── forecast.py
│   ├── services/
│   │   └── insight_engine.py
│   ├── tests/
│   └── utils/
│       └── evaluate.py
├── frontend/
│   └── app.py
└── .gitignore
```

## Features

- Stock data retrieval using Yahoo Finance
- Historical price preprocessing
- Forecasting for multiple time windows
- Model comparison using RMSE
- AI-style decision recommendation based on forecast trend
- Web dashboard for exploring stock insights
- REST API endpoint for programmatic access

## Tech Stack

- Python
- FastAPI
- Streamlit
- Pandas / NumPy
- scikit-learn
- statsmodels
- Prophet
- Plotly
- yfinance

## Getting Started

### Prerequisites

- Python 3.9+
- pip

### 1. Clone the repository

```bash
git clone https://github.com/Lahu-Dhotre/stock-prediction-time-series.git
cd stock-prediction-time-series
```

### 2. Install backend dependencies

```bash
cd backend
pip install -r requirements.txt
```

### 3. Run the backend

From the project root:

```bash
uvicorn backend.api:app --reload --host 0.0.0.0 --port 8000
```

The API will be available at:

```text
http://localhost:8000
```

### 4. Run the frontend

Open a new terminal and run:

```bash
cd frontend
streamlit run app.py
```

The Streamlit dashboard will open in the browser.

## API Endpoint

### GET /forecast

Fetches current stock details and forecast insights.

Example:

```bash
curl "http://localhost:8000/forecast?ticker=AAPL"
```

Response includes:

- company metadata
- current price
- forecast for 7-day and 30-day horizons
- trend and recommendation summary

## Example Usage

From the project root, you can also run the standalone forecasting script:

```bash
python main.py
```

This script prompts for a ticker and runs the forecasting pipeline for ARIMA, Prophet, and Random Forest models.

## Model Behavior

The system evaluates several forecasting models and compares them based on RMSE. The best-performing model is selected for the given dataset and the result is used to support the decision summary.

## Notes

- Forecasting is best used for short-term trend estimation rather than guaranteed investment advice.
- Data source depends on Yahoo Finance availability and ticker support.
- The application is intended as a demonstration project for time-series forecasting and stock analysis.

## License

This project is provided for educational and demonstration purposes.

## Contributing

Contributions, improvements, and bug fixes are welcome. Feel free to fork the repository and submit a pull request.
