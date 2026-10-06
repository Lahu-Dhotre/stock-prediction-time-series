"""
FastAPI backend for stock forecasting and AI decision engine.
Enterprise-grade, clean code, with validation and error handling.
"""
from fastapi import FastAPI
from backend.routes.forecast import router as forecast_router

app = FastAPI(title="Stock Forecast & AI Decision API", version="1.0")

app.include_router(forecast_router)
