from fastapi import APIRouter, Query, HTTPException
from fastapi.responses import JSONResponse
import yfinance as yf
from backend.services.insight_engine import generate_insights

router = APIRouter()

@router.get("/forecast")
def forecast(ticker: str = Query(..., min_length=1, max_length=10)):
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        if not info or 'regularMarketPrice' not in info:
            raise HTTPException(status_code=404, detail="Invalid or unsupported ticker.")
        current_price, insights = generate_insights(ticker, periods=(7, 30))
        company = {
            "name": info.get("shortName"),
            "industry": info.get("industry"),
            "sector": info.get("sector"),
            "market_cap": info.get("marketCap"),
            "pe_ratio": info.get("trailingPE"),
        }
        result = {
            "ticker": ticker,
            "company": company,
            "current_price": float(current_price),
            "forecast": {
                "7_days": insights[7],
                "30_days": insights[30]
            }
        }
        return JSONResponse(content=result)
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")
