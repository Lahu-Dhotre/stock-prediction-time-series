
import os
import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import numpy as np
from datetime import datetime, timedelta

BASE_URL = os.getenv("BASE_URL", "http://localhost:8000")

def get_stock_insights(ticker):
    url = f"{BASE_URL}/forecast?ticker={ticker}"
    try:
        resp = requests.get(url, timeout=10)
        if resp.status_code != 200:
            st.error(f"Backend error: {resp.status_code} {resp.reason}")
            return None
        return resp.json()
    except Exception as e:
        st.error(f"Connection error: {e}")
        return None

def synthetic_history(price, days=365, vol=0.02):
    np.random.seed(42)
    prices = [price]
    for _ in range(days-1):
        prices.append(prices[-1] * (1 + np.random.normal(0, vol)))
    dates = [datetime.today() - timedelta(days=days-i-1) for i in range(days)]
    return pd.DataFrame({"date": dates, "price": prices, "type": "History"})

def main():
    st.set_page_config(page_title="Stock Insights", layout="wide")
    st.title("Stock Insights")
    st.markdown("#### 1-Year History • Forecast • Trend • Decision")

    with st.form("search"):
        ticker = st.text_input("Stock Ticker", value="AAPL").strip().upper()
        horizon = st.radio("Forecast Horizon", ["7 days", "30 days"])
        go = st.form_submit_button("Get Insights")
    if not go:
        st.info("Enter a stock ticker and click 'Get Insights'.")
        return

    data = get_stock_insights(ticker)
    if not data:
        return

    company = data.get("company", {})
    current_price = data.get("current_price", None)
    forecast = data.get("forecast", {}).get("7_days" if horizon=="7 days" else "30_days", {})
    predicted = forecast.get("predicted_prices", [])

    # History
    if "history" in data and isinstance(data["history"], list) and data["history"]:
        hist = pd.DataFrame(data["history"])
        hist["date"] = pd.to_datetime(hist["date"])
        hist["type"] = "History"
    else:
        hist = synthetic_history(current_price)

    # Forecast
    last_date = hist["date"].iloc[-1]
    f_dates = [last_date + timedelta(days=i+1) for i in range(len(predicted))]
    f_df = pd.DataFrame({"date": f_dates, "price": [p if p is not None else None for p in predicted], "type": "Forecast"})
    chart_df = pd.concat([hist, f_df], ignore_index=True)

    fig = px.line(chart_df, x="date", y="price", color="type", line_dash="type")
    fig.update_traces(connectgaps=False)
    fig.update_layout(height=400, margin=dict(l=10, r=10, t=30, b=10), legend_title_text="")

    main_col, side_col = st.columns([3, 1])
    with main_col:
        st.plotly_chart(fig, use_container_width=True)

    with side_col:
        st.markdown(f"### {company.get('name', 'N/A')}")
        st.write(f"Sector: {company.get('sector', 'N/A')}")
        st.write(f"Industry: {company.get('industry', 'N/A')}")
        st.write(f"Market Cap: {company.get('market_cap', 'N/A')}")
        st.write(f"PE Ratio: {company.get('pe_ratio', 'N/A')}")
        st.metric("Current Price", f"${current_price:,.2f}" if current_price else "N/A")
        st.markdown("---")
        trend = forecast.get("trend", "SIDEWAYS")
        arrow = {"UP": "⬆️", "DOWN": "⬇️", "SIDEWAYS": "➡️"}.get(trend.upper(), "❓")
        st.write(f"Trend: {arrow} {trend}")
        pct = forecast.get("percent_change", 0.0)
        if pct is None:
            st.write("Percent Change: N/A")
        else:
            st.write(f"Percent Change: {pct:+.2%}")
        st.write(f"Volatility: {forecast.get('volatility', 'N/A')}")
        conf = forecast.get("confidence", 0.0)
        st.write("Confidence:")
        st.progress(min(max(conf if conf is not None else 0.0, 0.0), 1.0))
        st.write(f"{(conf if conf is not None else 0.0)*100:.1f}%")

    # Decision Card
    decision = forecast.get("decision", "WAIT")
    explanation = forecast.get("explanation", "No explanation provided.")
    color = {"BUY": "#27ae60", "WAIT": "#f39c12", "HIGH RISK": "#e74c3c"}.get(decision.upper(), "#7f8c8d")
    st.markdown(f"""
    <div style='background:{color};color:white;border-radius:12px;padding:24px 18px 12px 18px;margin-top:16px;box-shadow:0 2px 8px rgba(0,0,0,0.08);text-align:center;'>
    <h2 style='margin-bottom:8px;'>{decision}</h2>
    <p style='font-size:1.1em;margin-bottom:0;'>{explanation}</p>
    </div>
    """, unsafe_allow_html=True)

    # Stats
    st.markdown("### 1-Year Stats")
    min_price = hist['price'].min()
    max_price = hist['price'].max()
    avg_price = hist['price'].mean()
    stats = []
    stats.append(f"- **Min Price:** ${min_price:,.2f}" if min_price is not None else "- **Min Price:** N/A")
    stats.append(f"- **Max Price:** ${max_price:,.2f}" if max_price is not None else "- **Max Price:** N/A")
    stats.append(f"- **Avg Price:** ${avg_price:,.2f}" if avg_price is not None else "- **Avg Price:** N/A")
    st.markdown("\n".join(stats))

if __name__ == "__main__":
    main()
