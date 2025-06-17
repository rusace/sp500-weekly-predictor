import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.ensemble import RandomForestClassifier
from datetime import datetime, timedelta

st.set_page_config(page_title="S&P 500 Weekly Predictor", layout="wide", page_icon="📈")

st.title("📈 S&P 500 Weekly Stock Predictor")
st.markdown("Predicting next week's potential winners using historical technicals + machine learning")

@st.cache_data
def get_sp500():
    table = pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")[0]
    return table["Symbol"].tolist()

@st.cache_data
def fetch_data(ticker):
    end = datetime.today()
    start = end - timedelta(days=365 * 2)
    data = yf.download(ticker, start=start, end=end)
    return data

@st.cache_data
def engineer_features(df):
    df["Return"] = df["Adj Close"].pct_change()
    df["Volatility"] = df["Return"].rolling(window=5).std()
    df["Momentum"] = df["Adj Close"] / df["Adj Close"].shift(5) - 1
    df["SMA_5"] = df["Adj Close"].rolling(window=5).mean()
    df["SMA_20"] = df["Adj Close"].rolling(window=20).mean()
    df["Signal"] = (df["Adj Close"].shift(-5) > df["Adj Close"]).astype(int)
    return df.dropna()

@st.cache_data
def train_model(df):
    X = df[["Return", "Volatility", "Momentum", "SMA_5", "SMA_20"]]
    y = df["Signal"]
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    return model

@st.cache_data
def predict_ticker(ticker):
    try:
        df = fetch_data(ticker)
        df = engineer_features(df)
        model = train_model(df)
        latest = df[["Return", "Volatility", "Momentum", "SMA_5", "SMA_20"]].iloc[-1:]
        pred = model.predict_proba(latest)[0][1]
        return pred
    except:
        return None

sp500 = get_sp500()

st.markdown("🧠 Running predictions...")

results = []
for symbol in sp500:
    prob = predict_ticker(symbol)
    if prob is not None:
        results.append((symbol, prob))

top5 = sorted(results, key=lambda x: x[1], reverse=True)[:5]

df = pd.DataFrame(top5, columns=["Ticker", "Predicted Gain Probability"])
df["Predicted Gain Probability"] = (df["Predicted Gain Probability"] * 100).round(2).astype(str) + '%'

st.success("✅ Top 5 predicted weekly gainers:")
st.dataframe(df, use_container_width=True)

st.caption("Powered by Alpha Vantage, Yahoo Finance & Random Forest Model | v1.0")
