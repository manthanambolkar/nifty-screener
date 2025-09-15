# app.py
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import requests
import time
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

st.set_page_config(page_title="NIFTY Live Screener", layout="wide")

@st.cache_data(ttl=3600)
def get_nifty50_list():
    # Source: Nifty Indices / TradingView / Yahoo list — fallback to Wikipedia if needed
    # We'll try a reliable JSON/HTML source; if fails, fall back to a static known list.
    try:
        # Example: get components from TradingView page (scrape simplified table) OR use a maintained list
        tv_url = "https://s3.tradingview.com/external-embedding/embed-widget-technical-analysis.js"
        # Simpler: use a stable list provider (here we use a maintained list page)
        resp = requests.get("https://dhan.co/nifty-stocks-list/nifty-50/", timeout=8)
        txt = resp.text
        # crude parse: look for uppercase symbols followed by .NS later when querying, but fallback to static list
    except Exception:
        txt = ""

    # Fallback static minimal list (safe if scraping fails). Tickers appended with .NS for Yahoo Finance.
    # Note: For production, replace with direct NSE indices API or a maintained CSV from NSE Indices.
    static = [
        "RELIANCE.NS","TCS.NS","HDFCBANK.NS","INFY.NS","HDFC.NS","ICICIBANK.NS","HINDUNILVR.NS",
        "SBIN.NS","KOTAKBANK.NS","LT.NS","ITC.NS","AXISBANK.NS","BHARTIARTL.NS","HCLTECH.NS",
        "BAJFINANCE.NS","MARUTI.NS","ULTRACEMCO.NS","SUNPHARMA.NS","WIPRO.NS","POWERGRID.NS",
        "NTPC.NS","ONGC.NS","TITAN.NS","NESTLEIND.NS","GRASIM.NS","DRREDDY.NS","BPCL.NS",
        "TATASTEEL.NS","BRITANNIA.NS","DIVISLAB.NS","ADANIENT.NS","EICHERMOT.NS","HINDALCO.NS",
        "COALINDIA.NS","INDUSINDBK.NS","JSWSTEEL.NS","BAJAJ-AUTO.NS","BHARATFORG.NS","TATAMOTORS.NS",
        "ADANIPORTS.NS","IOC.NS","HAVELLS.NS","CIPLA.NS","SHREECEM.NS","TECHM.NS","NTPC.NS"
    ]
    return static

def rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -1*delta.clip(upper=0)
    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()
    rs = avg_gain / avg_loss
    rr = 100 - (100 / (1 + rs))
    return rr

def fetch_quotes(tickers):
    # use yfinance bulk download
    data = yf.download(tickers, period="60d", interval="1d", threads=True, progress=False)
    # data returns a multi-index columns (Adj Close, Close, Volume etc.)
    return data

def compute_metrics(data, tickers):
    results = []
    for tk in tickers:
        try:
            close = data['Close'][tk].dropna()
            vol = data['Volume'][tk].dropna()
            if close.empty:
                continue

            cur_price = close.iloc[-1]
            prev_price = close.iloc[-2] if len(close) > 1 else cur_price
            pct_change = (cur_price - prev_price) / prev_price * 100 if prev_price != 0 else 0

            # Technical indicators
            sma20 = close.rolling(window=20).mean().iloc[-1] if len(close) >= 20 else np.nan
            sma50 = close.rolling(window=50).mean().iloc[-1] if len(close) >= 50 else np.nan
            ema20 = close.ewm(span=20, adjust=False).mean().iloc[-1]
            rsi14 = rsi(close, 14).iloc[-1] if len(close) >= 15 else np.nan

            vol_avg20 = vol.rolling(window=20).mean().iloc[-1] if len(vol) >= 20 else np.nan
            vol_latest = vol.iloc[-1] if not vol.empty else np.nan
            vol_spike = (vol_latest / vol_avg20) if (vol_avg20 and not np.isnan(vol_avg20)) else np.nan

            # Fundamentals
            t = yf.Ticker(tk)
            info = t.info
            week52_high = info.get("fiftyTwoWeekHigh")
            week52_low = info.get("fiftyTwoWeekLow")
            market_cap = info.get("marketCap")
            pe_ratio = info.get("trailingPE")
            div_yield = info.get("dividendYield")

            # -------------------------
            # Verdict Logic
            # -------------------------
            verdict = "HOLD"
            if not np.isnan(rsi14) and not np.isnan(sma20):
                if (cur_price > sma20) and (30 <= rsi14 <= 60) and (vol_spike and vol_spike > 1.5):
                    verdict = "BUY"
                elif (rsi14 > 70) or (cur_price < sma20) or (pct_change < -2):
                    verdict = "SELL"

            results.append({
                "ticker": tk,
                "price": round(float(cur_price), 2),
                "pct_change_1d": round(float(pct_change), 2),
                "sma20": round(float(sma20), 2) if not np.isnan(sma20) else np.nan,
                "sma50": round(float(sma50), 2) if not np.isnan(sma50) else np.nan,
                "ema20": round(float(ema20), 2),
                "rsi14": round(float(rsi14), 2) if not np.isnan(rsi14) else np.nan,
                "vol_latest": int(vol_latest) if not np.isnan(vol_latest) else np.nan,
                "vol_avg20": int(vol_avg20) if not np.isnan(vol_avg20) else np.nan,
                "vol_spike": round(float(vol_spike), 2) if not np.isnan(vol_spike) else np.nan,
                "52w_high": week52_high,
                "52w_low": week52_low,
                "market_cap": market_cap,
                "PE_ratio": pe_ratio,
                "div_yield": div_yield,
                "verdict": verdict
            })
        except Exception:
            continue

    return pd.DataFrame(results)


# UI
st.title("📈 NIFTY Live Stock Screener")
st.caption("Screener for NIFTY constituents — live-ish quotes via Yahoo Finance. Adjust filters and hit Run.")

# Controls
tickers = get_nifty50_list()
left, right = st.columns([1,3])
with left:
    st.subheader("Filters")
    pct_min = st.number_input("% change >= ", value=-5.0, step=0.1)
    pct_max = st.number_input("% change <= ", value=10.0, step=0.1)
    rsi_low = st.number_input("RSI < ", value=30.0, step=1.0)
    rsi_high = st.number_input("RSI > ", value=70.0, step=1.0)
    vol_spike_min = st.number_input("Volume spike >= (x)", value=1.5, step=0.1)
    sma_cross = st.checkbox("Show price > SMA20", value=False)
    refresh = st.button("Run Screener")

with right:
    st.subheader("Constituents to screen")
    st.write(f"Number of tickers: {len(tickers)}")
    # show subset
    st.dataframe(pd.DataFrame({"ticker": tickers}))

if refresh:
    with st.spinner("Fetching quotes (this may take 10-30s)..."):
        data = fetch_quotes(tickers)
        df = compute_metrics(data, tickers)
        # apply filters
        filtered = df[
            (df['pct_change_1d'] >= pct_min) &
            (df['pct_change_1d'] <= pct_max) &
            (df['vol_spike'].fillna(0) >= vol_spike_min)
        ]
        if sma_cross:
            filtered = filtered[filtered['price'] > filtered['sma20']]
        # RSI filter - show both extremes (below low OR above high)
        filtered = filtered[(filtered['rsi14'] <= rsi_low) | (filtered['rsi14'] >= rsi_high)]
        st.success(f"Found {len(filtered)} stocks matching filters")
        st.dataframe(filtered.sort_values(by="vol_spike", ascending=False).reset_index(drop=True))

        # simple chart for first ticker
        if not filtered.empty:
            first = filtered.iloc[0]['ticker']
            st.markdown(f"### Chart — {first}")
            # small price chart using data
            close = data['Close'][first].dropna()
            fig, ax = plt.subplots(figsize=(10,3))
            ax.plot(close.index, close.values)
            ax.set_title(first)
            st.pyplot(fig)

        # download
        csv = filtered.to_csv(index=False)
        st.download_button("Download CSV", csv, file_name=f"nifty_screener_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")

st.markdown("---")
st.caption("Notes: data fetched from Yahoo Finance (yfinance). For production-grade real-time data use a paid market-data API or NSE-provided feeds.")


