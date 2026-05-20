import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from datetime import datetime

st.set_page_config(page_title="AI Stock Market Monitor", page_icon="📈", layout="wide")

DEFAULT_TICKERS = [
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA",
    "AMD", "AVGO", "ORCL", "CRM", "NFLX", "PLTR", "SMCI",
    "JPM", "BAC", "GS", "BRK-B", "V", "MA",
    "UNH", "LLY", "JNJ", "PFE", "COST", "WMT", "HD", "MCD", "NKE",
    "XOM", "CVX", "BA", "CAT", "GE", "SPY", "QQQ"
]

SECTOR_MAP = {
    "AAPL": "Technology", "MSFT": "Technology", "NVDA": "Semiconductors",
    "AMZN": "Consumer / Cloud", "GOOGL": "Communication / AI", "META": "Communication / Ads",
    "TSLA": "EV / AI", "AMD": "Semiconductors", "AVGO": "Semiconductors",
    "ORCL": "Cloud / Software", "CRM": "Software", "NFLX": "Streaming",
    "PLTR": "AI / Data", "SMCI": "AI Infrastructure", "JPM": "Banking",
    "BAC": "Banking", "GS": "Investment Banking", "BRK-B": "Conglomerate",
    "V": "Payments", "MA": "Payments", "UNH": "Healthcare", "LLY": "Healthcare",
    "JNJ": "Healthcare", "PFE": "Healthcare", "COST": "Retail", "WMT": "Retail",
    "HD": "Retail / Housing", "MCD": "Consumer Defensive", "NKE": "Consumer Discretionary",
    "XOM": "Energy", "CVX": "Energy", "BA": "Aerospace", "CAT": "Industrials",
    "GE": "Industrials", "SPY": "ETF - S&P 500", "QQQ": "ETF - Nasdaq 100"
}

st.sidebar.title("⚙️ Dashboard Controls")
watchlist_input = st.sidebar.text_area("Stock watchlist", value=", ".join(DEFAULT_TICKERS), height=160)
period = st.sidebar.selectbox("Historical period", ["1mo", "3mo", "6mo", "1y", "2y", "5y"], index=3)
st.sidebar.caption("Auto-refresh logic: data cache refreshes every 15 minutes. Refresh browser to force update.")
selected_tickers = [t.strip().upper() for t in watchlist_input.split(",") if t.strip()]

@st.cache_data(ttl=900)
def get_stock_history(ticker, period="1y"):
    try:
        data = yf.Ticker(ticker).history(period=period)
        return data if data is not None and not data.empty else pd.DataFrame()
    except Exception:
        return pd.DataFrame()

@st.cache_data(ttl=900)
def get_stock_info(ticker):
    try:
        info = yf.Ticker(ticker).info
        return info if isinstance(info, dict) else {}
    except Exception:
        return {}

def calculate_rsi(close, window=14):
    delta = close.diff()
    gain = delta.where(delta > 0, 0).rolling(window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def safe_pct(current, old):
    try:
        if old == 0 or pd.isna(old):
            return np.nan
        return ((current - old) / old) * 100
    except Exception:
        return np.nan

def score_stock(row):
    score = 50
    if pd.notna(row["Return_1M_%"]):
        score += 10 if row["Return_1M_%"] > 5 else (-10 if row["Return_1M_%"] < -5 else 0)
    if pd.notna(row["Return_6M_%"]):
        score += 12 if row["Return_6M_%"] > 10 else (-12 if row["Return_6M_%"] < -10 else 0)
    score += 8 if pd.notna(row["MA50"]) and row["Price"] > row["MA50"] else -5
    score += 10 if pd.notna(row["MA200"]) and row["Price"] > row["MA200"] else -8
    if pd.notna(row["RSI"]):
        if 40 <= row["RSI"] <= 65: score += 8
        elif row["RSI"] > 75: score -= 10
        elif row["RSI"] < 30: score -= 5
    if pd.notna(row["Volatility_%"]):
        if row["Volatility_%"] < 35: score += 5
        elif row["Volatility_%"] > 60: score -= 8
    if pd.notna(row["PE"]):
        if 5 <= row["PE"] <= 35: score += 7
        elif row["PE"] > 80: score -= 10
    return max(0, min(100, round(score, 1)))

def signal_from_score(score):
    if score >= 80: return "Strong Watch / Bullish"
    if score >= 65: return "Positive Watch"
    if score >= 50: return "Neutral / Monitor"
    if score >= 35: return "Caution"
    return "High Risk / Avoid"

def generate_ai_commentary(row):
    text = f"{row['Ticker']}: {row['Signal']}. "
    if pd.notna(row["MA50"]) and pd.notna(row["MA200"]):
        if row["Price"] > row["MA50"] and row["Price"] > row["MA200"]:
            text += "Technically constructive because price is above both 50-day and 200-day moving averages. "
        elif row["Price"] < row["MA50"] and row["Price"] < row["MA200"]:
            text += "Technically weak because price is below major moving averages. "
        else:
            text += "Trend is mixed and needs confirmation. "
    if pd.notna(row["RSI"]):
        if row["RSI"] > 75: text += "RSI indicates overbought risk. "
        elif row["RSI"] < 30: text += "RSI indicates oversold pressure. "
        else: text += "RSI is in a manageable zone. "
    if pd.notna(row["PE"]):
        if row["PE"] > 80: text += "Valuation appears stretched based on P/E. "
        elif 5 <= row["PE"] <= 35: text += "Valuation is relatively reasonable versus high-growth peers. "
    text += "Volatility is high, so position sizing should be conservative." if pd.notna(row["Volatility_%"]) and row["Volatility_%"] > 60 else "Volatility is acceptable for active monitoring."
    return text

st.title("📈 AI-Monitored US Stock Market Dashboard")
st.caption("Shareable US stock dashboard with AI scoring, technical signals, sector view, and commentary.")
st.info(f"Last refreshed: {datetime.now().strftime('%d-%b-%Y %I:%M %p')} | Data source: Yahoo Finance via yfinance | Cache refresh: 15 minutes")

rows = []
with st.spinner("Fetching market data..."):
    for ticker in selected_tickers:
        hist = get_stock_history(ticker, period)
        info = get_stock_info(ticker)
        if hist.empty or len(hist) < 20:
            continue
        close = hist["Close"]
        price = close.iloc[-1]
        previous = close.iloc[-2] if len(close) >= 2 else np.nan
        price_1m = close.iloc[-22] if len(close) >= 22 else close.iloc[0]
        price_3m = close.iloc[-66] if len(close) >= 66 else close.iloc[0]
        price_6m = close.iloc[-132] if len(close) >= 132 else close.iloc[0]
        price_start = close.iloc[0]
        ma50 = close.rolling(50).mean().iloc[-1]
        ma200 = close.rolling(200).mean().iloc[-1] if len(close) >= 200 else close.rolling(len(close)).mean().iloc[-1]
        rsi = calculate_rsi(close).iloc[-1]
        volatility = close.pct_change().std() * np.sqrt(252) * 100
        market_cap = info.get("marketCap", np.nan)
        pe = info.get("trailingPE", np.nan)
        rows.append({
            "Ticker": ticker,
            "Company": info.get("shortName", ticker),
            "Sector": SECTOR_MAP.get(ticker, info.get("sector", "N/A")),
            "Price": round(price, 2),
            "Day_%": round(safe_pct(price, previous), 2) if pd.notna(previous) else np.nan,
            "Return_1M_%": round(safe_pct(price, price_1m), 2),
            "Return_3M_%": round(safe_pct(price, price_3m), 2),
            "Return_6M_%": round(safe_pct(price, price_6m), 2),
            "Return_Period_%": round(safe_pct(price, price_start), 2),
            "MA50": round(ma50, 2) if pd.notna(ma50) else np.nan,
            "MA200": round(ma200, 2) if pd.notna(ma200) else np.nan,
            "RSI": round(rsi, 2) if pd.notna(rsi) else np.nan,
            "Volatility_%": round(volatility, 2) if pd.notna(volatility) else np.nan,
            "PE": round(pe, 2) if isinstance(pe, (int, float)) and not pd.isna(pe) else np.nan,
            "Market_Cap_Bn": round(market_cap / 1e9, 2) if isinstance(market_cap, (int, float)) else np.nan
        })

if not rows:
    st.error("No market data found. Check ticker symbols or try later.")
    st.stop()

df = pd.DataFrame(rows)
df["Score"] = df.apply(score_stock, axis=1)
df["Signal"] = df["Score"].apply(signal_from_score)
df["AI_Commentary"] = df.apply(generate_ai_commentary, axis=1)
df = df.sort_values("Score", ascending=False)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Stocks monitored", len(df))
c2.metric("Top score", df["Score"].max())
c3.metric("Avg day return", f"{df['Day_%'].mean():.2f}%")
c4.metric("High-risk names", df[df["Score"] < 40].shape[0])

tab1, tab2, tab3, tab4, tab5 = st.tabs(["Ranking", "Charts", "Sector View", "AI Commentary", "Export"])

with tab1:
    st.subheader("AI Stock Ranking")
    display_cols = ["Ticker", "Company", "Sector", "Price", "Day_%", "Return_1M_%", "Return_3M_%", "Return_6M_%", "RSI", "PE", "Volatility_%", "Market_Cap_Bn", "Score", "Signal"]
    st.dataframe(df[display_cols], use_container_width=True, hide_index=True)
    st.subheader("Top Opportunities")
    st.dataframe(df.head(10)[["Ticker", "Sector", "Score", "Signal", "AI_Commentary"]], use_container_width=True, hide_index=True)
    st.subheader("Caution / Avoid Monitor")
    st.dataframe(df.tail(10)[["Ticker", "Sector", "Score", "Signal", "AI_Commentary"]], use_container_width=True, hide_index=True)

with tab2:
    selected_chart_ticker = st.selectbox("Select ticker for chart", df["Ticker"].tolist())
    chart_hist = get_stock_history(selected_chart_ticker, period)
    if not chart_hist.empty:
        chart_hist["MA50"] = chart_hist["Close"].rolling(50).mean()
        chart_hist["MA200"] = chart_hist["Close"].rolling(200).mean()
        fig = go.Figure()
        fig.add_trace(go.Candlestick(x=chart_hist.index, open=chart_hist["Open"], high=chart_hist["High"], low=chart_hist["Low"], close=chart_hist["Close"], name="Price"))
        fig.add_trace(go.Scatter(x=chart_hist.index, y=chart_hist["MA50"], name="MA50"))
        fig.add_trace(go.Scatter(x=chart_hist.index, y=chart_hist["MA200"], name="MA200"))
        fig.update_layout(height=600, xaxis_rangeslider_visible=False)
        st.plotly_chart(fig, use_container_width=True)

with tab3:
    st.subheader("Sector Rotation Snapshot")
    sector_df = df.groupby("Sector", as_index=False).agg(Avg_Score=("Score", "mean"), Avg_Day_Return=("Day_%", "mean"), Avg_1M_Return=("Return_1M_%", "mean"), Count=("Ticker", "count")).sort_values("Avg_Score", ascending=False)
    st.dataframe(sector_df, use_container_width=True, hide_index=True)
    fig_sector = go.Figure()
    fig_sector.add_trace(go.Bar(x=sector_df["Sector"], y=sector_df["Avg_Score"], name="Avg Score"))
    fig_sector.update_layout(height=500, title="Average AI Score by Sector")
    st.plotly_chart(fig_sector, use_container_width=True)

with tab4:
    st.subheader("AI Research Commentary")
    for _, row in df.iterrows():
        with st.expander(f"{row['Ticker']} | Score {row['Score']} | {row['Signal']}"):
            st.write(row["AI_Commentary"])
            st.markdown("**Research checklist:**")
            st.write("- Check latest earnings and guidance")
            st.write("- Compare valuation against sector peers")
            st.write("- Monitor price vs MA50 and MA200")
            st.write("- Watch news catalyst and analyst revisions")

with tab5:
    st.subheader("Export Data")
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("Download Dashboard Data as CSV", data=csv, file_name="ai_stock_dashboard.csv", mime="text/csv")

st.divider()
st.caption("Disclaimer: This dashboard is for research and education only. It is not financial advice or a buy/sell recommendation.")
