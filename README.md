# Text-Analytics

## Indian Stock Market Trading Research Agent

This repository now includes a Python-based **trading research agent** focused on Indian equities (NSE/BSE via Yahoo ticker feeds).

### Features
- Near-real-time OHLCV fetch for Indian symbols (example: `RELIANCE.NS`, `TCS.NS`)
- Technical analytics: RSI(14), EMA(20/50), MACD, Bollinger Bands, ATR(14), rolling volatility
- Rule-based signal engine (`BUY`, `SELL`, `HOLD`) with confidence and risk levels
- Latest market-news enrichment for detailed research context
- JSON report output for easy downstream integration

### Setup
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Usage
```bash
python indian_trading_agent.py RELIANCE.NS --interval 5m --period 5d
```

### Example Output Sections
- `market_snapshot`: live indicator values
- `signal`: side, confidence, thesis, entry/stop/targets
- `recent_news`: latest headlines with links and publishers

### Important
This tool is for **research and educational use only** and does **not** constitute financial advice.
