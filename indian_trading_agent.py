"""
Indian Stock Market Trading Research Agent

Features
- Pulls live/near-real-time quotes for NSE/BSE tickers via yfinance
- Computes detailed technical indicators (RSI, MACD, ATR, Bollinger Bands)
- Runs a market micro-brief with trend and volatility context
- Fetches latest company/news headlines for research context
- Produces rule-based trade ideas with risk-managed entry/exit levels

Notes
- This is a research assistant, not financial advice.
- For NSE tickers use Yahoo suffix '.NS' (e.g., RELIANCE.NS).
"""

from __future__ import annotations

import argparse
import dataclasses
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import requests
import yfinance as yf


@dataclass
class Signal:
    symbol: str
    side: str
    confidence: float
    thesis: str
    entry: float
    stop_loss: float
    target_1: float
    target_2: float


class IndianTradingResearchAgent:
    def __init__(self, risk_per_trade: float = 0.01):
        self.risk_per_trade = risk_per_trade

    def fetch_ohlcv(self, symbol: str, interval: str = "5m", period: str = "5d") -> pd.DataFrame:
        df = yf.download(symbol, interval=interval, period=period, auto_adjust=True, progress=False)
        if df.empty:
            raise ValueError(f"No market data found for {symbol}")
        df = df.dropna().copy()
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [c[0] for c in df.columns]
        return df

    @staticmethod
    def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
        delta = close.diff()
        gain = delta.clip(lower=0).rolling(period).mean()
        loss = (-delta.clip(upper=0)).rolling(period).mean()
        rs = gain / loss.replace(0, np.nan)
        return 100 - (100 / (1 + rs))

    @staticmethod
    def _ema(series: pd.Series, span: int) -> pd.Series:
        return series.ewm(span=span, adjust=False).mean()

    def enrich_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        out["rsi_14"] = self._rsi(out["Close"])
        out["ema_20"] = self._ema(out["Close"], 20)
        out["ema_50"] = self._ema(out["Close"], 50)

        macd_fast = self._ema(out["Close"], 12)
        macd_slow = self._ema(out["Close"], 26)
        out["macd"] = macd_fast - macd_slow
        out["macd_signal"] = self._ema(out["macd"], 9)
        out["macd_hist"] = out["macd"] - out["macd_signal"]

        rolling_std = out["Close"].rolling(20).std()
        out["bb_mid"] = out["Close"].rolling(20).mean()
        out["bb_upper"] = out["bb_mid"] + 2 * rolling_std
        out["bb_lower"] = out["bb_mid"] - 2 * rolling_std

        high_low = out["High"] - out["Low"]
        high_close = (out["High"] - out["Close"].shift()).abs()
        low_close = (out["Low"] - out["Close"].shift()).abs()
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        out["atr_14"] = tr.rolling(14).mean()

        out["returns"] = out["Close"].pct_change()
        out["volatility_20"] = out["returns"].rolling(20).std() * np.sqrt(252)

        return out.dropna()

    def latest_news(self, symbol: str, limit: int = 8) -> List[Dict[str, str]]:
        ticker = yf.Ticker(symbol)
        items = []
        for n in (ticker.news or [])[:limit]:
            title = n.get("title", "")
            publisher = n.get("publisher", "")
            link = n.get("link", "")
            ts = n.get("providerPublishTime")
            if ts:
                published = datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
            else:
                published = ""
            items.append(
                {"title": title, "publisher": publisher, "published_utc": published, "link": link}
            )
        return items

    def generate_signal(self, symbol: str, df: pd.DataFrame) -> Signal:
        row = df.iloc[-1]
        close = float(row["Close"])
        atr = float(row["atr_14"])
        rsi = float(row["rsi_14"])
        macd_hist = float(row["macd_hist"])

        trend_up = row["ema_20"] > row["ema_50"]
        trend_down = row["ema_20"] < row["ema_50"]

        side = "HOLD"
        confidence = 0.45
        thesis_parts: List[str] = []

        if trend_up and rsi > 52 and macd_hist > 0:
            side = "BUY"
            confidence = min(0.92, 0.55 + (rsi - 50) / 100 + min(macd_hist * 2, 0.2))
            thesis_parts.append("Bullish momentum with EMA20>EMA50 and positive MACD histogram")
        elif trend_down and rsi < 48 and macd_hist < 0:
            side = "SELL"
            confidence = min(0.92, 0.55 + (50 - rsi) / 100 + min(abs(macd_hist) * 2, 0.2))
            thesis_parts.append("Bearish momentum with EMA20<EMA50 and negative MACD histogram")
        else:
            thesis_parts.append("Mixed setup; wait for clearer trend confirmation")

        if rsi > 70:
            thesis_parts.append("RSI indicates overbought condition")
        elif rsi < 30:
            thesis_parts.append("RSI indicates oversold condition")

        if side == "BUY":
            entry = close
            stop = close - 1.2 * atr
            t1 = close + 1.8 * atr
            t2 = close + 3.0 * atr
        elif side == "SELL":
            entry = close
            stop = close + 1.2 * atr
            t1 = close - 1.8 * atr
            t2 = close - 3.0 * atr
        else:
            entry = close
            stop = close - 1.0 * atr
            t1 = close + 1.0 * atr
            t2 = close + 1.8 * atr

        return Signal(
            symbol=symbol,
            side=side,
            confidence=round(confidence, 3),
            thesis="; ".join(thesis_parts),
            entry=round(entry, 2),
            stop_loss=round(stop, 2),
            target_1=round(t1, 2),
            target_2=round(t2, 2),
        )

    def build_research_report(self, symbol: str, interval: str, period: str) -> Dict[str, object]:
        raw = self.fetch_ohlcv(symbol, interval=interval, period=period)
        enriched = self.enrich_indicators(raw)
        signal = dataclasses.asdict(self.generate_signal(symbol, enriched))
        news = self.latest_news(symbol)

        last = enriched.iloc[-1]
        snapshot = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "price": round(float(last["Close"]), 2),
            "volume": int(last["Volume"]),
            "rsi_14": round(float(last["rsi_14"]), 2),
            "ema_20": round(float(last["ema_20"]), 2),
            "ema_50": round(float(last["ema_50"]), 2),
            "macd": round(float(last["macd"]), 4),
            "macd_signal": round(float(last["macd_signal"]), 4),
            "macd_hist": round(float(last["macd_hist"]), 4),
            "atr_14": round(float(last["atr_14"]), 2),
            "bb_upper": round(float(last["bb_upper"]), 2),
            "bb_lower": round(float(last["bb_lower"]), 2),
            "annualized_vol_20": round(float(last["volatility_20"]), 4),
        }

        return {
            "symbol": symbol,
            "interval": interval,
            "period": period,
            "market_snapshot": snapshot,
            "signal": signal,
            "recent_news": news,
            "disclaimer": "For research only. Not investment advice.",
        }


def main() -> None:
    parser = argparse.ArgumentParser(description="Indian stock market research/trading agent")
    parser.add_argument("symbol", help="Ticker symbol, e.g., RELIANCE.NS or TCS.NS")
    parser.add_argument("--interval", default="5m", help="Data interval (e.g., 1m, 5m, 15m, 1h, 1d)")
    parser.add_argument("--period", default="5d", help="Data lookback period (e.g., 1d, 5d, 1mo)")
    args = parser.parse_args()

    agent = IndianTradingResearchAgent()
    report = agent.build_research_report(args.symbol, args.interval, args.period)

    import json

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
