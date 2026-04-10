"""
Stock Market Data Collector
============================
Fetches price history, technical indicators, fundamentals, and news
for a watchlist of tickers using free APIs.

Requirements:
    pip install yfinance pandas ta requests python-dotenv

Optional (for news sentiment):
    Get a free API key at https://www.alphavantage.co/support/#api-key
    Create a .env file with: ALPHA_VANTAGE_KEY=your_key_here

DISCLAIMER: This module is for educational/portfolio purposes only.
            It does not constitute financial advice.
"""

import os
import time
import logging
from datetime import datetime, timedelta
from typing import Optional

import pandas as pd
import yfinance as yf
import requests
from dotenv import load_dotenv

try:
    import ta
    TA_AVAILABLE = True
except ImportError:
    TA_AVAILABLE = False
    logging.warning("'ta' library not found. Install with: pip install ta")

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

ALPHA_VANTAGE_KEY = os.getenv("ALPHA_VANTAGE_KEY", "demo")
DEFAULT_WATCHLIST = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA"]
DEFAULT_PERIOD = "6mo"   # yfinance period string
DEFAULT_INTERVAL = "1d"  # daily bars


# ─────────────────────────────────────────────
# 1. Price & Volume History
# ─────────────────────────────────────────────

def fetch_price_history(
    ticker: str,
    period: str = DEFAULT_PERIOD,
    interval: str = DEFAULT_INTERVAL,
) -> Optional[pd.DataFrame]:
    """
    Download OHLCV history for a single ticker via yfinance (no API key needed).

    Returns a DataFrame with columns:
        Open, High, Low, Close, Volume, Ticker
    Returns None on failure.
    """
    try:
        df = yf.download(ticker, period=period, interval=interval, progress=False, auto_adjust=True)
        if df.empty:
            logger.warning(f"No price data returned for {ticker}")
            return None
        df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
        df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
        df["Ticker"] = ticker
        df.index.name = "Date"
        logger.info(f"Fetched {len(df)} rows of price history for {ticker}")
        return df
    except Exception as e:
        logger.error(f"Price fetch failed for {ticker}: {e}")
        return None


# ─────────────────────────────────────────────
# 2. Technical Indicators
# ─────────────────────────────────────────────

def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Appends common technical indicators to a price DataFrame.

    Indicators added (requires the 'ta' library):
        - RSI (14)          – momentum oscillator, 0-100
        - MACD / Signal     – trend-following momentum
        - Bollinger Bands   – volatility bands around 20-day SMA
        - SMA 20 / SMA 50   – simple moving averages

    Falls back to manual pandas calculations if 'ta' is unavailable.
    """
    if df is None or df.empty:
        return df

    close = df["Close"]
    high  = df["High"]
    low   = df["Low"]

    if TA_AVAILABLE:
        df["RSI"]         = ta.momentum.RSIIndicator(close, window=14).rsi()
        macd_ind          = ta.trend.MACD(close)
        df["MACD"]        = macd_ind.macd()
        df["MACD_Signal"] = macd_ind.macd_signal()
        bb                = ta.volatility.BollingerBands(close, window=20, window_dev=2)
        df["BB_Upper"]    = bb.bollinger_hband()
        df["BB_Middle"]   = bb.bollinger_mavg()
        df["BB_Lower"]    = bb.bollinger_lband()
    else:
        # Manual fallback
        delta = close.diff()
        gain  = delta.clip(lower=0).rolling(14).mean()
        loss  = (-delta.clip(upper=0)).rolling(14).mean()
        rs    = gain / loss.replace(0, float("nan"))
        df["RSI"] = 100 - (100 / (1 + rs))

        ema12 = close.ewm(span=12, adjust=False).mean()
        ema26 = close.ewm(span=26, adjust=False).mean()
        df["MACD"]        = ema12 - ema26
        df["MACD_Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()

        sma20           = close.rolling(20).mean()
        std20           = close.rolling(20).std()
        df["BB_Upper"]  = sma20 + 2 * std20
        df["BB_Middle"] = sma20
        df["BB_Lower"]  = sma20 - 2 * std20

    df["SMA_20"] = close.rolling(20).mean()
    df["SMA_50"] = close.rolling(50).mean()

    logger.info(f"Technical indicators added for {df['Ticker'].iloc[0]}")
    return df


# ─────────────────────────────────────────────
# 3. Fundamental Data
# ─────────────────────────────────────────────

def fetch_fundamentals(ticker: str) -> dict:
    """
    Pulls key fundamental metrics via yfinance (no API key needed).

    Returns a dict with:
        pe_ratio, forward_pe, eps, revenue_growth, profit_margin,
        debt_to_equity, market_cap, dividend_yield, beta, sector, industry
    """
    fundamentals = {"ticker": ticker}
    try:
        info = yf.Ticker(ticker).info
        fields = {
            "pe_ratio":       "trailingPE",
            "forward_pe":     "forwardPE",
            "eps":            "trailingEps",
            "revenue_growth": "revenueGrowth",
            "profit_margin":  "profitMargins",
            "debt_to_equity": "debtToEquity",
            "market_cap":     "marketCap",
            "dividend_yield": "dividendYield",
            "beta":           "beta",
            "sector":         "sector",
            "industry":       "industry",
            "company_name":   "longName",
        }
        for key, yf_key in fields.items():
            fundamentals[key] = info.get(yf_key)

        logger.info(f"Fetched fundamentals for {ticker}")
    except Exception as e:
        logger.error(f"Fundamentals fetch failed for {ticker}: {e}")

    return fundamentals


# ─────────────────────────────────────────────
# 4. News Headlines (Alpha Vantage)
# ─────────────────────────────────────────────

def fetch_news(ticker: str, limit: int = 10) -> list[dict]:
    """
    Fetches recent news headlines and sentiment scores for a ticker
    via Alpha Vantage News Sentiment API (free tier: 25 req/day).

    Returns a list of dicts:
        { title, source, published, url, overall_sentiment_label,
          overall_sentiment_score, relevance_score }

    Falls back to an empty list if the key is missing or rate-limited.
    """
    if ALPHA_VANTAGE_KEY == "demo":
        logger.warning("No ALPHA_VANTAGE_KEY set — skipping news fetch. Add key to .env file.")
        return []

    url = (
        "https://www.alphavantage.co/query"
        f"?function=NEWS_SENTIMENT&tickers={ticker}"
        f"&limit={limit}&apikey={ALPHA_VANTAGE_KEY}"
    )
    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        data = resp.json()

        articles = []
        for item in data.get("feed", []):
            # Find sentiment score specific to this ticker
            ticker_sentiment = next(
                (t for t in item.get("ticker_sentiment", []) if t["ticker"] == ticker),
                {}
            )
            articles.append({
                "title":                    item.get("title"),
                "source":                   item.get("source"),
                "published":                item.get("time_published"),
                "url":                      item.get("url"),
                "overall_sentiment_label":  item.get("overall_sentiment_label"),
                "overall_sentiment_score":  float(item.get("overall_sentiment_score", 0)),
                "relevance_score":          float(ticker_sentiment.get("relevance_score", 0)),
                "ticker_sentiment_label":   ticker_sentiment.get("ticker_sentiment_label"),
            })

        logger.info(f"Fetched {len(articles)} news items for {ticker}")
        return articles

    except Exception as e:
        logger.error(f"News fetch failed for {ticker}: {e}")
        return []


# ─────────────────────────────────────────────
# 5. Master Collector — single ticker
# ─────────────────────────────────────────────

def collect_ticker_data(
    ticker: str,
    period: str = DEFAULT_PERIOD,
    include_news: bool = True,
) -> dict:
    """
    Collects all data for a single ticker and returns a unified dict:
        {
            "ticker":        str,
            "collected_at":  ISO timestamp,
            "price_history": DataFrame (with indicators),
            "latest_price":  dict of the most recent bar,
            "fundamentals":  dict,
            "news":          list[dict],
            "summary":       dict   ← key metrics ready to feed to Claude
        }
    """
    logger.info(f"Collecting data for {ticker}...")

    price_df = fetch_price_history(ticker, period=period)
    if price_df is not None:
        price_df = add_technical_indicators(price_df)

    fundamentals = fetch_fundamentals(ticker)
    news = fetch_news(ticker) if include_news else []

    # Build a compact summary dict for easy LLM consumption
    summary = {"ticker": ticker}
    if price_df is not None and not price_df.empty:
        latest = price_df.iloc[-1]
        prev   = price_df.iloc[-2] if len(price_df) > 1 else latest
        summary.update({
            "current_price":   round(float(latest["Close"]), 2),
            "price_change_1d": round(float(latest["Close"] - prev["Close"]), 2),
            "pct_change_1d":   round(float((latest["Close"] - prev["Close"]) / prev["Close"] * 100), 2),
            "volume":          int(latest["Volume"]),
            "rsi_14":          round(float(latest["RSI"]), 1) if pd.notna(latest.get("RSI")) else None,
            "macd":            round(float(latest["MACD"]), 3) if pd.notna(latest.get("MACD")) else None,
            "macd_signal":     round(float(latest["MACD_Signal"]), 3) if pd.notna(latest.get("MACD_Signal")) else None,
            "above_sma20":     bool(latest["Close"] > latest["SMA_20"]) if pd.notna(latest.get("SMA_20")) else None,
            "above_sma50":     bool(latest["Close"] > latest["SMA_50"]) if pd.notna(latest.get("SMA_50")) else None,
            "bb_position":     _bb_position(latest),
        })
        latest_bar = price_df.iloc[-1].to_dict()
    else:
        latest_bar = {}

    summary.update({
        "pe_ratio":       fundamentals.get("pe_ratio"),
        "forward_pe":     fundamentals.get("forward_pe"),
        "revenue_growth": fundamentals.get("revenue_growth"),
        "profit_margin":  fundamentals.get("profit_margin"),
        "beta":           fundamentals.get("beta"),
        "sector":         fundamentals.get("sector"),
    })

    if news:
        avg_sentiment = sum(n["overall_sentiment_score"] for n in news) / len(news)
        summary["news_sentiment_avg"] = round(avg_sentiment, 3)
        summary["recent_headlines"]   = [n["title"] for n in news[:3]]

    return {
        "ticker":        ticker,
        "collected_at":  datetime.utcnow().isoformat(),
        "price_history": price_df,
        "latest_price":  latest_bar,
        "fundamentals":  fundamentals,
        "news":          news,
        "summary":       summary,
    }


def _bb_position(row: pd.Series) -> Optional[str]:
    """Returns 'above_upper', 'below_lower', or 'within_bands'."""
    try:
        close = float(row["Close"])
        upper = float(row["BB_Upper"])
        lower = float(row["BB_Lower"])
        if pd.isna(upper) or pd.isna(lower):
            return None
        if close > upper:
            return "above_upper"
        if close < lower:
            return "below_lower"
        return "within_bands"
    except Exception:
        return None


# ─────────────────────────────────────────────
# 6. Watchlist Collector
# ─────────────────────────────────────────────

def collect_watchlist(
    tickers: list[str] = DEFAULT_WATCHLIST,
    period: str = DEFAULT_PERIOD,
    include_news: bool = True,
    delay_seconds: float = 1.0,
) -> dict[str, dict]:
    """
    Collects data for every ticker in the watchlist.
    Adds a small delay between requests to respect rate limits.

    Returns { ticker: data_dict, ... }
    """
    results = {}
    for ticker in tickers:
        results[ticker] = collect_ticker_data(ticker, period=period, include_news=include_news)
        if delay_seconds > 0:
            time.sleep(delay_seconds)
    logger.info(f"Watchlist collection complete. {len(results)} tickers processed.")
    return results


def summaries_for_claude(watchlist_data: dict[str, dict]) -> list[dict]:
    """
    Extracts the compact summary dicts — ready to drop into a Claude prompt.
    """
    return [data["summary"] for data in watchlist_data.values()]


# ─────────────────────────────────────────────
# 7. Quick demo
# ─────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("Stock Data Collector — Demo Run")
    print("DISCLAIMER: For educational purposes only. Not financial advice.")
    print("=" * 60)

    # Collect a small watchlist (no API key needed for price + fundamentals)
    watchlist = ["AAPL", "MSFT", "NVDA"]
    data = collect_watchlist(watchlist, include_news=False)

    # Print summaries
    summaries = summaries_for_claude(data)
    for s in summaries:
        print(f"\n── {s['ticker']} ──────────────────────────")
        for k, v in s.items():
            if k != "recent_headlines":
                print(f"  {k:<22} {v}")

    print("\n✓ Data ready to pass to Claude for analysis.")
    print("  Next step: feed `summaries` into your analysis agent.")
