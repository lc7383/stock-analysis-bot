"""
Stock Screener Module
======================
Scans a universe of stocks and surfaces the ones with the
strongest BUY signals based on technical indicators.

Instead of entering a ticker manually, the screener automatically
filters hundreds of stocks down to the best candidates.

Universes available:
    - S&P 500 (top 500 US companies)
    - NASDAQ 100 (top 100 tech-heavy)
    - Dow Jones 30 (30 blue chip stocks)
    - Custom list

Screening criteria (configurable):
    - RSI below threshold (oversold = potential bounce)
    - Price above SMA20 (uptrend confirmed)
    - MACD bullish crossover
    - Volume above average (conviction)
    - Minimum price and market activity filters

Requirements:
    pip install yfinance pandas ta requests

DISCLAIMER: For educational/portfolio purposes only.
            Screening results are not financial advice.
            Always do your own research before investing.
"""

import time
import logging
import pandas as pd
import yfinance as yf
from datetime import datetime

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# 1. Stock Universes
# ─────────────────────────────────────────────

SP500_TICKERS = [
    "AAPL","MSFT","NVDA","AMZN","GOOGL","META","TSLA","BRK-B","LLY","V",
    "JPM","UNH","XOM","MA","JNJ","PG","HD","COST","MRK","ABBV",
    "CRM","BAC","CVX","NFLX","PEP","KO","TMO","ADBE","WMT","ACN",
    "MCD","CSCO","ABT","AMD","ORCL","GE","NOW","PM","TXN","CAT",
    "IBM","DHR","NEE","ISRG","QCOM","INTU","AMGN","RTX","SPGI","PFE",
    "HON","UBER","LOW","BKNG","GS","T","SBUX","AXP","DE","AMAT",
    "GILD","MDT","SYK","ELV","MMC","BMY","ADI","VRTX","REGN","PLD",
    "LRCX","MU","C","PANW","ETN","BSX","CB","ZTS","FI","KLAC",
    "SO","DUK","CL","MSI","AON","ITW","CME","WM","PH","GD",
    "NOC","USB","TJX","EMR","HCA","FCX","NSC","APH","MCK","SHW",
]

NASDAQ100_TICKERS = [
    "AAPL","MSFT","NVDA","AMZN","GOOGL","META","TSLA","AVGO","COST","NFLX",
    "AMD","ADBE","QCOM","CSCO","INTC","INTU","TXN","AMAT","ISRG","MU",
    "LRCX","KLAC","PANW","SNPS","CDNS","MRVL","ORLY","REGN","ADI","FTNT",
    "MELI","ASML","ABNB","KDP","PCAR","TEAM","CRWD","DXCM","WDAY","CPRT",
    "NXPI","MNST","ODFL","FAST","ROST","PAYX","VRSK","BIIB","IDXX","EXC",
    "XEL","FANG","ZS","ANSS","ILMN","ON","GEHC","GFS","CEG","AEP",
]

DOW30_TICKERS = [
    "AAPL","AMGN","AXP","BA","CAT","CRM","CSCO","CVX","DIS","DOW",
    "GS","HD","HON","IBM","INTC","JNJ","JPM","KO","MCD","MMM",
    "MRK","MSFT","NKE","PG","TRV","UNH","V","VZ","WBA","WMT",
]

UNIVERSES = {
    "S&P 500 (100 stocks)":  SP500_TICKERS,
    "NASDAQ 100":             NASDAQ100_TICKERS,
    "Dow Jones 30":           DOW30_TICKERS,
}


# ─────────────────────────────────────────────
# 2. Screen Single Ticker
# ─────────────────────────────────────────────

def screen_ticker(ticker: str, criteria: dict) -> dict | None:
    """
    Screens a single ticker against the given criteria.
    Returns a result dict if the stock passes, None if it fails.

    Criteria keys (all optional):
        max_rsi          — RSI must be below this (default 45)
        min_rsi          — RSI must be above this (default 20)
        require_above_sma20  — price must be above SMA20
        require_macd_bullish — MACD must be above signal line
        min_volume_ratio — volume must be above X times 20-day avg
        min_price        — minimum stock price filter
        max_price        — maximum stock price filter
    """
    try:
        df = yf.download(ticker, period="3mo", interval="1d", progress=False, auto_adjust=True)
        if df.empty or len(df) < 30:
            return None

        df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
        df["Ticker"] = ticker

        from data_collector import add_technical_indicators
        df = add_technical_indicators(df)

        latest = df.iloc[-1]
        close  = float(latest["Close"])
        volume = float(latest["Volume"])

        # Price filters
        min_price = criteria.get("min_price", 5.0)
        max_price = criteria.get("max_price", 99999)
        if close < min_price or close > max_price:
            return None

        # RSI filter
        rsi = latest.get("RSI")
        if pd.isna(rsi):
            return None
        rsi = float(rsi)
        if rsi > criteria.get("max_rsi", 45):
            return None
        if rsi < criteria.get("min_rsi", 20):
            return None

        # SMA20 filter
        sma20 = latest.get("SMA_20")
        if criteria.get("require_above_sma20", True) and not pd.isna(sma20):
            if close < float(sma20):
                return None

        # MACD filter
        macd       = latest.get("MACD")
        macd_signal = latest.get("MACD_Signal")
        if criteria.get("require_macd_bullish", False):
            if pd.isna(macd) or pd.isna(macd_signal):
                return None
            if float(macd) < float(macd_signal):
                return None

        # Volume filter
        vol_ma20 = df["Volume"].rolling(20).mean().iloc[-1]
        vol_ratio = volume / float(vol_ma20) if vol_ma20 > 0 else 1.0
        min_vol_ratio = criteria.get("min_volume_ratio", 0.5)
        if vol_ratio < min_vol_ratio:
            return None

        # Calculate score (higher = stronger signal)
        score = 0
        score += max(0, (45 - rsi))           # lower RSI = higher score
        if not pd.isna(macd) and not pd.isna(macd_signal):
            if float(macd) > float(macd_signal):
                score += 10                   # bullish MACD crossover
        if vol_ratio > 1.5:
            score += 5                        # above average volume
        sma50 = latest.get("SMA_50")
        if not pd.isna(sma50) and close > float(sma50):
            score += 5                        # above SMA50

        # Bollinger band position
        bb_lower = latest.get("BB_Lower")
        bb_upper = latest.get("BB_Upper")
        if not pd.isna(bb_lower) and not pd.isna(bb_upper):
            bb_range = float(bb_upper) - float(bb_lower)
            if bb_range > 0:
                bb_pos = (close - float(bb_lower)) / bb_range
                if bb_pos < 0.3:
                    score += 8               # near lower band = oversold

        # 1-day and 5-day returns
        prev_close    = float(df["Close"].iloc[-2]) if len(df) > 1 else close
        return_1d     = round((close - prev_close) / prev_close * 100, 2)
        prev_5d_close = float(df["Close"].iloc[-6]) if len(df) > 5 else close
        return_5d     = round((close - prev_5d_close) / prev_5d_close * 100, 2)

        return {
            "ticker":       ticker,
            "price":        round(close, 2),
            "rsi":          round(rsi, 1),
            "macd_bullish": not pd.isna(macd) and not pd.isna(macd_signal) and float(macd) > float(macd_signal),
            "above_sma20":  not pd.isna(sma20) and close > float(sma20),
            "above_sma50":  not pd.isna(sma50) and close > float(sma50),
            "volume_ratio": round(vol_ratio, 2),
            "return_1d":    return_1d,
            "return_5d":    return_5d,
            "score":        round(score, 1),
        }

    except Exception as e:
        logger.debug(f"Screen failed for {ticker}: {e}")
        return None


# ─────────────────────────────────────────────
# 3. Run Full Screener
# ─────────────────────────────────────────────

def run_screener(
    universe: str = "Dow Jones 30",
    criteria: dict = None,
    max_results: int = 20,
    progress_callback=None,
) -> dict:
    """
    Scans a universe of stocks and returns the top candidates.

    Args:
        universe:          one of the UNIVERSES keys
        criteria:          screening criteria dict (see screen_ticker)
        max_results:       maximum number of results to return
        progress_callback: optional function(current, total, ticker) for progress updates

    Returns a dict with:
        results     — list of passing stocks sorted by score
        screened    — total stocks scanned
        passed      — number that passed filters
        failed      — number filtered out
        duration    — time taken in seconds
        criteria    — criteria used
        timestamp   — when the screen was run
    """
    if criteria is None:
        criteria = {
            "max_rsi":              45,
            "min_rsi":              20,
            "require_above_sma20":  True,
            "require_macd_bullish": False,
            "min_volume_ratio":     0.5,
            "min_price":            5.0,
        }

    tickers   = UNIVERSES.get(universe, DOW30_TICKERS)
    results   = []
    start     = datetime.now()

    logger.info(f"Screening {len(tickers)} stocks in {universe}...")

    for i, ticker in enumerate(tickers):
        if progress_callback:
            progress_callback(i + 1, len(tickers), ticker)

        result = screen_ticker(ticker, criteria)
        if result:
            results.append(result)

        time.sleep(0.15)  # rate limit yfinance

    # Sort by score descending
    results.sort(key=lambda x: x["score"], reverse=True)
    top_results = results[:max_results]

    duration = round((datetime.now() - start).total_seconds(), 1)
    logger.info(f"Screener complete — {len(results)}/{len(tickers)} passed in {duration}s")

    return {
        "results":   top_results,
        "screened":  len(tickers),
        "passed":    len(results),
        "failed":    len(tickers) - len(results),
        "duration":  duration,
        "criteria":  criteria,
        "universe":  universe,
        "timestamp": datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC"),
    }


# ─────────────────────────────────────────────
# 4. Demo
# ─────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("Stock Screener — Demo (Dow Jones 30)")
    print("DISCLAIMER: Educational use only. Not financial advice.")
    print("=" * 60)

    screen_results = run_screener(
        universe="Dow Jones 30",
        criteria={
            "max_rsi":             45,
            "min_rsi":             20,
            "require_above_sma20": True,
            "min_price":           5.0,
        },
        max_results=10,
    )

    results = screen_results["results"]
    print(f"\nScanned {screen_results['screened']} stocks — {screen_results['passed']} passed filters\n")

    if results:
        print(f"{'Ticker':<8} {'Price':>8} {'RSI':>6} {'Score':>7} {'1D%':>7} {'5D%':>7} {'Vol Ratio':>10}")
        print("-" * 60)
        for r in results:
            print(f"{r['ticker']:<8} ${r['price']:>7.2f} {r['rsi']:>6.1f} {r['score']:>7.1f} "
                  f"{r['return_1d']:>+6.2f}% {r['return_5d']:>+6.2f}% {r['volume_ratio']:>9.2f}x")
    else:
        print("No stocks passed the screening criteria.")
