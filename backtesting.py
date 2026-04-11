"""
Backtesting Module
===================
Tests how well the Claude signal strategy would have performed
historically by simulating trades based on Buy/Hold/Sell signals.

Strategy:
    - Start with $10,000 virtual cash per stock
    - BUY signal  → buy as many shares as possible
    - SELL signal → sell all shares held
    - HOLD signal → do nothing
    - Compare final portfolio value vs simply buying and holding

This is a simplified backtest for educational purposes.
Real backtesting requires accounting for slippage, commissions,
taxes, and other real-world factors.

Requirements:
    pip install yfinance pandas numpy matplotlib

DISCLAIMER: For educational/portfolio purposes only.
            Past performance does not guarantee future results.
            This is not financial advice.
"""

import logging
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from typing import Optional

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

STARTING_CASH    = 10_000.0   # Virtual cash per stock
COMMISSION       = 0.0        # Set to e.g. 5.0 for $5 per trade


# ─────────────────────────────────────────────
# 1. Generate Historical Signals
# ─────────────────────────────────────────────

def generate_signals(df: pd.DataFrame) -> pd.Series:
    """
    Generates BUY/HOLD/SELL signals from technical indicators.
    This replicates the logic Claude uses but in a rule-based way
    so we can apply it across historical data.

    Signal rules:
        BUY  — RSI < 40 AND price crosses above SMA20 AND MACD > Signal
        SELL — RSI > 70 OR price crosses below SMA20 AND MACD < Signal
        HOLD — everything else

    Returns a Series of "BUY", "HOLD", or "SELL" for each date.
    """
    signals = pd.Series("HOLD", index=df.index)

    # Ensure required columns exist
    required = ["Close", "RSI", "MACD", "MACD_Signal", "SMA_20", "SMA_50"]
    for col in required:
        if col not in df.columns:
            logger.warning(f"Missing column {col} — signals may be incomplete")
            return signals

    close       = df["Close"]
    rsi         = df["RSI"]
    macd        = df["MACD"]
    macd_signal = df["MACD_Signal"]
    sma20       = df["SMA_20"]

    # Price crosses above SMA20 (previous day below, today above)
    cross_above_sma20 = (close > sma20) & (close.shift(1) <= sma20.shift(1))
    cross_below_sma20 = (close < sma20) & (close.shift(1) >= sma20.shift(1))

    # BUY conditions
    buy_condition = (
        (rsi < 40) &
        (macd > macd_signal) &
        (close > sma20)
    ) | cross_above_sma20

    # SELL conditions
    sell_condition = (
        (rsi > 70) |
        (cross_below_sma20 & (macd < macd_signal))
    )

    signals[buy_condition]  = "BUY"
    signals[sell_condition] = "SELL"

    # Don't generate signals before indicators are ready (first 50 days)
    warmup = min(20, len(df) - 1)
    signals.iloc[:warmup] = "HOLD"

    return signals


# ─────────────────────────────────────────────
# 2. Run Backtest for Single Stock
# ─────────────────────────────────────────────

def backtest_ticker(
    ticker: str,
    period: str = "1y",
    starting_cash: float = STARTING_CASH,
) -> dict:
    """
    Runs a backtest for a single ticker over the given period.

    Args:
        ticker:        stock symbol e.g. "AAPL"
        period:        yfinance period string e.g. "1y", "2y", "6mo"
        starting_cash: virtual cash to start with

    Returns a dict with full trade history and performance metrics.
    """
    logger.info(f"Running backtest for {ticker} over {period}...")

    # Fetch price history
    try:
        df = yf.download(ticker, period=period, interval="1d", progress=False, auto_adjust=True)
        if df.empty:
            return {"error": f"No data for {ticker}"}
        df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
    except Exception as e:
        return {"error": str(e)}

    # Add technical indicators
    try:
        from data_collector import add_technical_indicators
        df["Ticker"] = ticker
        df = add_technical_indicators(df)
    except Exception as e:
        logger.warning(f"Could not add indicators: {e}")
        return {"error": "Could not calculate indicators"}

    # Generate signals
    df["Signal"] = generate_signals(df)

    # ── Simulate trades ───────────────────────
    cash        = starting_cash
    shares      = 0.0
    trades      = []
    portfolio   = []

    for date, row in df.iterrows():
        price  = float(row["Close"])
        signal = row["Signal"]

        if signal == "BUY" and cash > price:
            # Buy as many shares as possible
            shares_to_buy = (cash - COMMISSION) // price
            if shares_to_buy > 0:
                cost   = shares_to_buy * price + COMMISSION
                cash  -= cost
                shares += shares_to_buy
                trades.append({
                    "date":   date,
                    "action": "BUY",
                    "shares": shares_to_buy,
                    "price":  round(price, 2),
                    "value":  round(cost, 2),
                    "cash":   round(cash, 2),
                })

        elif signal == "SELL" and shares > 0:
            # Sell all shares
            proceeds = shares * price - COMMISSION
            cash    += proceeds
            trades.append({
                "date":   date,
                "action": "SELL",
                "shares": shares,
                "price":  round(price, 2),
                "value":  round(proceeds, 2),
                "cash":   round(cash, 2),
            })
            shares = 0.0

        # Track portfolio value daily
        portfolio_value = cash + shares * price
        portfolio.append({
            "date":            date,
            "price":           round(price, 2),
            "signal":          signal,
            "shares":          shares,
            "cash":            round(cash, 2),
            "portfolio_value": round(portfolio_value, 2),
        })

    # Final portfolio value (liquidate remaining shares)
    final_price          = float(df["Close"].iloc[-1])
    final_portfolio_value = cash + shares * final_price
    valid_idx = min(50, len(df) - 1)
    start_price = float(df["Close"].iloc[valid_idx])
    end_price            = final_price

    # ── Buy and hold comparison ───────────────
    bah_shares = (starting_cash - COMMISSION) // start_price
    bah_value  = bah_shares * end_price + (starting_cash - bah_shares * start_price)

    # ── Performance metrics ───────────────────
    portfolio_df = pd.DataFrame(portfolio)
    portfolio_df["date"] = pd.to_datetime(portfolio_df["date"])

    total_return     = ((final_portfolio_value - starting_cash) / starting_cash) * 100
    bah_return       = ((bah_value - starting_cash) / starting_cash) * 100
    alpha            = total_return - bah_return
    num_trades       = len(trades)
    num_buys         = len([t for t in trades if t["action"] == "BUY"])
    num_sells        = len([t for t in trades if t["action"] == "SELL"])

    # Win rate — percentage of sell trades that were profitable
    win_rate = _calculate_win_rate(trades)

    # Max drawdown
    portfolio_values = portfolio_df["portfolio_value"]
    rolling_max      = portfolio_values.cummax()
    drawdowns        = (portfolio_values - rolling_max) / rolling_max * 100
    max_drawdown     = round(float(drawdowns.min()), 2)

    # Sharpe ratio (simplified — daily returns, annualized)
    portfolio_df["daily_return"] = portfolio_df["portfolio_value"].pct_change()
    sharpe = _calculate_sharpe(portfolio_df["daily_return"])

    result = {
        "ticker":               ticker,
        "period":               period,
        "starting_cash":        starting_cash,
        "final_value":          round(final_portfolio_value, 2),
        "total_return_pct":     round(total_return, 2),
        "buy_and_hold_value":   round(bah_value, 2),
        "buy_and_hold_return":  round(bah_return, 2),
        "alpha":                round(alpha, 2),
        "num_trades":           num_trades,
        "num_buys":             num_buys,
        "num_sells":            num_sells,
        "win_rate":             win_rate,
        "max_drawdown":         max_drawdown,
        "sharpe_ratio":         sharpe,
        "outperformed":         total_return > bah_return,
        "trades":               trades,
        "portfolio_history":    portfolio_df.to_dict("records"),
        "signal_counts":        df["Signal"].value_counts().to_dict(),
    }

    logger.info(
        f"{ticker}: Strategy {total_return:+.1f}% vs B&H {bah_return:+.1f}% "
        f"| Alpha: {alpha:+.1f}% | Trades: {num_trades} | Win rate: {win_rate}%"
    )

    return result


def _calculate_win_rate(trades: list[dict]) -> float:
    """Calculates the percentage of round-trip trades that were profitable."""
    buys  = [t for t in trades if t["action"] == "BUY"]
    sells = [t for t in trades if t["action"] == "SELL"]

    if not buys or not sells:
        return 0.0

    wins = 0
    pairs = min(len(buys), len(sells))
    for i in range(pairs):
        if sells[i]["price"] > buys[i]["price"]:
            wins += 1

    return round((wins / pairs) * 100, 1) if pairs > 0 else 0.0


def _calculate_sharpe(daily_returns: pd.Series, risk_free_rate: float = 0.05) -> float:
    """
    Calculates annualized Sharpe ratio.
    Risk-free rate defaults to 5% (approximate current T-bill rate).
    """
    daily_returns = daily_returns.dropna()
    if len(daily_returns) < 2:
        return 0.0
    daily_rf = risk_free_rate / 252
    excess   = daily_returns - daily_rf
    if excess.std() == 0:
        return 0.0
    sharpe = (excess.mean() / excess.std()) * np.sqrt(252)
    return round(float(sharpe), 2)


# ─────────────────────────────────────────────
# 3. Backtest Watchlist
# ─────────────────────────────────────────────

def backtest_watchlist(
    tickers: list[str],
    period: str = "1y",
    starting_cash: float = STARTING_CASH,
) -> dict[str, dict]:
    """
    Runs backtests for all tickers in a watchlist.
    Returns { ticker: backtest_result }
    """
    results = {}
    for ticker in tickers:
        results[ticker] = backtest_ticker(ticker, period=period, starting_cash=starting_cash)
    return results


def backtest_summary(results: dict[str, dict]) -> dict:
    """
    Summarizes backtest results across all tickers.
    """
    valid = {k: v for k, v in results.items() if "error" not in v}
    if not valid:
        return {"error": "No valid backtest results"}

    returns      = [v["total_return_pct"] for v in valid.values()]
    bah_returns  = [v["buy_and_hold_return"] for v in valid.values()]
    alphas       = [v["alpha"] for v in valid.values()]
    outperformed = sum(1 for v in valid.values() if v["outperformed"])

    return {
        "tickers_tested":        len(valid),
        "avg_strategy_return":   round(np.mean(returns), 2),
        "avg_bah_return":        round(np.mean(bah_returns), 2),
        "avg_alpha":             round(np.mean(alphas), 2),
        "outperformed_count":    outperformed,
        "outperformed_pct":      round(outperformed / len(valid) * 100, 1),
        "best_ticker":           max(valid, key=lambda k: valid[k]["alpha"]),
        "worst_ticker":          min(valid, key=lambda k: valid[k]["alpha"]),
        "results":               valid,
    }


# ─────────────────────────────────────────────
# 4. Print Report
# ─────────────────────────────────────────────

def print_backtest_report(results: dict[str, dict]) -> None:
    """Prints a formatted backtest report to the terminal."""
    summary = backtest_summary(results)

    print("\n" + "=" * 65)
    print("  BACKTEST REPORT")
    print("  DISCLAIMER: Past performance does not guarantee future results.")
    print("=" * 65)

    for ticker, r in results.items():
        if "error" in r:
            print(f"\n{ticker}: ERROR — {r['error']}")
            continue

        beat = "✓ beat" if r["outperformed"] else "✗ missed"
        print(f"\n── {ticker} ──────────────────────────────────────────")
        print(f"  Strategy return   : {r['total_return_pct']:+.1f}%  (${r['final_value']:,.0f})")
        print(f"  Buy & Hold return : {r['buy_and_hold_return']:+.1f}%  (${r['buy_and_hold_value']:,.0f})")
        print(f"  Alpha             : {r['alpha']:+.1f}%  {beat} buy & hold")
        print(f"  Trades            : {r['num_trades']} ({r['num_buys']} buys, {r['num_sells']} sells)")
        print(f"  Win rate          : {r['win_rate']}%")
        print(f"  Max drawdown      : {r['max_drawdown']}%")
        print(f"  Sharpe ratio      : {r['sharpe_ratio']}")

    if "error" not in summary:
        print(f"\n── SUMMARY ────────────────────────────────────────────")
        print(f"  Avg strategy return : {summary['avg_strategy_return']:+.1f}%")
        print(f"  Avg B&H return      : {summary['avg_bah_return']:+.1f}%")
        print(f"  Avg alpha           : {summary['avg_alpha']:+.1f}%")
        print(f"  Beat B&H            : {summary['outperformed_count']}/{summary['tickers_tested']} stocks")
    print("=" * 65)


# ─────────────────────────────────────────────
# 5. Demo
# ─────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 65)
    print("Backtesting Module — Demo")
    print("DISCLAIMER: Educational use only. Not financial advice.")
    print("=" * 65)

    tickers = ["AAPL", "MSFT", "NVDA"]
    results = backtest_watchlist(tickers, period="1y")
    print_backtest_report(results)
