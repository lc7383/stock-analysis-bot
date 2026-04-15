"""
earnings_calendar.py  —  Earnings Calendar Awareness
======================================================
Fetches upcoming and recent earnings dates for a list of tickers
and flags stocks where predictions should be treated with caution.

No new API key required — uses yfinance which is already installed.

Usage:
    from earnings_calendar import get_earnings_context, flag_earnings_risk

    context = get_earnings_context(["AAPL", "NVDA", "MSFT"])
    for ticker, info in context.items():
        print(ticker, info["risk_level"], info["summary"])
"""

import yfinance as yf
from datetime import datetime, timedelta, date
import pandas as pd


# ── Risk thresholds (days) ────────────────────────────────────────────────
CRITICAL_WINDOW = 2    # earnings within 2 days  → HIGH risk, avoid predictions
WARNING_WINDOW  = 5    # earnings within 5 days  → MEDIUM risk, reduce confidence
RECENT_WINDOW   = 2    # earnings within last 2 days → just reported, volatile


def get_earnings_context(tickers: list[str]) -> dict:
    """
    For each ticker return earnings timing info and a risk assessment.

    Returns dict keyed by ticker:
    {
        "next_earnings_date":  date | None,
        "last_earnings_date":  date | None,
        "days_until_earnings": int | None,
        "days_since_earnings": int | None,
        "risk_level":          "HIGH" | "MEDIUM" | "LOW" | "UNKNOWN",
        "risk_reason":         str,
        "summary":             str,   # plain-English for Claude's system prompt
        "should_skip":         bool,  # True = don't make a prediction
        "confidence_adj":      float, # multiply model confidence by this (0.0–1.0)
    }
    """
    results = {}
    today   = date.today()

    for ticker in tickers:
        try:
            stock = yf.Ticker(ticker)
            cal   = stock.calendar  # dict with "Earnings Date" key

            next_date = None
            last_date = None

            # ── Parse next earnings date ──────────────────────────────
            if cal is not None and not (isinstance(cal, pd.DataFrame) and cal.empty):
                # yfinance returns either a dict or a DataFrame depending on version
                if isinstance(cal, dict):
                    raw = cal.get("Earnings Date")
                    if raw:
                        if isinstance(raw, (list, tuple)) and len(raw) > 0:
                            raw = raw[0]
                        if hasattr(raw, "date"):
                            next_date = raw.date()
                        elif isinstance(raw, (datetime, date)):
                            next_date = raw if isinstance(raw, date) else raw.date()
                elif isinstance(cal, pd.DataFrame) and not cal.empty:
                    try:
                        raw = cal.loc["Earnings Date"].iloc[0] if "Earnings Date" in cal.index else None
                        if raw is not None and hasattr(raw, "date"):
                            next_date = raw.date()
                    except Exception:
                        pass

            # ── Parse last earnings date from earnings history ────────
            try:
                hist = stock.earnings_dates
                if hist is not None and not hist.empty:
                    past = hist[hist.index.date < today]
                    if not past.empty:
                        last_date = past.index[0].date()
            except Exception:
                pass

            # ── Calculate days ────────────────────────────────────────
            days_until  = (next_date - today).days if next_date else None
            days_since  = (today - last_date).days if last_date else None

            # ── Assess risk ───────────────────────────────────────────
            risk_level    = "UNKNOWN"
            risk_reason   = "No earnings date data available."
            should_skip   = False
            confidence_adj = 1.0

            if days_until is not None and days_until >= 0:
                if days_until <= CRITICAL_WINDOW:
                    risk_level     = "HIGH"
                    risk_reason    = f"Earnings in {days_until} day(s) — price can move 10-20% on the report."
                    should_skip    = True
                    confidence_adj = 0.0   # skip prediction entirely
                elif days_until <= WARNING_WINDOW:
                    risk_level     = "MEDIUM"
                    risk_reason    = f"Earnings in {days_until} day(s) — elevated uncertainty."
                    should_skip    = False
                    confidence_adj = 0.6
                else:
                    risk_level     = "LOW"
                    risk_reason    = f"Next earnings in {days_until} day(s) — outside risk window."
                    confidence_adj = 1.0
            elif days_since is not None and days_since <= RECENT_WINDOW:
                risk_level     = "MEDIUM"
                risk_reason    = f"Earnings reported {days_since} day(s) ago — post-earnings volatility may persist."
                confidence_adj = 0.75
            elif days_since is not None:
                risk_level     = "LOW"
                risk_reason    = f"Last earnings {days_since} day(s) ago — well outside earnings window."
                confidence_adj = 1.0

            # ── Plain-English summary for Claude ──────────────────────
            parts = []
            if next_date:
                parts.append(f"Next earnings: {next_date.strftime('%b %d, %Y')} ({days_until} days away)")
            if last_date:
                parts.append(f"Last earnings: {last_date.strftime('%b %d, %Y')} ({days_since} days ago)")
            parts.append(f"Earnings risk: {risk_level} — {risk_reason}")
            if should_skip:
                parts.append("⚠ Prediction skipped — too close to earnings.")
            elif confidence_adj < 1.0:
                parts.append(f"Model confidence reduced to {int(confidence_adj * 100)}% of normal due to earnings proximity.")

            results[ticker] = {
                "next_earnings_date":  next_date,
                "last_earnings_date":  last_date,
                "days_until_earnings": days_until,
                "days_since_earnings": days_since,
                "risk_level":          risk_level,
                "risk_reason":         risk_reason,
                "summary":             " | ".join(parts),
                "should_skip":         should_skip,
                "confidence_adj":      confidence_adj,
            }

        except Exception as e:
            results[ticker] = {
                "next_earnings_date":  None,
                "last_earnings_date":  None,
                "days_until_earnings": None,
                "days_since_earnings": None,
                "risk_level":          "UNKNOWN",
                "risk_reason":         f"Could not fetch earnings data: {e}",
                "summary":             f"Earnings data unavailable: {e}",
                "should_skip":         False,
                "confidence_adj":      1.0,
            }

    return results


def flag_earnings_risk(tickers: list[str]) -> dict:
    """
    Lightweight wrapper — returns just the risk level and summary per ticker.
    Useful for injecting into Claude's system prompt.
    """
    context = get_earnings_context(tickers)
    return {
        ticker: {
            "risk_level":  info["risk_level"],
            "summary":     info["summary"],
            "should_skip": info["should_skip"],
            "confidence_adj": info["confidence_adj"],
        }
        for ticker, info in context.items()
    }


def earnings_prompt_block(tickers: list[str]) -> str:
    """
    Returns a formatted text block ready to inject into Claude's system prompt.
    """
    context = get_earnings_context(tickers)
    lines   = ["=== EARNINGS CALENDAR ==="]
    for ticker, info in context.items():
        lines.append(f"[{ticker}] {info['summary']}")
    lines.append("")
    lines.append(
        "IMPORTANT: For any ticker marked HIGH earnings risk, do NOT make a "
        "directional prediction. Instead flag the earnings date and recommend "
        "the user wait until after the report before acting."
    )
    return "\n".join(lines)


# ── Quick test ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    test_tickers = ["AAPL", "NVDA", "MSFT", "TSLA", "META"]
    print("Fetching earnings calendar...\n")
    ctx = get_earnings_context(test_tickers)
    for ticker, info in ctx.items():
        print(f"{ticker:6} | Risk: {info['risk_level']:7} | {info['risk_reason']}")
        if info["next_earnings_date"]:
            print(f"       | Next:  {info['next_earnings_date']} ({info['days_until_earnings']} days)")
        if info["last_earnings_date"]:
            print(f"       | Last:  {info['last_earnings_date']} ({info['days_since_earnings']} days ago)")
        print()
