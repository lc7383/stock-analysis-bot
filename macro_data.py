"""
Macro & Sentiment Data Module
================================
Fetches two powerful free data sources to enrich stock analysis:

1. FRED (Federal Reserve Economic Data)
   - Interest rates, inflation, unemployment, GDP growth
   - Free API key at: https://fred.stlouisfed.org/docs/api/api_key.html
   - Add to .env: FRED_API_KEY=your_key_here

2. CNN Fear & Greed Index
   - Single 0-100 score of overall market sentiment
   - No API key needed

These macro signals help Claude understand the broader market
environment when making stock recommendations.

Requirements:
    pip install fredapi requests

DISCLAIMER: For educational/portfolio purposes only. Not financial advice.
"""

import os
import logging
import requests
from datetime import datetime, timedelta
from typing import Optional

from dotenv import load_dotenv

try:
    from fredapi import Fred
    FRED_AVAILABLE = True
except ImportError:
    FRED_AVAILABLE = False
    logging.warning("fredapi not installed. Run: pip install fredapi")

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

FRED_API_KEY = os.getenv("FRED_API_KEY")


# ─────────────────────────────────────────────
# 1. Fear & Greed Index (no API key needed)
# ─────────────────────────────────────────────

def fetch_fear_and_greed() -> dict:
    """
    Fetches the CNN Fear & Greed Index — a 0-100 composite score
    measuring overall market sentiment.

    Score interpretation:
        0-24   = Extreme Fear    (potential buying opportunity)
        25-44  = Fear
        45-55  = Neutral
        56-74  = Greed
        75-100 = Extreme Greed  (market may be overheated)

    Returns a dict with score, label, and previous values.
    """
    url = "https://api.alternative.me/fng/"
    headers = {
        "User-Agent": "Mozilla/5.0",
        "Referer":    "https://www.cnn.com/markets/fear-and-greed",
    }

    try:
        resp = requests.get(url, headers=headers, timeout=10)
        resp.raise_for_status()
        data = resp.json()

        item  = data.get("data", [{}])[0]
        score = round(float(item.get("value", 50)), 1)
        rating = item.get("value_classification", "Neutral").title()

        result = {
            "score":           score,
            "label":           rating,
            "previous_week":   None,
            "previous_year":   None,
            "interpretation":  _interpret_fg(score),
            "week_trend":      "unknown",
        }

        logger.info(f"Fear & Greed Index: {score} ({rating})")
        return result

    except Exception as e:
        logger.error(f"Fear & Greed fetch failed: {e}")
        return _fallback_fear_greed()


def _interpret_fg(score: float) -> str:
    if score <= 24:  return "Extreme Fear — market panic, potential buying opportunity"
    if score <= 44:  return "Fear — investors cautious, valuations may be depressed"
    if score <= 55:  return "Neutral — balanced market sentiment"
    if score <= 74:  return "Greed — bullish momentum, watch for overvaluation"
    return "Extreme Greed — market euphoria, elevated crash risk"


def _trend(current: float, previous: Optional[float]) -> str:
    if previous is None:
        return "unknown"
    diff = current - previous
    if diff > 5:   return "rising sharply"
    if diff > 1:   return "rising"
    if diff < -5:  return "falling sharply"
    if diff < -1:  return "falling"
    return "stable"


def _fallback_fear_greed() -> dict:
    """Returns a neutral fallback if the API is unavailable."""
    return {
        "score":           50,
        "label":           "Neutral",
        "previous_week":   None,
        "previous_year":   None,
        "interpretation":  "Data unavailable — defaulting to neutral",
        "week_trend":      "unknown",
    }


# ─────────────────────────────────────────────
# 2. FRED Economic Indicators
# ─────────────────────────────────────────────

# Key FRED series IDs and their human-readable names
FRED_SERIES = {
    "DFF":     "fed_funds_rate",        # Federal Funds Rate (daily)
    "T10YIE":  "inflation_expectation", # 10-Year Breakeven Inflation Rate
    "UNRATE":  "unemployment_rate",     # Unemployment Rate (monthly)
    "GDP":     "gdp_growth",            # GDP (quarterly)
    "T10Y2Y":  "yield_curve",           # 10Y-2Y Treasury Spread (recession indicator)
    "VIXCLS":  "vix",                   # CBOE Volatility Index
}


def fetch_fred_indicators() -> dict:
    """
    Fetches key macroeconomic indicators from the Federal Reserve (FRED).

    Requires a free FRED API key — get one at:
    https://fred.stlouisfed.org/docs/api/api_key.html

    Add to your .env file:
        FRED_API_KEY=your_key_here

    Returns a dict of the latest values for each indicator,
    plus a macro_summary string ready for Claude to interpret.
    """
    if not FRED_AVAILABLE:
        logger.warning("fredapi not installed — skipping FRED data. Run: pip install fredapi")
        return _fallback_fred()

    if not FRED_API_KEY:
        logger.warning("No FRED_API_KEY in .env — skipping FRED data.")
        return _fallback_fred()

    fred    = Fred(api_key=FRED_API_KEY)
    results = {}

    for series_id, name in FRED_SERIES.items():
        try:
            series = fred.get_series(series_id, observation_start=datetime.now() - timedelta(days=90))
            series = series.dropna()
            if not series.empty:
                latest_value = round(float(series.iloc[-1]), 3)
                latest_date  = series.index[-1].strftime("%Y-%m-%d")
                results[name] = {
                    "value": latest_value,
                    "date":  latest_date,
                }
                logger.info(f"FRED {series_id}: {latest_value} ({latest_date})")
        except Exception as e:
            logger.warning(f"FRED series {series_id} failed: {e}")
            results[name] = None

    results["macro_context"] = _build_macro_context(results)
    return results


def _build_macro_context(indicators: dict) -> str:
    """
    Builds a plain-English macro context string for Claude to use.
    """
    parts = []

    fed_rate = indicators.get("fed_funds_rate")
    if fed_rate:
        v = fed_rate["value"]
        parts.append(f"Fed funds rate is {v}% ({'restrictive, pressuring valuations' if v > 4 else 'accommodative, supportive of equities'})")

    inflation = indicators.get("inflation_expectation")
    if inflation:
        v = inflation["value"]
        parts.append(f"10-year inflation expectation is {v}% ({'elevated' if v > 2.5 else 'anchored'})")

    unemployment = indicators.get("unemployment_rate")
    if unemployment:
        v = unemployment["value"]
        parts.append(f"Unemployment is {v}% ({'low, strong labor market' if v < 4.5 else 'elevated, weakening labor market'})")

    yield_curve = indicators.get("yield_curve")
    if yield_curve:
        v = yield_curve["value"]
        if v < 0:
            parts.append(f"Yield curve is inverted at {v}% (recession warning signal)")
        else:
            parts.append(f"Yield curve is positive at {v}% (normal, no recession signal)")

    vix = indicators.get("vix")
    if vix:
        v = vix["value"]
        parts.append(f"VIX volatility index is {v} ({'high fear' if v > 25 else 'low fear, complacent market' if v < 15 else 'moderate'})")

    return ". ".join(parts) + "." if parts else "Macro data unavailable."


def _fallback_fred() -> dict:
    """Returns an empty fallback if FRED is unavailable."""
    return {
        name: None for name in FRED_SERIES.values()
    } | {"macro_context": "FRED data unavailable — no API key or library not installed."}


# ─────────────────────────────────────────────
# 3. Combined Macro Context
# ─────────────────────────────────────────────

def fetch_macro_context() -> dict:
    """
    Fetches all macro and sentiment data in one call.

    Returns a unified dict with:
        fear_and_greed   — CNN Fear & Greed Index
        fred             — FRED economic indicators
        macro_summary    — plain English summary for Claude
    """
    logger.info("Fetching macro context...")

    fear_greed = fetch_fear_and_greed()
    fred       = fetch_fred_indicators()

    # Build a combined summary for Claude
    fg_summary   = f"Market sentiment: {fear_greed['label']} ({fear_greed['score']}/100, {fear_greed['week_trend']} vs last week). {fear_greed['interpretation']}."
    fred_summary = fred.get("macro_context", "")

    macro_summary = f"{fg_summary} {fred_summary}".strip()

    return {
        "fear_and_greed": fear_greed,
        "fred":           fred,
        "macro_summary":  macro_summary,
    }


# ─────────────────────────────────────────────
# 4. Demo
# ─────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("Macro & Sentiment Data — Demo")
    print("DISCLAIMER: Educational use only. Not financial advice.")
    print("=" * 60)

    context = fetch_macro_context()

    print("\n── Fear & Greed Index ──────────────────────────────")
    fg = context["fear_and_greed"]
    print(f"  Score      : {fg['score']}/100")
    print(f"  Label      : {fg['label']}")
    print(f"  Trend      : {fg['week_trend']} vs last week")
    print(f"  Meaning    : {fg['interpretation']}")

    print("\n── FRED Economic Indicators ────────────────────────")
    fred = context["fred"]
    for key, val in fred.items():
        if key != "macro_context" and val is not None:
            print(f"  {key:<25} {val['value']}  ({val['date']})")

    print("\n── Macro Summary for Claude ────────────────────────")
    print(f"  {context['macro_summary']}")
