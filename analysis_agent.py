"""
Stock Analysis Agent (with Macro + Insider Context)
=====================================================
Feeds stock data, macro/sentiment context, and SEC insider
trading signals into Claude for richer recommendations.

Requirements:
    pip install anthropic python-dotenv fredapi requests

DISCLAIMER: For educational/portfolio purposes only. Not financial advice.
"""

import os
import json
import logging
from datetime import datetime

import anthropic
from dotenv import load_dotenv

from data_collector import collect_watchlist, summaries_for_claude, DEFAULT_WATCHLIST
from macro_data import fetch_macro_context
from insider_trading import fetch_insider_data_watchlist

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

def get_client():
    api_key = os.getenv("ANTHROPIC_API_KEY")
    return anthropic.Anthropic(api_key=api_key)
MODEL  = "claude-sonnet-4-6"


SYSTEM_PROMPT = """You are a quantitative stock analysis assistant with expertise in
technical analysis, macroeconomics, and insider trading signals.

IMPORTANT DISCLAIMER: Your output is for educational and demonstration
purposes only. It does not constitute financial advice.

When analyzing stocks you will consider:
1. Technical indicators (RSI, MACD, Bollinger Bands, moving averages)
2. Fundamental health (P/E, growth, margins, beta)
3. Macro environment (interest rates, inflation, yield curve, VIX)
4. Market sentiment (Fear & Greed Index)
5. SEC insider trading signals (open market buys/sells by executives)

Insider trading notes:
- Open market PURCHASES by insiders = strong bullish signal (insiders rarely buy unless confident)
- Open market SALES may be routine (diversification, taxes) — weight less than buys
- Cluster buying (multiple insiders buying) = very strong bullish signal
- Awards/grants and option exercises are less meaningful than open market transactions

Always respond with valid JSON in exactly this format:
{
  "analysis_date": "YYYY-MM-DD",
  "disclaimer": "For educational purposes only. Not financial advice.",
  "macro_environment": "2-3 sentence macro summary",
  "recommendations": [
    {
      "ticker": "AAPL",
      "recommendation": "BUY",
      "confidence": 7,
      "target_timeframe": "1-4 weeks",
      "key_signals": {
        "technical": "Brief technical summary",
        "fundamental": "Brief fundamental summary",
        "macro_impact": "How macro conditions affect this stock",
        "insider_signal": "Summary of insider buying/selling activity"
      },
      "reasoning": "2-3 sentence explanation referencing all signal types",
      "risk_factors": "Key risks including macro and insider risks"
    }
  ],
  "market_summary": "1-2 sentence overview considering all data sources"
}"""


def build_user_prompt(summaries: list[dict], macro: dict, insider: dict) -> str:
    stocks_block = json.dumps(summaries, indent=2, default=str)
    macro_block  = json.dumps({
        "fear_and_greed": macro.get("fear_and_greed", {}),
        "macro_summary":  macro.get("macro_summary", ""),
        "fred_indicators": {
            k: v for k, v in macro.get("fred", {}).items()
            if k != "macro_context" and v is not None
        }
    }, indent=2, default=str)
    insider_block = json.dumps({
        ticker: {
            "signal":              data.get("signal"),
            "open_market_buys":    data.get("open_market_buys"),
            "open_market_sells":   data.get("open_market_sells"),
            "total_buy_value":     data.get("total_buy_value"),
            "total_sell_value":    data.get("total_sell_value"),
            "summary":             data.get("summary"),
            "recent_transactions": data.get("recent_transactions", [])[:3],
        }
        for ticker, data in insider.items()
    }, indent=2, default=str)

    return f"""Analyze the following stocks using all available data sources.

MACRO & SENTIMENT CONTEXT:
{macro_block}

SEC INSIDER TRADING DATA:
{insider_block}

STOCK MARKET DATA:
{stocks_block}

Respond with JSON only — no markdown, no preamble."""


def analyze_stocks(summaries: list[dict], macro: dict, insider: dict) -> dict:
    if not summaries:
        return {"error": "No summaries provided"}

    logger.info(f"Sending {len(summaries)} tickers + macro + insider data to Claude...")

    try:
        response = get_client().messages.create(
            model=MODEL,
            max_tokens=10000,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": build_user_prompt(summaries, macro, insider)}]
        )

        raw = response.content[0].text.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]

        result = json.loads(raw)
        logger.info("Analysis complete.")
        return result

    except json.JSONDecodeError as e:
        logger.error(f"Claude returned invalid JSON: {e}")
        return {"error": "JSON parse failed"}
    except Exception as e:
        logger.error(f"Claude API call failed: {e}")
        return {"error": str(e)}


ICONS  = {"BUY": "▲", "HOLD": "●", "SELL": "▼"}
COLORS = {"BUY": "\033[92m", "HOLD": "\033[93m", "SELL": "\033[91m", "RESET": "\033[0m"}


def print_report(analysis: dict) -> None:
    if "error" in analysis:
        print(f"\n✗ Analysis failed: {analysis['error']}")
        return

    print("\n" + "=" * 60)
    print("  STOCK ANALYSIS REPORT")
    print(f"  {analysis.get('analysis_date', datetime.utcnow().date())}")
    print("=" * 60)
    print(f"\n⚠  {analysis.get('disclaimer', '')}\n")

    if macro_env := analysis.get("macro_environment"):
        print(f"🌍 Macro: {macro_env}\n")

    for rec in analysis.get("recommendations", []):
        action = rec.get("recommendation", "HOLD")
        color  = COLORS.get(action, "")
        reset  = COLORS["RESET"]
        print(f"{color}{ICONS.get(action,'●')} {rec['ticker']:6}  {action:4}  Confidence: {rec['confidence']}/10{reset}")
        print(f"   Timeframe  : {rec.get('target_timeframe', 'N/A')}")
        print(f"   Technical  : {rec['key_signals'].get('technical', 'N/A')}")
        print(f"   Fundamental: {rec['key_signals'].get('fundamental', 'N/A')}")
        print(f"   Macro      : {rec['key_signals'].get('macro_impact', 'N/A')}")
        print(f"   Insider    : {rec['key_signals'].get('insider_signal', 'N/A')}")
        print(f"   Reasoning  : {rec.get('reasoning', '')}")
        print(f"   Risk       : {rec.get('risk_factors', '')}")
        print()

    if summary := analysis.get("market_summary"):
        print(f"Market overview: {summary}")
    print("=" * 60)


def save_report(analysis: dict, filepath: str = "report.json") -> None:
    with open(filepath, "w") as f:
        json.dump(analysis, f, indent=2, default=str)
    logger.info(f"Report saved to {filepath}")


def run_pipeline(tickers: list[str] = DEFAULT_WATCHLIST, save_json: bool = True, period: str = "6mo") -> dict:
    print(f"\nCollecting stock data for: {', '.join(tickers)}")
    watchlist_data = collect_watchlist(tickers, include_news=False, period=period)
    summaries      = summaries_for_claude(watchlist_data)

    print("Fetching macro & sentiment context...")
    macro = fetch_macro_context()
    print(f"Fear & Greed: {macro['fear_and_greed']['score']}/100 ({macro['fear_and_greed']['label']})")

    print("Fetching SEC insider trading data...")
    insider = fetch_insider_data_watchlist(tickers)
    for ticker, data in insider.items():
        print(f"  {ticker}: {data['signal']} — {data['signal_strength']}")

    analysis = analyze_stocks(summaries, macro, insider)
    print_report(analysis)

    if save_json and "error" not in analysis:
        save_report(analysis)

    return analysis


if __name__ == "__main__":
    print("=" * 60)
    print("Stock Analysis Agent (Macro + Insider Edition)")
    print("Powered by Claude " + MODEL)
    print("DISCLAIMER: Educational use only. Not financial advice.")
    print("=" * 60)

    MY_WATCHLIST = ["AAPL", "MSFT", "NVDA"]
    run_pipeline(MY_WATCHLIST)
