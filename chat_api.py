"""
chat_api.py  —  StockMind Chat Backend
Drop this file in your project root (same folder as analysis_agent.py).

Install dependency (if not already):
    pip install flask flask-cors

Run:
    python chat_api.py

Then open stock_chat.html in your browser (or add a link in dashboard.py).
The server runs on http://localhost:5050 by default.
"""

import json
import os
import glob
from datetime import datetime
from pathlib import Path

from flask import Flask, request, jsonify
from flask_cors import CORS
from anthropic import Anthropic
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
CORS(app)  # allows the HTML file to call this from any origin

client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

REPORTS_DIR = Path("reports")

# ── Load latest report data ────────────────────────────────────────────────

def load_latest_report() -> dict | None:
    """Return the most recent timestamped JSON report, or None if none exist."""
    pattern = str(REPORTS_DIR / "report_*.json")
    files = sorted(glob.glob(pattern), reverse=True)
    if not files:
        return None
    with open(files[0]) as f:
        return json.load(f)


def load_history_tail(n: int = 5) -> list[dict]:
    """Return the last n entries from history.jsonl."""
    history_path = REPORTS_DIR / "history.jsonl"
    if not history_path.exists():
        return []
    lines = history_path.read_text().strip().splitlines()
    entries = []
    for line in lines[-n:]:
        try:
            entries.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return entries


def format_report_for_prompt(report: dict) -> str:
    """Turn the latest report JSON into a compact text block for the system prompt."""
    if not report:
        return "No report data available yet — the scheduler has not run."

    lines = []
    timestamp = report.get("timestamp", "unknown time")
    lines.append(f"Latest analysis run: {timestamp}\n")

    stocks = report.get("analysis", report)  # support both wrapped and flat formats
    if isinstance(stocks, dict):
        for ticker, data in stocks.items():
            if not isinstance(data, dict):
                continue
            rec   = data.get("recommendation", "—")
            conf  = data.get("confidence", "—")
            frame = data.get("target_timeframe", "—")
            reason = data.get("reasoning", "")[:200]  # truncate for token efficiency
            risks  = "; ".join(data.get("risk_factors", [])[:2])

            # Key signals (nested dict)
            signals = data.get("key_signals", {})
            tech  = signals.get("technical", "")[:120] if isinstance(signals, dict) else ""
            fund  = signals.get("fundamental", "")[:120] if isinstance(signals, dict) else ""
            sent  = signals.get("sentiment", "")[:80]   if isinstance(signals, dict) else ""

            lines.append(
                f"[{ticker}]\n"
                f"  Recommendation: {rec}  |  Confidence: {conf}/10  |  Timeframe: {frame}\n"
                f"  Technical:    {tech}\n"
                f"  Fundamental:  {fund}\n"
                f"  Sentiment:    {sent}\n"
                f"  Reasoning:    {reason}\n"
                f"  Risks:        {risks}\n"
            )

    return "\n".join(lines)


def build_system_prompt(watchlist: list[str]) -> str:
    report      = load_latest_report()
    history     = load_history_tail(5)
    report_text = format_report_for_prompt(report)

    history_summary = ""
    if history:
        history_summary = "Recent run history (last 5 entries):\n"
        for entry in history:
            ts     = entry.get("timestamp", "?")
            ticks  = ", ".join(entry.get("tickers", []))
            history_summary += f"  • {ts}: {ticks}\n"

    watchlist_str = ", ".join(watchlist) if watchlist else "not set"

    return f"""You are StockMind, an expert AI stock analyst embedded in a Python stock analysis bot.

USER'S WATCHLIST: {watchlist_str}

=== LATEST BOT REPORT ===
{report_text}

=== RUN HISTORY ===
{history_summary or "No history available yet."}

=== BOT CAPABILITIES ===
The bot collects and analyzes:
- Technical indicators: RSI(14), MACD, Bollinger Bands, SMA 20/50
- Fundamentals: P/E, EPS, revenue growth, profit margins, beta (via yfinance)
- Macro: FRED indicators (Fed Funds Rate, Yield Curve, Unemployment, VIX),
         CNN Fear & Greed Index
- Insider signals: SEC EDGAR Form 4 filings
- ML models: Logistic Regression and Random Forest (next-day UP/DOWN prediction)
- Backtesting: strategy vs buy & hold with Sharpe, max drawdown, win rate

=== YOUR ROLE ===
1. Answer questions about the watchlist using the REAL report data above.
2. Explain signals and indicators clearly and concisely.
3. Help the user interpret specific values (e.g. "RSI 28 + bullish MACD means…").
4. Compare stocks and prioritize opportunities from the actual data.
5. Suggest strategy improvements and flag risks honestly.
6. Be direct — traders want signal, not fluff.

Always note that analysis is educational and not financial advice.
When discussing specific metrics from the report, cite them explicitly (e.g. "NVDA has a confidence of 8/10 in the latest run").
"""


# ── Routes ─────────────────────────────────────────────────────────────────

@app.route("/chat", methods=["POST"])
def chat():
    body = request.get_json(force=True)
    messages  = body.get("messages", [])
    watchlist = body.get("watchlist", [])

    if not messages:
        return jsonify({"error": "No messages provided"}), 400

    try:
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1000,
            system=build_system_prompt(watchlist),
            messages=messages,
        )
        reply = "".join(
            block.text for block in response.content if block.type == "text"
        )
        return jsonify({"reply": reply})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/report-status", methods=["GET"])
def report_status():
    """Let the frontend know whether report data is available."""
    report  = load_latest_report()
    history = load_history_tail(1)
    return jsonify({
        "has_report": report is not None,
        "last_run":   history[-1].get("timestamp") if history else None,
        "tickers":    list(report.get("analysis", {}).keys()) if report else [],
    })


if __name__ == "__main__":
    print("StockMind Chat API running on http://localhost:5050")
    print(f"Reports directory: {REPORTS_DIR.resolve()}")
    app.run(port=5050, debug=False)
