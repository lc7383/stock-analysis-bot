"""
chat_page.py  —  StockMind AI Chat Page
========================================
Drop this file in your project root alongside dashboard.py.

Then add one line to your dashboard.py sidebar navigation, e.g.:

    from chat_page import render_chat_page

    # Inside your page routing (wherever you handle sidebar selection):
    if page == "AI Chat":
        render_chat_page()

And add "AI Chat" to your sidebar page list, e.g.:
    page = st.sidebar.radio("Navigation", [
        "Screener", "Latest Report", "History",
        "Run Analysis", "Backtest", "Predictions",
        "Watchlist & Alerts", "AI Chat"          # <-- add this
    ])
"""

import json
import glob
import os
from pathlib import Path
from datetime import datetime

import streamlit as st
from anthropic import Anthropic

# ── Constants ──────────────────────────────────────────────────────────────
REPORTS_DIR   = Path("reports")
MODEL         = "claude-sonnet-4-20250514"
MAX_TOKENS    = 1000
HISTORY_LIMIT = 20   # max messages kept in context to avoid token bloat

# ── Data loaders ───────────────────────────────────────────────────────────

def load_latest_report() -> dict | None:
    """Return the most recent timestamped JSON report, or None."""
    files = sorted(glob.glob(str(REPORTS_DIR / "report_*.json")), reverse=True)
    if not files:
        return None
    try:
        with open(files[0]) as f:
            return json.load(f)
    except Exception:
        return None


def load_history_tail(n: int = 5) -> list[dict]:
    """Return last n entries from history.jsonl."""
    path = REPORTS_DIR / "history.jsonl"
    if not path.exists():
        return []
    lines = path.read_text().strip().splitlines()
    entries = []
    for line in lines[-n:]:
        try:
            entries.append(json.loads(line))
        except Exception:
            continue
    return entries


def format_report_for_prompt(report: dict | None) -> str:
    if not report:
        return "No report data available yet. The scheduler has not completed a run."

    lines = [f"Latest analysis run: {report.get('timestamp', 'unknown')}\n"]
    stocks = report.get("analysis", report)

    if isinstance(stocks, dict):
        for ticker, data in stocks.items():
            if not isinstance(data, dict):
                continue
            rec    = data.get("recommendation", "—")
            conf   = data.get("confidence", "—")
            frame  = data.get("target_timeframe", "—")
            reason = data.get("reasoning", "")[:200]
            risks  = "; ".join(data.get("risk_factors", [])[:2])
            sigs   = data.get("key_signals", {})
            tech   = (sigs.get("technical", "")[:120]   if isinstance(sigs, dict) else "")
            fund   = (sigs.get("fundamental", "")[:120] if isinstance(sigs, dict) else "")
            sent   = (sigs.get("sentiment", "")[:80]    if isinstance(sigs, dict) else "")
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
    report       = load_latest_report()
    history      = load_history_tail(5)
    report_text  = format_report_for_prompt(report)
    watchlist_str = ", ".join(watchlist) if watchlist else "not set"

    history_summary = ""
    if history:
        history_summary = "Recent run history (last 5 entries):\n"
        for e in history:
            ts    = e.get("timestamp", "?")
            ticks = ", ".join(e.get("tickers", []))
            history_summary += f"  • {ts}: {ticks}\n"

    return f"""You are StockMind, an expert AI stock analyst embedded in a Streamlit stock analysis dashboard.

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
3. Help interpret specific values (e.g. "RSI 28 + bullish MACD means…").
4. Compare stocks and prioritize opportunities from the actual data.
5. Suggest strategy improvements and flag risks honestly.
6. Be direct — traders want signal, not fluff.

When discussing specific metrics from the report, cite them explicitly
(e.g. "NVDA has a confidence of 8/10 in the latest run").
Always note that analysis is educational and not financial advice."""


# ── Streamlit page ─────────────────────────────────────────────────────────

def render_chat_page():
    # ── Session state init ──────────────────────────────────────────────
    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = []   # list of {role, content}
    if "chat_watchlist" not in st.session_state:
        # Try to pull watchlist from elsewhere in session_state, else default
        st.session_state.chat_watchlist = st.session_state.get(
            "watchlist", ["AAPL", "MSFT", "NVDA", "TSLA", "META"]
        )

    client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

    # ── Page header ─────────────────────────────────────────────────────
    st.title("🤖 StockMind AI Chat")
    st.caption("Ask me anything about your watchlist, signals, or strategy.")

    # ── Sidebar controls ────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("---")
        st.markdown("**💬 Chat Settings**")

        # Report status
        report = load_latest_report()
        if report:
            ts = report.get("timestamp", "unknown")
            st.success(f"📊 Live data loaded\n\n{ts}")
        else:
            st.warning("⚠️ No report data yet.\nRun `python analysis_agent.py` first.")

        # Watchlist editor
        wl_input = st.text_input(
            "Watchlist",
            value=", ".join(st.session_state.chat_watchlist),
            help="Comma-separated tickers"
        )
        if wl_input:
            st.session_state.chat_watchlist = [
                t.strip().upper() for t in wl_input.split(",") if t.strip()
            ]

        if st.button("🗑 Clear chat", use_container_width=True):
            st.session_state.chat_messages = []
            st.rerun()

        st.markdown("---")
        st.caption("Powered by Claude claude-sonnet-4-6")

    # ── Quick suggestion chips ───────────────────────────────────────────
    if not st.session_state.chat_messages:
        st.markdown("**Quick questions:**")
        suggestions = [
            "Which stocks look most oversold?",
            "Summarise today's signals",
            "Compare signals across my watchlist",
            "Where do my ML models agree?",
            "Explain the Bollinger Band signals",
            "Best stocks for high volatility?",
        ]
        cols = st.columns(3)
        for i, suggestion in enumerate(suggestions):
            if cols[i % 3].button(suggestion, use_container_width=True, key=f"chip_{i}"):
                st.session_state._chip_input = suggestion
                st.rerun()

        st.markdown("---")

    # ── Render existing messages ─────────────────────────────────────────
    for msg in st.session_state.chat_messages:
        with st.chat_message(msg["role"], avatar="📈" if msg["role"] == "assistant" else None):
            st.markdown(msg["content"])

    # ── Handle chip pre-fill ─────────────────────────────────────────────
    prefill = st.session_state.pop("_chip_input", None)

    # ── Chat input ───────────────────────────────────────────────────────
    user_input = st.chat_input("Ask about your watchlist, signals, or strategy…")

    # Use chip prefill if present
    if prefill and not user_input:
        user_input = prefill

    if user_input:
        # Show user message
        st.session_state.chat_messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        # Trim history to avoid token bloat
        trimmed = st.session_state.chat_messages[-HISTORY_LIMIT:]

        # Call Claude
        with st.chat_message("assistant", avatar="📈"):
            with st.spinner("Analysing…"):
                try:
                    response = client.messages.create(
                        model=MODEL,
                        max_tokens=MAX_TOKENS,
                        system=build_system_prompt(st.session_state.chat_watchlist),
                        messages=[
                            {"role": m["role"], "content": m["content"]}
                            for m in trimmed
                        ],
                    )
                    reply = "".join(
                        block.text for block in response.content
                        if block.type == "text"
                    )
                except Exception as e:
                    reply = f"⚠️ Error: {e}\n\nCheck that `ANTHROPIC_API_KEY` is set in your `.env` file."

            st.markdown(reply)

        st.session_state.chat_messages.append({"role": "assistant", "content": reply})
