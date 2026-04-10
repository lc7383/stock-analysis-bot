"""
Stock Analysis Dashboard
=========================
A Streamlit web dashboard that displays stock recommendations
and historical analysis from saved reports.

Requirements:
    pip install streamlit plotly pandas

Run with:
    streamlit run dashboard.py

DISCLAIMER: For educational/portfolio purposes only. Not financial advice.
"""
import streamlit as st
import os
# Load secrets from Streamlit Cloud or fall back to .env
try:
    os.environ["ANTHROPIC_API_KEY"] = st.secrets["ANTHROPIC_API_KEY"]
    os.environ["FRED_API_KEY"] = st.secrets["FRED_API_KEY"]
except Exception:
    pass  # Falls back to .env file when running locally
    



import json
import glob
from pathlib import Path
from datetime import datetime

import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

# ── Page config ───────────────────────────────
st.set_page_config(
    page_title="Stock Analysis Bot",
    page_icon="📈",
    layout="wide",
)
REPORTS_DIR = Path("reports")

# ── Helpers ───────────────────────────────────

def load_latest_report() -> dict | None:
    files = sorted(glob.glob(str(REPORTS_DIR / "report_*.json")), reverse=True)
    if not files:
        return None
    with open(files[0]) as f:
        return json.load(f)


def load_all_reports() -> list[dict]:
    files = sorted(glob.glob(str(REPORTS_DIR / "report_*.json")))
    reports = []
    for fp in files:
        with open(fp) as f:
            reports.append(json.load(f))
    return reports


def load_history() -> pd.DataFrame:
    log_path = REPORTS_DIR / "history.jsonl"
    if not log_path.exists():
        return pd.DataFrame()
    rows = []
    with open(log_path) as f:
        for line in f:
            rows.append(json.loads(line.strip()))
    df = pd.DataFrame(rows)
    if not df.empty:
        df["datetime"] = pd.to_datetime(df["timestamp"], format="%Y%m%d_%H%M%S")
    return df


def rec_color(rec: str) -> str:
    return {"BUY": "#22c55e", "HOLD": "#f59e0b", "SELL": "#ef4444"}.get(rec, "#6b7280")


def rec_icon(rec: str) -> str:
    return {"BUY": "▲", "HOLD": "●", "SELL": "▼"}.get(rec, "●")


# ── Sidebar ───────────────────────────────────

st.sidebar.title("📈 Stock Analysis Bot")
st.sidebar.caption("⚠️ Educational use only. Not financial advice.")
st.sidebar.divider()

page = st.sidebar.radio("View", ["Latest Report", "History", "Run Analysis"])

# ── Latest Report ─────────────────────────────

if page == "Latest Report":
    st.title("Latest Analysis")

    report = load_latest_report()
    if not report:
        st.warning("No reports found. Run the scheduler or analysis agent first.")
        st.stop()

    # Header
    date_str = report.get("analysis_date", "Unknown date")
    st.caption(f"Analysis date: {date_str}   |   {report.get('disclaimer', '')}")
    st.divider()

    # Market summary
    if summary := report.get("market_summary"):
        st.info(f"**Market Overview:** {summary}")

    st.subheader("Recommendations")

    recs = report.get("recommendations", [])
    cols = st.columns(len(recs))

    for col, rec in zip(cols, recs):
        action = rec["recommendation"]
        color  = rec_color(action)
        icon   = rec_icon(action)
        with col:
            st.markdown(
                f"""
                <div style="border:1px solid {color}; border-radius:12px; padding:16px; text-align:center">
                    <div style="font-size:2rem; color:{color}">{icon}</div>
                    <div style="font-size:1.4rem; font-weight:600">{rec['ticker']}</div>
                    <div style="font-size:1.1rem; color:{color}; font-weight:500">{action}</div>
                    <div style="color:#888; font-size:0.9rem">Confidence: {rec['confidence']}/10</div>
                    <div style="color:#888; font-size:0.85rem">{rec.get('target_timeframe','')}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.divider()

    # Detailed cards
    for rec in recs:
        action = rec["recommendation"]
        color  = rec_color(action)
        with st.expander(f"{rec_icon(action)}  {rec['ticker']} — {action}  (Confidence {rec['confidence']}/10)", expanded=True):
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**Technical**")
                st.write(rec["key_signals"].get("technical", "N/A"))
                st.markdown("**Fundamental**")
                st.write(rec["key_signals"].get("fundamental", "N/A"))
            with c2:
                st.markdown("**Reasoning**")
                st.write(rec.get("reasoning", ""))
                st.markdown("**Risk Factors**")
                st.write(rec.get("risk_factors", ""))

    # Confidence chart
    st.divider()
    st.subheader("Confidence Scores")
    fig = go.Figure(go.Bar(
        x=[r["ticker"] for r in recs],
        y=[r["confidence"] for r in recs],
        marker_color=[rec_color(r["recommendation"]) for r in recs],
        text=[f"{r['recommendation']} {r['confidence']}/10" for r in recs],
        textposition="outside",
    ))
    fig.update_layout(
        yaxis=dict(range=[0, 10], title="Confidence"),
        xaxis_title="Ticker",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        height=350,
    )
    st.plotly_chart(fig, use_container_width=True)


# ── History ───────────────────────────────────

elif page == "History":
    st.title("Analysis History")

    df = load_history()
    if df.empty:
        st.warning("No history found yet. Run the scheduler to build up history.")
        st.stop()

    # Filter by ticker
    tickers = sorted(df["ticker"].unique())
    selected = st.multiselect("Filter by ticker", tickers, default=tickers)
    df = df[df["ticker"].isin(selected)]

    # Recommendation over time
    st.subheader("Recommendations Over Time")
    rec_map = {"BUY": 1, "HOLD": 0, "SELL": -1}
    df["rec_value"] = df["recommendation"].map(rec_map)

    fig = px.line(
        df, x="datetime", y="confidence", color="ticker",
        markers=True, title="Confidence Score Over Time",
    )
    fig.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        height=350,
    )
    st.plotly_chart(fig, use_container_width=True)

    # Raw history table
    st.subheader("Full History")
    display_df = df[["datetime", "ticker", "recommendation", "confidence"]].sort_values("datetime", ascending=False)
    st.dataframe(display_df, use_container_width=True)

    # Download
    csv = display_df.to_csv(index=False)
    st.download_button("Download CSV", csv, "history.csv", "text/csv")


# ── Run Analysis ──────────────────────────────

elif page == "Run Analysis":
    st.title("Run Analysis Now")
    st.caption("⚠️ This will use your Anthropic API credits.")

    tickers_input = st.text_input(
        "Watchlist (comma separated)",
        value="AAPL, MSFT, NVDA",
    )

    if st.button("▶  Run Analysis", type="primary"):
        tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]
        if not tickers:
            st.error("Please enter at least one ticker.")
        else:
            with st.spinner(f"Collecting data and analyzing {', '.join(tickers)}..."):
                try:
                    from scheduler import run_and_save
                    from analysis_agent import run_pipeline
                    analysis = run_pipeline(tickers, save_json=True)
                    if "error" in analysis:
                        st.error(f"Analysis failed: {analysis['error']}")
                    else:
                        st.success("Analysis complete! View results in Latest Report.")
                        st.json(analysis)
                except Exception as e:
                    st.error(f"Error: {e}")
