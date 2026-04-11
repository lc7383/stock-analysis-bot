"""
Stock Analysis Dashboard
=========================
Streamlit web dashboard with saved watchlist and email alerts.

DISCLAIMER: For educational/portfolio purposes only. Not financial advice.
"""

import streamlit as st
import os

# Load secrets from Streamlit Cloud or fall back to .env
try:
    os.environ["ANTHROPIC_API_KEY"] = st.secrets["ANTHROPIC_API_KEY"]
    os.environ["FRED_API_KEY"]      = st.secrets["FRED_API_KEY"]
    if "ALERT_EMAIL" in st.secrets:
        os.environ["ALERT_EMAIL"]          = st.secrets["ALERT_EMAIL"]
        os.environ["ALERT_EMAIL_PASSWORD"] = st.secrets["ALERT_EMAIL_PASSWORD"]
        os.environ["ALERT_TO_EMAIL"]       = st.secrets.get("ALERT_TO_EMAIL", st.secrets["ALERT_EMAIL"])
except Exception:
    pass

from pathlib import Path
import json
import glob
from datetime import datetime
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

st.set_page_config(
    page_title="Stock Analysis Bot",
    page_icon="📈",
    layout="wide",
)

REPORTS_DIR = Path("reports")

# ── Default watchlist ─────────────────────────
DEFAULT_WATCHLIST = ["AAPL", "MSFT", "NVDA"]

# ── Session state defaults ────────────────────
if "watchlist" not in st.session_state:
    st.session_state["watchlist"] = DEFAULT_WATCHLIST
if "latest_analysis" not in st.session_state:
    st.session_state["latest_analysis"] = None
if "alert_on_buy" not in st.session_state:
    st.session_state["alert_on_buy"] = True
if "alert_on_sell" not in st.session_state:
    st.session_state["alert_on_sell"] = True
if "min_confidence" not in st.session_state:
    st.session_state["min_confidence"] = 6


# ── Helpers ───────────────────────────────────
def load_latest_report() -> dict | None:
    files = sorted(glob.glob(str(REPORTS_DIR / "report_*.json")), reverse=True)
    if not files:
        return None
    with open(files[0]) as f:
        return json.load(f)


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


def rec_color(rec):
    return {"BUY": "#22c55e", "HOLD": "#f59e0b", "SELL": "#ef4444"}.get(rec, "#6b7280")


def rec_icon(rec):
    return {"BUY": "▲", "HOLD": "●", "SELL": "▼"}.get(rec, "●")


# ── Sidebar ───────────────────────────────────
st.sidebar.title("📈 Stock Analysis Bot")
st.sidebar.caption("⚠️ Educational use only. Not financial advice.")
st.sidebar.divider()

page = st.sidebar.radio("View", ["Latest Report", "History", "Run Analysis", "Backtest", "Watchlist & Alerts"])


# ── Latest Report ─────────────────────────────
if page == "Latest Report":
    st.title("Latest Analysis")

    if "latest_analysis" in st.session_state and st.session_state["latest_analysis"]:
        report = st.session_state["latest_analysis"]
    elif load_latest_report():
        report = load_latest_report()
    else:
        st.warning("No reports found. Go to Run Analysis to generate one.")
        st.stop()

    date_str = report.get("analysis_date", "Unknown date")
    st.caption(f"Analysis date: {date_str}   |   {report.get('disclaimer', '')}")
    st.divider()

    if summary := report.get("macro_environment"):
        st.info(f"🌍 **Macro Environment:** {summary}")

    if summary := report.get("market_summary"):
        st.success(f"**Market Overview:** {summary}")

    st.subheader("Recommendations")
    recs = report.get("recommendations", [])
    cols = st.columns(len(recs)) if recs else []

    for col, rec in zip(cols, recs):
        action = rec["recommendation"]
        color  = rec_color(action)
        icon   = rec_icon(action)
        with col:
            st.markdown(
                f"""<div style="border:1px solid {color};border-radius:12px;padding:16px;text-align:center">
                    <div style="font-size:2rem;color:{color}">{icon}</div>
                    <div style="font-size:1.4rem;font-weight:600">{rec['ticker']}</div>
                    <div style="font-size:1.1rem;color:{color};font-weight:500">{action}</div>
                    <div style="color:#888;font-size:0.9rem">Confidence: {rec['confidence']}/10</div>
                    <div style="color:#888;font-size:0.85rem">{rec.get('target_timeframe','')}</div>
                </div>""",
                unsafe_allow_html=True,
            )

    st.divider()
    for rec in recs:
        action = rec["recommendation"]
        with st.expander(f"{rec_icon(action)}  {rec['ticker']} — {action}  (Confidence {rec['confidence']}/10)", expanded=True):
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**Technical**")
                st.write(rec["key_signals"].get("technical", "N/A"))
                st.markdown("**Fundamental**")
                st.write(rec["key_signals"].get("fundamental", "N/A"))
                if macro := rec["key_signals"].get("macro_impact"):
                    st.markdown("**Macro Impact**")
                    st.write(macro)
            with c2:
                st.markdown("**Insider Signal**")
                st.write(rec["key_signals"].get("insider_signal", "N/A"))
                st.markdown("**Reasoning**")
                st.write(rec.get("reasoning", ""))
                st.markdown("**Risk Factors**")
                st.write(rec.get("risk_factors", ""))

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

    tickers  = sorted(df["ticker"].unique())
    selected = st.multiselect("Filter by ticker", tickers, default=tickers)
    df       = df[df["ticker"].isin(selected)]

    st.subheader("Confidence Score Over Time")
    fig = px.line(df, x="datetime", y="confidence", color="ticker", markers=True)
    fig.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", height=350)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Full History")
    display_df = df[["datetime", "ticker", "recommendation", "confidence"]].sort_values("datetime", ascending=False)
    st.dataframe(display_df, use_container_width=True)
    st.download_button("Download CSV", display_df.to_csv(index=False), "history.csv", "text/csv")


# ── Run Analysis ──────────────────────────────
elif page == "Run Analysis":
    st.title("Run Analysis Now")
    st.caption("⚠️ This will use your Anthropic API credits.")

    # Show saved watchlist as default
    watchlist_str = ", ".join(st.session_state["watchlist"])
    tickers_input = st.text_input("Watchlist (comma separated)", value=watchlist_str)

    col1, col2 = st.columns([1, 3])
    with col1:
        save_watchlist = st.checkbox("Save as my watchlist", value=True)

    if st.button("▶  Run Analysis", type="primary"):
        tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]
        if not tickers:
            st.error("Please enter at least one ticker.")
        else:
            if save_watchlist:
                st.session_state["watchlist"] = tickers
                st.sidebar.success(f"Watchlist saved: {', '.join(tickers)}")

            with st.spinner(f"Analyzing {', '.join(tickers)}... this may take 2-3 minutes"):
                try:
                    from analysis_agent import run_pipeline
                    analysis = run_pipeline(tickers, save_json=False)

                    if "error" in analysis:
                        st.error(f"Analysis failed: {analysis['error']}")
                    else:
                        st.session_state["latest_analysis"] = analysis
                        st.success("✅ Analysis complete!")

                        # Check and send alerts
                        alert_on = []
                        if st.session_state["alert_on_buy"]:
                            alert_on.append("BUY")
                        if st.session_state["alert_on_sell"]:
                            alert_on.append("SELL")

                        if alert_on and os.getenv("ALERT_EMAIL"):
                            try:
                                from email_alerts import check_and_send_alerts
                                triggered = check_and_send_alerts(
                                    analysis,
                                    alert_on=alert_on,
                                    min_confidence=st.session_state["min_confidence"],
                                )
                                if triggered:
                                    tickers_alerted = ", ".join(t["ticker"] for t in triggered)
                                    st.info(f"📧 Alert email sent for: {tickers_alerted}")
                            except Exception as e:
                                st.warning(f"Alert email failed: {e}")

                        st.json(analysis)

                except Exception as e:
                    st.error(f"Error: {e}")
                    import traceback
                    st.code(traceback.format_exc())


# ── Backtest ──────────────────────────────────
elif page == "Backtest":
    st.title("Strategy Backtesting")
    st.caption("⚠️ Past performance does not guarantee future results. Educational use only.")

    col1, col2, col3 = st.columns(3)
    with col1:
        bt_tickers = st.text_input("Tickers", value=", ".join(st.session_state["watchlist"]))
    with col2:
        bt_period = st.selectbox(
            "Period",
            options=["6mo", "1y", "2y"],
            index=1,
            format_func=lambda x: {"6mo": "6 months", "1y": "1 year", "2y": "2 years"}[x]
        )
    with col3:
        bt_cash = st.number_input("Starting cash ($)", value=10000, step=1000, min_value=1000)

    if st.button("▶  Run Backtest", type="primary"):
        tickers = [t.strip().upper() for t in bt_tickers.split(",") if t.strip()]
        if not tickers:
            st.error("Please enter at least one ticker.")
        else:
            with st.spinner(f"Backtesting {', '.join(tickers)} over {bt_period}..."):
                try:
                    from backtesting import backtest_watchlist, backtest_summary

                    results = backtest_watchlist(tickers, period=bt_period, starting_cash=float(bt_cash))
                    summary = backtest_summary(results)

                    if "error" in summary:
                        st.error(summary["error"])
                    else:
                        st.session_state["backtest_results"] = results
                        st.session_state["backtest_summary"] = summary

                except Exception as e:
                    st.error(f"Backtest failed: {e}")
                    import traceback
                    st.code(traceback.format_exc())

    # Display results if available
    if "backtest_results" in st.session_state:
        results = st.session_state["backtest_results"]
        summary = st.session_state["backtest_summary"]

        # Summary metrics
        st.divider()
        st.subheader("Summary")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Avg Strategy Return", f"{summary['avg_strategy_return']:+.1f}%")
        m2.metric("Avg Buy & Hold Return", f"{summary['avg_bah_return']:+.1f}%")
        m3.metric("Avg Alpha", f"{summary['avg_alpha']:+.1f}%")
        m4.metric("Beat Buy & Hold", f"{summary['outperformed_count']}/{summary['tickers_tested']}")

        # Per stock results table
        st.divider()
        st.subheader("Results by Stock")
        table_data = []
        for ticker, r in results.items():
            if "error" not in r:
                table_data.append({
                    "Ticker":           ticker,
                    "Strategy Return":  f"{r['total_return_pct']:+.1f}%",
                    "Buy & Hold":       f"{r['buy_and_hold_return']:+.1f}%",
                    "Alpha":            f"{r['alpha']:+.1f}%",
                    "Trades":           r["num_trades"],
                    "Win Rate":         f"{r['win_rate']}%",
                    "Max Drawdown":     f"{r['max_drawdown']}%",
                    "Sharpe Ratio":     r["sharpe_ratio"],
                    "Beat B&H":         "✓" if r["outperformed"] else "✗",
                })
        if table_data:
            st.dataframe(pd.DataFrame(table_data), use_container_width=True)

        # Portfolio value chart
        st.divider()
        st.subheader("Portfolio Value Over Time")
        fig = go.Figure()
        for ticker, r in results.items():
            if "error" not in r and r.get("portfolio_history"):
                hist = pd.DataFrame(r["portfolio_history"])
                hist["date"] = pd.to_datetime(hist["date"])
                fig.add_trace(go.Scatter(
                    x=hist["date"],
                    y=hist["portfolio_value"],
                    name=f"{ticker} Strategy",
                    mode="lines",
                ))
        fig.update_layout(
            yaxis_title="Portfolio Value ($)",
            xaxis_title="Date",
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            height=400,
            hovermode="x unified",
        )
        st.plotly_chart(fig, use_container_width=True)

        # Trade history
        st.divider()
        st.subheader("Trade History")
        ticker_select = st.selectbox("Select ticker", list(results.keys()))
        if ticker_select and "error" not in results[ticker_select]:
            trades_df = pd.DataFrame(results[ticker_select]["trades"])
            if not trades_df.empty:
                trades_df["date"] = pd.to_datetime(trades_df["date"]).dt.strftime("%Y-%m-%d")
                st.dataframe(trades_df, use_container_width=True)
                csv = trades_df.to_csv(index=False)
                st.download_button(f"Download {ticker_select} trades CSV", csv, f"{ticker_select}_trades.csv", "text/csv")
            else:
                st.info("No trades were executed for this ticker.")


# ── Watchlist & Alerts ────────────────────────
elif page == "Watchlist & Alerts":
    st.title("Watchlist & Alerts")

    # ── Saved Watchlist ───────────────────────
    st.subheader("My Watchlist")
    st.caption("This watchlist is pre-filled every time you run an analysis.")

    watchlist_str = ", ".join(st.session_state["watchlist"])
    new_watchlist = st.text_input("Tickers (comma separated)", value=watchlist_str)

    if st.button("💾  Save Watchlist"):
        tickers = [t.strip().upper() for t in new_watchlist.split(",") if t.strip()]
        if tickers:
            st.session_state["watchlist"] = tickers
            st.success(f"Watchlist saved: {', '.join(tickers)}")
        else:
            st.error("Please enter at least one ticker.")

    st.divider()

    # ── Email Alerts ──────────────────────────
    st.subheader("Email Alerts")
    st.caption("Get notified by email when BUY or SELL signals are detected.")

    email_configured = bool(os.getenv("ALERT_EMAIL") and os.getenv("ALERT_EMAIL_PASSWORD"))

    if not email_configured:
        st.warning("""
Email alerts are not configured yet. Add these to your Streamlit secrets:

```toml
ALERT_EMAIL = "your.gmail@gmail.com"
ALERT_EMAIL_PASSWORD = "your-16-char-app-password"
ALERT_TO_EMAIL = "destination@email.com"
```

Get a Gmail App Password at: myaccount.google.com → Security → App Passwords
        """)
    else:
        st.success(f"✓ Email alerts configured — sending to {os.getenv('ALERT_TO_EMAIL')}")

    col1, col2 = st.columns(2)
    with col1:
        st.session_state["alert_on_buy"] = st.checkbox(
            "Alert on BUY signals", value=st.session_state["alert_on_buy"]
        )
        st.session_state["alert_on_sell"] = st.checkbox(
            "Alert on SELL signals", value=st.session_state["alert_on_sell"]
        )
    with col2:
        st.session_state["min_confidence"] = st.slider(
            "Minimum confidence to alert",
            min_value=1, max_value=10,
            value=st.session_state["min_confidence"],
            help="Only send alerts for signals above this confidence score"
        )
        st.caption(f"Alerts will trigger on signals with confidence ≥ {st.session_state['min_confidence']}/10")

    if email_configured:
        st.divider()
        if st.button("📧  Send Test Email"):
            try:
                from email_alerts import send_alert_email
                test_analysis = {
                    "analysis_date": datetime.utcnow().strftime("%Y-%m-%d"),
                    "macro_environment": "This is a test alert from your Stock Analysis Bot.",
                    "market_summary": "Test alert — ignore this.",
                }
                test_triggered = [{
                    "ticker": "TEST",
                    "recommendation": "BUY",
                    "confidence": 9,
                    "reasoning": "This is a test alert to verify your email configuration is working correctly.",
                }]
                sent = send_alert_email(test_triggered, test_analysis)
                if sent:
                    st.success("✅ Test email sent! Check your inbox.")
                else:
                    st.error("Failed to send test email — check your credentials.")
            except Exception as e:
                st.error(f"Error: {e}")
