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

page = st.sidebar.radio("View", ["Portfolio", "Latest Report", "History", "Run Analysis", "Screener", "Backtest", "Predictions", "Watchlist & Alerts"])


# ── Latest Report ─────────────────────────────
# ── Portfolio Tracker ─────────────────────────
if page == "Portfolio":
    st.title("Portfolio Tracker")
    st.caption("⚠️ For educational purposes only. Not financial advice.")

    from portfolio_tracker import (
        load_portfolio, save_portfolio, add_holding,
        remove_holding, calculate_portfolio_value,
        compare_with_recommendations
    )

    tab1, tab2, tab3 = st.tabs(["My Holdings", "Add / Remove", "vs Recommendations"])

    # ── Tab 1: Holdings overview ──────────────
    with tab1:
        st.subheader("Current Holdings")

        with st.spinner("Fetching current prices..."):
            result = calculate_portfolio_value()

        holdings = result["holdings"]
        if not holdings:
            st.info("No holdings yet. Go to the Add / Remove tab to add your stocks.")
        else:
            # Summary metrics
            gain_color = "normal" if result["total_gain_loss"] >= 0 else "inverse"
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Total Invested",   f"${result['total_cost']:,.2f}")
            m2.metric("Current Value",    f"${result['total_value']:,.2f}")
            m3.metric("Total P&L",        f"${result['total_gain_loss']:+,.2f}",
                      f"{result['total_return_pct']:+.2f}%",
                      delta_color=gain_color)
            m4.metric("Positions",        len(holdings))

            st.caption(f"Last updated: {result['as_of']}")
            st.divider()

            # Holdings table
            table_data = []
            for h in holdings:
                table_data.append({
                    "Ticker":         h["ticker"],
                    "Shares":         h["shares"],
                    "Buy Price":      f"${h['buy_price']:.2f}",
                    "Current Price":  f"${h['current_price']:.2f}",
                    "Cost Basis":     f"${h['cost']:,.2f}",
                    "Market Value":   f"${h['value']:,.2f}",
                    "P&L ($)":        f"${h['gain_loss']:+,.2f}",
                    "P&L (%)":        f"{h['return_pct']:+.2f}%",
                    "Days Held":      h["days_held"],
                    "Buy Date":       h["buy_date"],
                })
            st.dataframe(pd.DataFrame(table_data), use_container_width=True)

            # P&L chart
            st.divider()
            st.subheader("P&L by Position")
            colors = ["#22c55e" if h["gain_loss"] >= 0 else "#ef4444" for h in holdings]
            fig = go.Figure(go.Bar(
                x=[h["ticker"] for h in holdings],
                y=[h["return_pct"] for h in holdings],
                marker_color=colors,
                text=[f"{h['return_pct']:+.1f}%" for h in holdings],
                textposition="outside",
            ))
            fig.add_hline(y=0, line_dash="dash", line_color="gray", line_width=1)
            fig.update_layout(
                yaxis_title="Return %",
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                height=350,
            )
            st.plotly_chart(fig, use_container_width=True)

            # Portfolio allocation pie chart
            st.divider()
            st.subheader("Portfolio Allocation")
            fig2 = go.Figure(go.Pie(
                labels=[h["ticker"] for h in holdings],
                values=[h["value"] for h in holdings],
                hole=0.4,
                textinfo="label+percent",
            ))
            fig2.update_layout(
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                height=350,
                showlegend=False,
            )
            st.plotly_chart(fig2, use_container_width=True)

            # Download
            csv = pd.DataFrame(table_data).to_csv(index=False)
            st.download_button("Download Holdings CSV", csv, "portfolio.csv", "text/csv")

    # ── Tab 2: Add / Remove ───────────────────
    with tab2:
        st.subheader("Add a Holding")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            new_ticker = st.text_input("Ticker", placeholder="AAPL").upper()
        with col2:
            new_shares = st.number_input("Shares", min_value=0.01, value=1.0, step=0.01)
        with col3:
            new_buy_price = st.number_input("Buy price ($)", min_value=0.01, value=100.0, step=0.01)
        with col4:
            new_buy_date = st.date_input("Buy date", value=datetime.utcnow().date())

        if st.button("➕ Add Holding", type="primary"):
            if new_ticker:
                add_holding(new_ticker, new_shares, new_buy_price, str(new_buy_date))
                st.success(f"Added {new_shares} shares of {new_ticker} @ ${new_buy_price:.2f}")
                st.rerun()
            else:
                st.error("Please enter a ticker symbol.")

        st.divider()
        st.subheader("Remove a Holding")

        portfolio = load_portfolio()
        holdings  = portfolio.get("holdings", [])

        if not holdings:
            st.info("No holdings to remove.")
        else:
            remove_ticker = st.selectbox(
                "Select holding to remove",
                [h["ticker"] for h in holdings]
            )
            if st.button("🗑️ Remove Holding", type="secondary"):
                remove_holding(remove_ticker)
                st.success(f"Removed {remove_ticker} from portfolio.")
                st.rerun()

        st.divider()
        st.subheader("Current Portfolio")
        if holdings:
            for h in holdings:
                st.write(f"**{h['ticker']}** — {h['shares']} shares @ ${h['buy_price']:.2f} (bought {h['buy_date']})")
        else:
            st.info("No holdings yet.")

        st.divider()
        st.subheader("Backup & Restore")
        st.caption("On Streamlit Cloud your portfolio resets when the app restarts. Export regularly to back it up.")

        col_exp, col_imp = st.columns(2)
        with col_exp:
            from portfolio_tracker import export_portfolio_json
            json_export = export_portfolio_json()
            st.download_button(
                "💾  Export Portfolio JSON",
                json_export,
                "portfolio_backup.json",
                "application/json",
                help="Download your portfolio as a JSON file. Keep this safe and use it to restore your holdings."
            )
        with col_imp:
            uploaded = st.file_uploader("📂  Import Portfolio JSON", type="json",
                help="Upload a previously exported portfolio JSON file to restore your holdings.")
            if uploaded:
                try:
                    from portfolio_tracker import import_portfolio_json
                    json_str  = uploaded.read().decode("utf-8")
                    imported  = import_portfolio_json(json_str)
                    n         = len(imported.get("holdings", []))
                    st.success(f"Imported {n} holdings successfully!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Import failed: {e}")

    # ── Tab 3: vs Recommendations ─────────────
    with tab3:
        st.subheader("Holdings vs Bot Recommendations")
        st.caption("Compares your positions against the latest Claude analysis.")

        analysis = st.session_state.get("latest_analysis")
        if not analysis:
            st.warning("No analysis available. Go to Run Analysis first, then come back here.")
        else:
            result      = calculate_portfolio_value()
            comparisons = compare_with_recommendations(result, analysis)

            if not comparisons:
                st.info("None of your holdings appear in the latest analysis. Run analysis on your portfolio stocks first.")
            else:
                for c in comparisons:
                    priority_color = {"HIGH": "#ef4444", "MEDIUM": "#f59e0b", "LOW": "#22c55e"}[c["priority"]]
                    action_icon    = {"BUY": "▲", "HOLD": "●", "SELL": "▼"}.get(c["action"], "●")
                    action_color   = {"BUY": "#22c55e", "HOLD": "#f59e0b", "SELL": "#ef4444"}.get(c["action"], "#888")

                    st.markdown(
                        f"""<div style="border-left:4px solid {priority_color};padding:12px 16px;
                            background:var(--color-background-secondary);border-radius:0 8px 8px 0;margin-bottom:12px">
                            <div style="display:flex;justify-content:space-between;align-items:center">
                                <div>
                                    <strong style="font-size:1.1rem">{c['ticker']}</strong>
                                    &nbsp;&nbsp;
                                    <span style="color:{action_color};font-weight:600">{action_icon} {c['action']}</span>
                                    &nbsp;(confidence {c['confidence']}/10)
                                </div>
                                <div style="text-align:right">
                                    <span style="color:{'#22c55e' if c['return_pct'] >= 0 else '#ef4444'};font-weight:600">
                                        {c['return_pct']:+.2f}% (${c['gain_loss']:+,.2f})
                                    </span>
                                </div>
                            </div>
                            <div style="color:var(--color-text-secondary);font-size:0.9rem;margin-top:4px">
                                {c['alignment']}
                            </div>
                        </div>""",
                        unsafe_allow_html=True,
                    )


elif page == "Latest Report":
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

    # Use portfolio tickers if available, otherwise use saved watchlist
    from portfolio_tracker import load_portfolio
    portfolio_holdings = load_portfolio().get("holdings", [])
    portfolio_tickers  = [h["ticker"] for h in portfolio_holdings]
    default_tickers    = portfolio_tickers if portfolio_tickers else st.session_state["watchlist"]
    watchlist_str      = ", ".join(default_tickers)

    tickers_input = st.text_input(
        "Watchlist (comma separated)",
        value=watchlist_str,
        help="Pre-filled from your portfolio holdings. Edit as needed."
    )

    col1, col2, col3 = st.columns([2, 2, 2])
    with col1:
        save_watchlist = st.checkbox("Save as my watchlist", value=True)
    with col2:
        period = st.selectbox(
            "Date range",
            options=["1mo", "3mo", "6mo", "1y", "ytd"],
            index=2,
            format_func=lambda x: {
                "1mo": "1 month", "3mo": "3 months",
                "6mo": "6 months", "1y": "1 year", "ytd": "Year to date",
            }[x]
        )
    with col3:
        if portfolio_tickers:
            if st.button("📋 Use my portfolio tickers"):
                st.session_state["watchlist"] = portfolio_tickers
                st.rerun()

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
                    analysis = run_pipeline(tickers, save_json=False, period=period)

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


# ── Screener ──────────────────────────────────
elif page == "Screener":
    st.title("Stock Screener")
    st.caption("⚠️ Screening results are not financial advice. Always do your own research.")

    st.info("The screener scans hundreds of stocks and surfaces the ones with the strongest technical BUY signals — no need to know the ticker first.")

    st.markdown("""
    **How the screener works:** It scans every stock in the universe and scores them on technical signals.
    Lower RSI = more oversold (potential bounce). Higher score = stronger combined signal.
    The top candidates are ranked by score for you to investigate further.
    """)
    st.caption("💡 **Tip:** Start with RSI 55, SMA20 unchecked, and Dow Jones 30 to see how the screener works. Then tighten filters once you understand the results.")
    st.divider()

    col1, col2, col3 = st.columns(3)
    with col1:
        universe = st.selectbox(
            "Universe",
            options=["Dow Jones 30", "NASDAQ 100", "S&P 500 (100 stocks)"],
            index=0,
            help="Dow Jones 30 is fastest (30 stocks). S&P 500 scans 100 stocks and takes 2-3 minutes."
        )
    with col2:
        max_rsi = st.slider(
            "Max RSI",
            min_value=25, max_value=70, value=55,
            help="RSI measures momentum 0-100. Below 30 = very oversold. Below 45 = moderately oversold. Above 70 = overbought. Raise this to get more results."
        )
    with col3:
        max_results = st.slider(
            "Max results",
            min_value=5, max_value=30, value=15,
            help="How many stocks to show in the results. The screener ranks all passing stocks by score and shows the top N. 15 is a good starting point."
        )

    col4, col5 = st.columns(2)
    with col4:
        require_above_sma20 = st.checkbox(
            "Must be above SMA20 (uptrend)",
            value=False,
            help="SMA20 is the 20-day average price. Being above it suggests an uptrend. WARNING: This conflicts with low RSI — oversold stocks are often below their SMA20. Uncheck this to get more results."
        )
        require_macd_bullish = st.checkbox(
            "Must have bullish MACD",
            value=False,
            help="MACD bullish means the fast momentum line is above the slow signal line — suggests upward momentum. This is an additional filter that reduces results significantly. Leave unchecked for a wider search."
        )
    with col5:
        min_price = st.number_input(
            "Min price ($)",
            value=5.0, step=1.0,
            help="Filters out very cheap penny stocks which can be volatile and illiquid. $5 is a sensible minimum."
        )
        min_volume_ratio = st.slider(
            "Min volume ratio",
            min_value=0.1, max_value=2.0, value=0.3, step=0.1,
            help="Compares today's volume to the 20-day average. 1.0 = average volume. 0.3 means at least 30% of normal volume. Lower this to get more results."
        )

    if st.button("🔍  Run Screen", type="primary"):
        criteria = {
            "max_rsi":              max_rsi,
            "min_rsi":              20,
            "require_above_sma20":  require_above_sma20,
            "require_macd_bullish": require_macd_bullish,
            "min_volume_ratio":     min_volume_ratio,
            "min_price":            min_price,
        }

        from screener import UNIVERSES
        total = len(UNIVERSES.get(universe, []))
        progress_bar = st.progress(0, text=f"Scanning {total} stocks...")
        status_text  = st.empty()

        def update_progress(current, total, ticker):
            pct = current / total
            progress_bar.progress(pct, text=f"Scanning {current}/{total} — {ticker}")
            status_text.caption(f"Checking {ticker}...")

        try:
            from screener import run_screener
            screen_results = run_screener(
                universe=universe,
                criteria=criteria,
                max_results=max_results,
                progress_callback=update_progress,
            )
            progress_bar.progress(1.0, text="Scan complete!")
            status_text.empty()
            st.session_state["screen_results"] = screen_results
        except Exception as e:
            st.error(f"Screener failed: {e}")
            import traceback
            st.code(traceback.format_exc())

    # Display results
    if "screen_results" in st.session_state:
        r = st.session_state["screen_results"]
        st.divider()

        # Summary metrics
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Stocks Scanned",   r["screened"])
        m2.metric("Passed Filters",   r["passed"])
        m3.metric("Filtered Out",     r["failed"])
        m4.metric("Scan Duration",    f"{r['duration']}s")
        st.caption(f"Universe: {r['universe']}   |   {r['timestamp']}")

        results = r["results"]
        if not results:
            st.warning("No stocks passed the screening criteria. Try relaxing the filters — increase Max RSI or uncheck SMA20 requirement.")
        else:
            st.subheader(f"Top {len(results)} Candidates")

            # Results table
            table_data = []
            for stock in results:
                table_data.append({
                    "Ticker":       stock["ticker"],
                    "Price":        f"${stock['price']}",
                    "RSI":          stock["rsi"],
                    "Score":        stock["score"],
                    "1D Return":    f"{stock['return_1d']:+.2f}%",
                    "5D Return":    f"{stock['return_5d']:+.2f}%",
                    "Vol Ratio":    f"{stock['volume_ratio']}x",
                    "Above SMA20":  "✓" if stock["above_sma20"] else "✗",
                    "Above SMA50":  "✓" if stock["above_sma50"] else "✗",
                    "MACD Bullish": "✓" if stock["macd_bullish"] else "✗",
                })
            st.dataframe(pd.DataFrame(table_data), use_container_width=True)

            # Score chart
            st.divider()
            st.subheader("Signal Strength")
            tickers_list = [s["ticker"] for s in results]
            scores_list  = [s["score"] for s in results]
            rsi_list     = [s["rsi"] for s in results]

            fig = go.Figure()
            fig.add_trace(go.Bar(
                name="Score", x=tickers_list, y=scores_list,
                marker_color="#2E75B6", yaxis="y"
            ))
            fig.add_trace(go.Scatter(
                name="RSI", x=tickers_list, y=rsi_list,
                mode="lines+markers", marker_color="#ef4444",
                yaxis="y2"
            ))
            fig.update_layout(
                yaxis=dict(title="Score", side="left"),
                yaxis2=dict(title="RSI", side="right", overlaying="y", range=[0, 80]),
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                height=380,
                hovermode="x unified",
                legend=dict(orientation="h", yanchor="bottom", y=1.02),
            )
            st.plotly_chart(fig, use_container_width=True)

            # Add to watchlist button
            st.divider()
            st.subheader("Add to Watchlist")
            top5 = [s["ticker"] for s in results[:5]]
            st.caption(f"Top 5 candidates: {', '.join(top5)}")

            col_a, col_b = st.columns(2)
            with col_a:
                if st.button("➕ Add top 5 to watchlist"):
                    current = st.session_state["watchlist"]
                    combined = list(dict.fromkeys(current + top5))
                    st.session_state["watchlist"] = combined
                    st.success(f"Added to watchlist: {', '.join(top5)}")
            with col_b:
                custom_add = st.text_input("Or add specific tickers from results", placeholder="e.g. AAPL, MSFT")
                if st.button("➕ Add these"):
                    to_add = [t.strip().upper() for t in custom_add.split(",") if t.strip()]
                    if to_add:
                        current  = st.session_state["watchlist"]
                        combined = list(dict.fromkeys(current + to_add))
                        st.session_state["watchlist"] = combined
                        st.success(f"Added: {', '.join(to_add)}")

            # Download
            csv = pd.DataFrame(table_data).to_csv(index=False)
            st.download_button("Download Results CSV", csv, "screen_results.csv", "text/csv")


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



# ── Predictions ───────────────────────────────
elif page == "Predictions":
    st.title("Price Direction Predictions")
    st.caption("⚠️ ML predictions are not guaranteed to be accurate. Educational use only. Not financial advice.")

    col1, col2, col3 = st.columns(3)
    with col1:
        pred_tickers = st.text_input("Tickers", value=", ".join(st.session_state["watchlist"]))
    with col2:
        pred_period = st.selectbox(
            "Training period",
            options=["1y", "2y", "3y"],
            index=0,
            format_func=lambda x: {"1y": "1 year", "2y": "2 years", "3y": "3 years"}[x],
            help="More data = better model accuracy. Minimum 1 year recommended."
        )
    with col3:
        model_choice = st.selectbox(
            "Model",
            options=["Compare Both", "Linear Regression", "Random Forest"],
            index=0,
            help="Compare Both runs LR and RF and shows where they agree"
        )

    st.info("Accuracy above 55% beats random guessing. When both models agree the signal is stronger.")

    if st.button("▶  Run Predictions", type="primary"):
        tickers = [t.strip().upper() for t in pred_tickers.split(",") if t.strip()]
        if not tickers:
            st.error("Please enter at least one ticker.")
        else:
            with st.spinner(f"Training models for {', '.join(tickers)}... this may take a minute"):
                try:
                    if model_choice == "Linear Regression":
                        from prediction_lr import run_predictions_watchlist
                        results = run_predictions_watchlist(tickers, period=pred_period)
                        st.session_state["prediction_results"] = results
                        st.session_state["prediction_mode"]    = "lr"
                    elif model_choice == "Random Forest":
                        from prediction_rf import run_predictions_rf_watchlist
                        results = run_predictions_rf_watchlist(tickers, period=pred_period)
                        st.session_state["prediction_results"] = results
                        st.session_state["prediction_mode"]    = "rf"
                    else:
                        from prediction_rf import compare_models_watchlist
                        results = compare_models_watchlist(tickers, period=pred_period)
                        st.session_state["prediction_results"] = results
                        st.session_state["prediction_mode"]    = "compare"
                    st.success("✅ Predictions ready!")
                except Exception as e:
                    st.error(f"Prediction failed: {e}")
                    import traceback
                    st.code(traceback.format_exc())

    if "prediction_results" in st.session_state:
        results = st.session_state["prediction_results"]
        mode    = st.session_state.get("prediction_mode", "lr")
        st.divider()

        valid = {k: v for k, v in results.items() if "error" not in v}

        # Market regime banner
        if valid:
            first  = next(iter(valid.values()))
            regime = first.get("regime", {})
            if regime:
                regime_name = regime.get("regime", "Unknown")
                color       = regime.get("color", "#6b7280")
                vix         = regime.get("vix", "N/A")
                fg          = regime.get("fear_greed", "N/A")
                warning     = regime.get("warning", "")
                adj         = int(regime.get("confidence_adj", 0) * 100)
                st.markdown(
                    f"""<div style="border-left:4px solid {color};padding:12px 16px;background:{'#fef2f2' if adj >= 20 else '#fff7ed' if adj >= 10 else '#f0fdf4'};border-radius:0 8px 8px 0;margin-bottom:16px">
                        <strong style="color:{color}">Market Regime: {regime_name}</strong>
                        {"&nbsp;&nbsp;|&nbsp;&nbsp;VIX: " + str(vix) if vix else ""}
                        {"&nbsp;&nbsp;|&nbsp;&nbsp;Fear & Greed: " + str(fg) if fg else ""}
                        {f"&nbsp;&nbsp;|&nbsp;&nbsp;Confidence reduced by {adj}%" if adj > 0 else ""}
                        <br><span style="font-size:0.85rem;color:#666">{warning}</span>
                    </div>""",
                    unsafe_allow_html=True,
                )

        if mode == "compare":
            st.subheader("Model Comparison — LR vs Random Forest")
            st.caption("Green border = models agree. Grey = models disagree (HOLD recommended).")
            cols = st.columns(len(valid)) if valid else []
            for col, (ticker, r) in zip(cols, valid.items()):
                agree      = r["models_agree"]
                border_col = "#22c55e" if agree else "#9ca3af"
                rf_color   = "#22c55e" if r["rf_direction"] == "UP" else "#ef4444"
                lr_color   = "#22c55e" if r["lr_direction"] == "UP" else "#ef4444"
                with col:
                    st.markdown(
                        f"""<div style="border:2px solid {border_col};border-radius:12px;padding:16px;text-align:center">
                            <div style="font-size:1.4rem;font-weight:600">{ticker}</div>
                            <div style="font-size:0.85rem;color:#888">${r['current_price']} ({r['price_change_pct']:+.2f}%)</div>
                            <hr style="margin:8px 0;border-color:#eee">
                            <div style="display:flex;justify-content:space-around;margin:8px 0">
                                <div>
                                    <div style="font-size:0.75rem;color:#888">Linear Reg</div>
                                    <div style="color:{lr_color};font-weight:600">{r['lr_direction']}</div>
                                    <div style="font-size:0.8rem;color:#888">{r['lr_confidence']:.0f}%</div>
                                    <div style="font-size:0.75rem;color:#888">{r['lr_accuracy']}% acc</div>
                                </div>
                                <div>
                                    <div style="font-size:0.75rem;color:#888">Random Forest</div>
                                    <div style="color:{rf_color};font-weight:600">{r['rf_direction']}</div>
                                    <div style="font-size:0.8rem;color:#888">{r['rf_confidence']:.0f}%</div>
                                    <div style="font-size:0.75rem;color:#888">{r['rf_accuracy']}% acc</div>
                                </div>
                            </div>
                            <hr style="margin:8px 0;border-color:#eee">
                            <div style="font-weight:600;color:{'#22c55e' if agree else '#6b7280'}">
                                {'✓ AGREE' if agree else '✗ DISAGREE'}
                            </div>
                            <div style="font-size:0.9rem">Combined: {r['combined_signal']}</div>
                            <div style="font-size:0.8rem;color:#888">{r['combined_confidence']:.0f}% confidence</div>
                        </div>""",
                        unsafe_allow_html=True,
                    )

            st.divider()
            st.subheader("Comparison Summary")
            table = []
            for ticker, r in valid.items():
                table.append({
                    "Ticker":          ticker,
                    "LR":              f"{r['lr_direction']} {r['lr_confidence']:.0f}%",
                    "LR Accuracy":     f"{r['lr_accuracy']}%",
                    "RF":              f"{r['rf_direction']} {r['rf_confidence']:.0f}%",
                    "RF Accuracy":     f"{r['rf_accuracy']}%",
                    "Agree":           "✓" if r["models_agree"] else "✗",
                    "Combined Signal": r["combined_signal"],
                    "Confidence":      f"{r['combined_confidence']:.0f}%",
                })
            st.dataframe(pd.DataFrame(table), use_container_width=True)

            st.divider()
            st.subheader("Random Forest Feature Importance")
            ticker_sel = st.selectbox("Select ticker", list(valid.keys()), key="rf_feat_ticker")
            if ticker_sel and "top_features" in valid[ticker_sel]:
                imp = valid[ticker_sel]["top_features"]
                fig = go.Figure(go.Bar(
                    x=list(imp.values()), y=list(imp.keys()),
                    orientation="h", marker_color="#2E75B6",
                ))
                fig.update_layout(
                    xaxis_title="Importance", plot_bgcolor="rgba(0,0,0,0)",
                    paper_bgcolor="rgba(0,0,0,0)", height=380,
                    yaxis=dict(autorange="reversed"),
                )
                st.plotly_chart(fig, use_container_width=True)

        else:
            model_label = "Random Forest" if mode == "rf" else "Linear Regression"
            st.subheader(f"Tomorrow's Predictions — {model_label}")
            cols = st.columns(len(valid)) if valid else []
            for col, (ticker, r) in zip(cols, valid.items()):
                pred      = r["prediction"]
                color     = "#22c55e" if pred["direction"] == "UP" else "#ef4444"
                icon      = "▲" if pred["direction"] == "UP" else "▼"
                adj_label = f"<div style='color:#f59e0b;font-size:0.75rem'>adj for {r['regime']['regime']}</div>" if pred.get("regime_adjusted") else ""
                with col:
                    st.markdown(
                        f"""<div style="border:1px solid {color};border-radius:12px;padding:16px;text-align:center">
                            <div style="font-size:2rem;color:{color}">{icon}</div>
                            <div style="font-size:1.4rem;font-weight:600">{ticker}</div>
                            <div style="font-size:1.1rem;color:{color};font-weight:500">{pred['direction']}</div>
                            <div style="color:#888;font-size:0.9rem">Signal: {pred['signal']}</div>
                            <div style="color:#888;font-size:0.85rem">{pred['confidence']:.0f}% confidence</div>
                            {adj_label}
                            <div style="color:#888;font-size:0.8rem">Model: {r['model_accuracy']}% accurate</div>
                        </div>""",
                        unsafe_allow_html=True,
                    )

            st.divider()
            st.subheader("Model Performance")
            perf_data = []
            for ticker, r in valid.items():
                pred           = r["prediction"]
                raw_conf       = pred.get("raw_confidence", pred["confidence"])
                beats_baseline = r["model_accuracy"] > r["baseline_accuracy"]
                perf_data.append({
                    "Ticker":           ticker,
                    "Current Price":    f"${r['current_price']}",
                    "Price Change":     f"{r['price_change_pct']:+.2f}%",
                    "Prediction":       pred["direction"],
                    "Signal":           pred["signal"],
                    "Confidence (adj)": f"{pred['confidence']:.0f}%",
                    "Confidence (raw)": f"{raw_conf:.0f}%",
                    "Model Accuracy":   f"{r['model_accuracy']}%",
                    "Baseline":         f"{r['baseline_accuracy']}%",
                    "Beats Baseline":   "✓" if beats_baseline else "✗",
                    "Regime":           r.get("regime", {}).get("regime", "N/A"),
                })
            if perf_data:
                st.dataframe(pd.DataFrame(perf_data), use_container_width=True)

            st.divider()
            st.subheader("Feature Importance")
            ticker_sel = st.selectbox("Select ticker", list(valid.keys()), key="feat_ticker")
            if ticker_sel:
                imp   = valid[ticker_sel]["feature_importance"]
                top10 = dict(list(imp.items())[:10])
                fig   = go.Figure(go.Bar(
                    x=list(top10.values()), y=list(top10.keys()),
                    orientation="h", marker_color="#2E75B6",
                ))
                fig.update_layout(
                    xaxis_title="Importance", plot_bgcolor="rgba(0,0,0,0)",
                    paper_bgcolor="rgba(0,0,0,0)", height=350,
                    yaxis=dict(autorange="reversed"),
                )
                st.plotly_chart(fig, use_container_width=True)

        errors = {k: v for k, v in results.items() if "error" in v}
        if errors:
            st.divider()
            for ticker, r in errors.items():
                st.warning(f"{ticker}: {r['error']}")




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
