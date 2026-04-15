"""
HOW TO ADD THE EARNINGS CALENDAR PAGE TO dashboard.py
======================================================
Add "Earnings Calendar" to your sidebar radio list, then add the
elif block below to your page routing.

1. Sidebar — add "Earnings Calendar" to the list:

    page = st.sidebar.radio("View", [
        "Portfolio", "Latest Report", "History", "Run Analysis",
        "Screener", "Backtest", "Predictions", "Watchlist & Alerts",
        "AI Chat", "Earnings Calendar"    # <-- add this
    ])

2. Routing — add this elif at the end of your if/elif chain:
"""

# ── Earnings Calendar page ────────────────────────────────────────────────
# Paste everything below this line into dashboard.py

elif page == "Earnings Calendar":
    st.title("📅 Earnings Calendar")
    st.caption("Flags stocks where upcoming earnings make predictions unreliable.")

    from earnings_calendar import get_earnings_context

    tickers = st.session_state.get("watchlist", ["AAPL", "MSFT", "NVDA"])

    col1, col2 = st.columns([3, 1])
    with col1:
        custom = st.text_input("Tickers", value=", ".join(tickers))
    with col2:
        st.write("")
        st.write("")
        run = st.button("🔍 Check Earnings", type="primary", use_container_width=True)

    if run or "earnings_results" not in st.session_state:
        tickers = [t.strip().upper() for t in custom.split(",") if t.strip()]
        with st.spinner("Fetching earnings dates..."):
            try:
                results = get_earnings_context(tickers)
                st.session_state["earnings_results"] = results
            except Exception as e:
                st.error(f"Failed to fetch earnings data: {e}")
                st.stop()

    if "earnings_results" in st.session_state:
        results = st.session_state["earnings_results"]

        # ── Risk summary cards ────────────────────────────────────────
        st.divider()
        high    = [t for t, r in results.items() if r["risk_level"] == "HIGH"]
        medium  = [t for t, r in results.items() if r["risk_level"] == "MEDIUM"]
        low     = [t for t, r in results.items() if r["risk_level"] == "LOW"]
        unknown = [t for t, r in results.items() if r["risk_level"] == "UNKNOWN"]

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("🔴 High Risk",    len(high),    help="Earnings within 2 days — avoid predictions")
        m2.metric("🟡 Medium Risk",  len(medium),  help="Earnings within 5 days — reduced confidence")
        m3.metric("🟢 Low Risk",     len(low),     help="Outside earnings window — predictions reliable")
        m4.metric("⚪ Unknown",      len(unknown), help="Could not fetch earnings date")

        if high:
            st.error(f"⚠️ **Avoid predictions for: {', '.join(high)}** — earnings imminent.")
        if medium:
            st.warning(f"⚠️ **Reduce confidence for: {', '.join(medium)}** — earnings approaching.")

        # ── Detailed table ────────────────────────────────────────────
        st.divider()
        st.subheader("Earnings Detail")

        risk_colors = {"HIGH": "🔴", "MEDIUM": "🟡", "LOW": "🟢", "UNKNOWN": "⚪"}

        table_data = []
        for ticker, r in results.items():
            table_data.append({
                "Ticker":           ticker,
                "Risk":             f"{risk_colors[r['risk_level']]} {r['risk_level']}",
                "Next Earnings":    str(r["next_earnings_date"]) if r["next_earnings_date"] else "Unknown",
                "Days Until":       r["days_until_earnings"] if r["days_until_earnings"] is not None else "—",
                "Last Earnings":    str(r["last_earnings_date"]) if r["last_earnings_date"] else "Unknown",
                "Days Since":       r["days_since_earnings"] if r["days_since_earnings"] is not None else "—",
                "Confidence Adj":   f"{int(r['confidence_adj'] * 100)}%",
                "Reason":           r["risk_reason"],
            })

        st.dataframe(pd.DataFrame(table_data), use_container_width=True)

        # ── Guidance ──────────────────────────────────────────────────
        st.divider()
        st.subheader("How to use this")
        st.markdown("""
- **🔴 HIGH** — Earnings in 0–2 days. Don't act on BUY/SELL signals. Wait until after the report.
- **🟡 MEDIUM** — Earnings in 3–5 days. Signals are less reliable. Reduce position size.
- **🟢 LOW** — Well outside earnings window. Predictions are as reliable as normal.

*Earnings reports can move a stock 10–20% regardless of technical signals. Always check this page before acting on a recommendation.*
        """)
