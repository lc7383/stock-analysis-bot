# ============================================================
# HOW TO ADD AI CHAT TO YOUR dashboard.py
# ============================================================
# This file shows the exact lines to add. Don't copy the whole
# file — just find the matching sections in your dashboard.py
# and add the highlighted lines.
# ============================================================


# ── STEP 1: Add this import at the top of dashboard.py ──────
from chat_page import render_chat_page


# ── STEP 2: Find your sidebar page list and add "AI Chat" ───
# Your current code probably looks something like this:
page = st.sidebar.radio("Navigation", [
    "Screener",
    "Latest Report",
    "History",
    "Run Analysis",
    "Backtest",
    "Predictions",
    "Watchlist & Alerts",
    "AI Chat",           # <-- ADD THIS LINE
])


# ── STEP 3: Find your page routing block and add this ────────
# Your current code probably has a series of if/elif blocks.
# Add this elif at the end:

elif page == "AI Chat":
    render_chat_page()


# ── That's it. No other changes needed. ─────────────────────
# The chat page reads your .env API key, loads the latest
# report from reports/, and runs entirely inside Streamlit.
# No separate server or HTML file required.
