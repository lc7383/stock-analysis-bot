"""
Portfolio Tracker Module
=========================
Tracks your actual stock holdings, calculates P&L,
and compares your positions against the bot's recommendations.

Storage strategy:
    - Local: saves to portfolio.json in the project folder
    - Cloud: uses Streamlit session state (persists during session)
    - Both: supports JSON export/import to back up and restore holdings

DISCLAIMER: For educational/portfolio purposes only.
            This is not financial advice.
"""

import json
import logging
from datetime import datetime
from pathlib import Path

import yfinance as yf

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

PORTFOLIO_FILE = Path("portfolio.json")


# ─────────────────────────────────────────────
# 1. Storage
# ─────────────────────────────────────────────

def _is_streamlit() -> bool:
    try:
        import streamlit as st
        return True
    except ImportError:
        return False


def load_portfolio() -> dict:
    if _is_streamlit():
        import streamlit as st
        if "portfolio_data" in st.session_state and st.session_state["portfolio_data"]:
            return st.session_state["portfolio_data"]

    if PORTFOLIO_FILE.exists():
        try:
            with open(PORTFOLIO_FILE) as f:
                data = json.load(f)
                if _is_streamlit():
                    import streamlit as st
                    st.session_state["portfolio_data"] = data
                return data
        except Exception:
            pass

    empty = {"holdings": [], "updated_at": None}
    if _is_streamlit():
        import streamlit as st
        st.session_state["portfolio_data"] = empty
    return empty


def save_portfolio(portfolio: dict) -> None:
    portfolio["updated_at"] = datetime.utcnow().isoformat()
    if _is_streamlit():
        import streamlit as st
        st.session_state["portfolio_data"] = portfolio
    try:
        with open(PORTFOLIO_FILE, "w") as f:
            json.dump(portfolio, f, indent=2)
        logger.info("Portfolio saved to file.")
    except Exception as e:
        logger.warning(f"Could not save to file (normal on cloud): {e}")


def export_portfolio_json(portfolio: dict = None) -> str:
    if portfolio is None:
        portfolio = load_portfolio()
    return json.dumps(portfolio, indent=2)


def import_portfolio_json(json_str: str) -> dict:
    portfolio = json.loads(json_str)
    save_portfolio(portfolio)
    logger.info(f"Imported portfolio with {len(portfolio.get('holdings', []))} holdings.")
    return portfolio


# ─────────────────────────────────────────────
# 2. Holdings Management
# ─────────────────────────────────────────────

def add_holding(ticker: str, shares: float, buy_price: float, buy_date: str = None) -> dict:
    portfolio = load_portfolio()
    ticker    = ticker.upper()
    buy_date  = buy_date or datetime.utcnow().strftime("%Y-%m-%d")

    for holding in portfolio["holdings"]:
        if holding["ticker"] == ticker:
            total_shares = holding["shares"] + shares
            avg_price    = (holding["shares"] * holding["buy_price"] + shares * buy_price) / total_shares
            holding["shares"]    = round(total_shares, 4)
            holding["buy_price"] = round(avg_price, 4)
            holding["buy_date"]  = buy_date
            save_portfolio(portfolio)
            logger.info(f"Updated {ticker}: {total_shares} shares @ ${avg_price:.2f} avg")
            return portfolio

    portfolio["holdings"].append({
        "ticker":    ticker,
        "shares":    round(shares, 4),
        "buy_price": round(buy_price, 4),
        "buy_date":  buy_date,
        "notes":     "",
    })
    save_portfolio(portfolio)
    logger.info(f"Added {ticker}: {shares} shares @ ${buy_price:.2f}")
    return portfolio


def remove_holding(ticker: str) -> dict:
    portfolio = load_portfolio()
    portfolio["holdings"] = [h for h in portfolio["holdings"] if h["ticker"] != ticker.upper()]
    save_portfolio(portfolio)
    return portfolio


# ─────────────────────────────────────────────
# 3. Price Fetching
# ─────────────────────────────────────────────

def get_current_prices(tickers: list[str]) -> dict[str, float]:
    """Fetches current prices handling yfinance MultiIndex columns."""
    prices = {}

    # Batch fetch
    try:
        data = yf.download(tickers, period="2d", interval="1d",
                           progress=False, auto_adjust=True)
    except Exception as e:
        logger.warning(f"Batch fetch failed: {e}")
        data = None

    if data is not None and not data.empty:
        try:
            close = data["Close"]
            if hasattr(close, "columns"):
                for ticker in tickers:
                    # Handle both plain and tuple column names
                    col_key = None
                    if ticker in close.columns:
                        col_key = ticker
                    else:
                        for col in close.columns:
                            if isinstance(col, tuple) and ticker in col:
                                col_key = col
                                break
                    if col_key is not None:
                        val = close[col_key].dropna()
                        if not val.empty:
                            prices[ticker] = round(float(val.iloc[-1]), 2)
            else:
                # Single ticker
                val = close.dropna()
                if not val.empty and len(tickers) == 1:
                    prices[tickers[0]] = round(float(val.iloc[-1]), 2)
        except Exception as e:
            logger.warning(f"Price parse failed: {e}")

    # Fallback for missing tickers
    for ticker in [t for t in tickers if t not in prices]:
        try:
            df = yf.download(ticker, period="2d", interval="1d",
                             progress=False, auto_adjust=True)
            if not df.empty:
                close_col = df["Close"]
                if hasattr(close_col, "columns"):
                    close_col = close_col.iloc[:, 0]
                prices[ticker] = round(float(close_col.dropna().iloc[-1]), 2)
        except Exception as e:
            logger.warning(f"Individual fetch failed for {ticker}: {e}")

    return prices


# ─────────────────────────────────────────────
# 4. Portfolio Valuation
# ─────────────────────────────────────────────

def calculate_portfolio_value(portfolio: dict = None) -> dict:
    if portfolio is None:
        portfolio = load_portfolio()

    holdings = portfolio.get("holdings", [])
    if not holdings:
        return {
            "holdings": [], "total_cost": 0, "total_value": 0,
            "total_gain_loss": 0, "total_return_pct": 0,
            "best_performer": None, "worst_performer": None,
            "as_of": datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC"),
        }

    tickers = [h["ticker"] for h in holdings]
    prices  = get_current_prices(tickers)

    enriched    = []
    total_cost  = 0
    total_value = 0

    for h in holdings:
        ticker     = h["ticker"]
        shares     = h["shares"]
        buy_price  = h["buy_price"]
        curr_price = prices.get(ticker, buy_price)

        cost       = round(shares * buy_price, 2)
        value      = round(shares * curr_price, 2)
        gain_loss  = round(value - cost, 2)
        return_pct = round((gain_loss / cost) * 100, 2) if cost > 0 else 0

        try:
            buy_dt    = datetime.strptime(h["buy_date"], "%Y-%m-%d")
            days_held = (datetime.utcnow() - buy_dt).days
        except Exception:
            days_held = 0

        enriched.append({
            "ticker":        ticker,
            "shares":        shares,
            "buy_price":     buy_price,
            "current_price": curr_price,
            "cost":          cost,
            "value":         value,
            "gain_loss":     gain_loss,
            "return_pct":    return_pct,
            "buy_date":      h.get("buy_date", ""),
            "days_held":     days_held,
            "notes":         h.get("notes", ""),
        })

        total_cost  += cost
        total_value += value

    total_gain_loss  = round(total_value - total_cost, 2)
    total_return_pct = round((total_gain_loss / total_cost) * 100, 2) if total_cost > 0 else 0
    best  = max(enriched, key=lambda x: x["return_pct"]) if enriched else None
    worst = min(enriched, key=lambda x: x["return_pct"]) if enriched else None

    return {
        "holdings":         enriched,
        "total_cost":       round(total_cost, 2),
        "total_value":      round(total_value, 2),
        "total_gain_loss":  total_gain_loss,
        "total_return_pct": total_return_pct,
        "best_performer":   best["ticker"] if best else None,
        "worst_performer":  worst["ticker"] if worst else None,
        "as_of":            datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC"),
    }


# ─────────────────────────────────────────────
# 5. Compare with Recommendations
# ─────────────────────────────────────────────

def compare_with_recommendations(portfolio_value: dict, analysis: dict) -> list[dict]:
    if not analysis or "recommendations" not in analysis:
        return []

    rec_map = {r["ticker"]: r for r in analysis.get("recommendations", [])}
    results = []

    for holding in portfolio_value.get("holdings", []):
        ticker = holding["ticker"]
        rec    = rec_map.get(ticker)
        if not rec:
            continue

        action     = rec["recommendation"]
        confidence = rec["confidence"]
        return_pct = holding["return_pct"]

        if action == "SELL" and return_pct > 0:
            alignment = "Take profit — bot says SELL while you are in profit"
            priority  = "HIGH"
        elif action == "SELL" and return_pct < -5:
            alignment = "Cut losses — bot says SELL while you are down"
            priority  = "HIGH"
        elif action == "BUY" and return_pct < -10:
            alignment = "Potential add — bot says BUY and you are down 10%+"
            priority  = "MEDIUM"
        elif action == "BUY" and return_pct > 0:
            alignment = "Bot confirms your position — currently profitable"
            priority  = "LOW"
        elif action == "HOLD":
            alignment = "Hold and monitor — no action needed"
            priority  = "LOW"
        else:
            alignment = f"Bot says {action} — review recommended"
            priority  = "MEDIUM"

        results.append({
            "ticker":     ticker,
            "shares":     holding["shares"],
            "return_pct": return_pct,
            "gain_loss":  holding["gain_loss"],
            "action":     action,
            "confidence": confidence,
            "alignment":  alignment,
            "priority":   priority,
        })

    results.sort(key=lambda x: {"HIGH": 0, "MEDIUM": 1, "LOW": 2}[x["priority"]])
    return results


# ─────────────────────────────────────────────
# 6. Demo
# ─────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("Portfolio Tracker — Demo")
    print("DISCLAIMER: Educational use only. Not financial advice.")
    print("=" * 60)

    add_holding("AAPL", 10, 175.00, "2024-01-15")
    add_holding("MSFT",  5, 380.00, "2024-02-01")
    add_holding("NVDA",  3, 500.00, "2024-03-01")

    result = calculate_portfolio_value()
    print(f"\nPortfolio as of {result['as_of']}")
    print(f"{'Ticker':<8} {'Shares':>8} {'Buy':>10} {'Current':>10} {'Value':>10} {'P&L':>10} {'Ret%':>8}")
    print("-" * 70)
    for h in result["holdings"]:
        print(f"{h['ticker']:<8} {h['shares']:>8.2f} ${h['buy_price']:>9.2f} "
              f"${h['current_price']:>9.2f} ${h['value']:>9.2f} "
              f"${h['gain_loss']:>+9.2f} {h['return_pct']:>+7.2f}%")
    print("-" * 70)
    print(f"{'TOTAL':<50} ${result['total_value']:>9.2f} "
          f"${result['total_gain_loss']:>+9.2f} {result['total_return_pct']:>+7.2f}%")
