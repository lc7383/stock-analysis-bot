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

import pandas as pd
import yfinance as yf

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

PORTFOLIO_FILE = Path("portfolio.json")


# ─────────────────────────────────────────────
# 1. Storage — works locally and on Streamlit Cloud
# ─────────────────────────────────────────────

def _is_streamlit() -> bool:
    """Check if running inside Streamlit."""
    try:
        import streamlit as st
        return True
    except ImportError:
        return False


def load_portfolio() -> dict:
    """
    Loads portfolio from the best available source:
        1. Streamlit session state (if running in Streamlit)
        2. Local portfolio.json file
        3. Empty portfolio as fallback
    """
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
    """
    Saves portfolio to session state and local file.
    On Streamlit Cloud the local file may not persist between
    deployments — use the Export feature to back up your holdings.
    """
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
    """Returns the portfolio as a JSON string for download."""
    if portfolio is None:
        portfolio = load_portfolio()
    return json.dumps(portfolio, indent=2)


def import_portfolio_json(json_str: str) -> dict:
    """
    Imports a portfolio from a JSON string.
    Use this to restore holdings after a cloud session reset.
    """
    portfolio = json.loads(json_str)
    save_portfolio(portfolio)
    logger.info(f"Imported portfolio with {len(portfolio.get('holdings', []))} holdings.")
    return portfolio


# ─────────────────────────────────────────────
# 2. Holdings Management
# ─────────────────────────────────────────────

def add_holding(ticker: str, shares: float, buy_price: float, buy_date: str = None) -> dict:
    """
    Adds or updates a holding in the portfolio.
    If the ticker already exists, averages the cost basis.
    """
    portfolio = load_portfolio()
    ticker    = ticker.upper()
    buy_date  = buy_date or datetime.utcnow().strftime("%Y-%m-%d")

    for holding in portfolio["holdings"]:
        if holding["ticker"] == ticker:
            total_shares    = holding["shares"] + shares
            avg_price       = (holding["shares"] * holding["buy_price"] + shares * buy_price) / total_shares
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
    """Removes a holding from the portfolio."""
    portfolio = load_portfolio()
    portfolio["holdings"] = [h for h in portfolio["holdings"] if h["ticker"] != ticker.upper()]
    save_portfolio(portfolio)
    return portfolio


# ─────────────────────────────────────────────
# 3. Portfolio Valuation
# ─────────────────────────────────────────────

def get_current_prices(tickers: list[str]) -> dict[str, float]:
    """Fetches current prices for a list of tickers."""
    prices = {}
    try:
        # Fetch all tickers at once — faster and avoids rate limiting
        data = yf.download(tickers, period="2d", interval="1d",
                          progress=False, auto_adjust=True)
        print("DATA COLUMNS:", data.columns.tolist())
        print("DATA TAIL:", data.tail(2))
        if not data.empty:
    close = data["Close"]
    for ticker in tickers:
        try:
            if len(tickers) == 1:
                # Single ticker — flat column structure
                prices[ticker] = round(float(close.dropna().iloc[-1]), 2)
            elif ticker in close.columns:
                # Multiple tickers — column per ticker
                prices[ticker] = round(float(close[ticker].dropna().iloc[-1]), 2)
        except Exception as e:
        logger.warning(f"Price parse failed for {ticker}: {e}")        
        # Fallback — fetch individually
        for ticker in tickers:
            try:
                df = yf.download(ticker, period="2d", interval="1d",
                               progress=False, auto_adjust=True)
                if not df.empty:
                    prices[ticker] = round(float(df["Close"].iloc[-1]), 2)
            except Exception:
                pass
    return prices


def calculate_portfolio_value(portfolio: dict = None) -> dict:
    """
    Calculates current portfolio value, P&L, and per-holding metrics.
    """
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
        ticker      = h["ticker"]
        shares      = h["shares"]
        buy_price   = h["buy_price"]
        curr_price  = prices.get(ticker, buy_price)

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
# 4. Compare with Bot Recommendations
# ─────────────────────────────────────────────

def compare_with_recommendations(portfolio_value: dict, analysis: dict) -> list[dict]:
    """Compares holdings against latest Claude recommendations."""
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
# 5. Demo
# ─────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("Portfolio Tracker — Demo")
    print("DISCLAIMER: Educational use only. Not financial advice.")
    print("=" * 60)

    add_holding("AAPL",  10, 175.00, "2024-01-15")
    add_holding("MSFT",   5, 380.00, "2024-02-01")
    add_holding("NVDA",   3, 500.00, "2024-03-01")

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



# ─────────────────────────────────────────────
# 1. Portfolio Storage
# ─────────────────────────────────────────────

def load_portfolio() -> dict:
    """Loads portfolio from portfolio.json. Returns empty portfolio if not found."""
    if PORTFOLIO_FILE.exists():
        with open(PORTFOLIO_FILE) as f:
            return json.load(f)
    return {"holdings": [], "updated_at": None}


def save_portfolio(portfolio: dict) -> None:
    """Saves portfolio to portfolio.json."""
    portfolio["updated_at"] = datetime.utcnow().isoformat()
    with open(PORTFOLIO_FILE, "w") as f:
        json.dump(portfolio, f, indent=2)
    logger.info("Portfolio saved.")


def add_holding(ticker: str, shares: float, buy_price: float, buy_date: str = None) -> dict:
    """
    Adds or updates a holding in the portfolio.

    Args:
        ticker:    stock symbol e.g. "AAPL"
        shares:    number of shares held
        buy_price: average purchase price per share
        buy_date:  date purchased (YYYY-MM-DD), defaults to today

    Returns the updated portfolio.
    """
    portfolio = load_portfolio()
    ticker    = ticker.upper()
    buy_date  = buy_date or datetime.utcnow().strftime("%Y-%m-%d")

    # Check if holding already exists — update if so
    for holding in portfolio["holdings"]:
        if holding["ticker"] == ticker:
            # Average down/up the cost basis
            total_shares    = holding["shares"] + shares
            avg_price       = (holding["shares"] * holding["buy_price"] + shares * buy_price) / total_shares
            holding["shares"]    = round(total_shares, 4)
            holding["buy_price"] = round(avg_price, 4)
            holding["buy_date"]  = buy_date
            save_portfolio(portfolio)
            logger.info(f"Updated holding: {ticker} — {total_shares} shares @ ${avg_price:.2f}")
            return portfolio

    # New holding
    portfolio["holdings"].append({
        "ticker":    ticker,
        "shares":    round(shares, 4),
        "buy_price": round(buy_price, 4),
        "buy_date":  buy_date,
        "notes":     "",
    })
    save_portfolio(portfolio)
    logger.info(f"Added holding: {ticker} — {shares} shares @ ${buy_price:.2f}")
    return portfolio


def remove_holding(ticker: str) -> dict:
    """Removes a holding from the portfolio."""
    portfolio = load_portfolio()
    portfolio["holdings"] = [h for h in portfolio["holdings"] if h["ticker"] != ticker.upper()]
    save_portfolio(portfolio)
    logger.info(f"Removed holding: {ticker}")
    return portfolio


def update_shares(ticker: str, shares: float) -> dict:
    """Updates the share count for an existing holding."""
    portfolio = load_portfolio()
    for holding in portfolio["holdings"]:
        if holding["ticker"] == ticker.upper():
            holding["shares"] = round(shares, 4)
            save_portfolio(portfolio)
            return portfolio
    logger.warning(f"Holding not found: {ticker}")
    return portfolio


# ─────────────────────────────────────────────
# 2. Portfolio Valuation
# ─────────────────────────────────────────────

def get_current_prices(tickers: list[str]) -> dict[str, float]:
    """Fetches current prices for a list of tickers using yfinance."""
    prices = {}
    for ticker in tickers:
        try:
            df = yf.download(ticker, period="2d", interval="1d", progress=False, auto_adjust=True)
            if not df.empty:
                prices[ticker] = round(float(df["Close"].iloc[-1]), 2)
        except Exception as e:
            logger.warning(f"Could not fetch price for {ticker}: {e}")
    return prices


def calculate_portfolio_value(portfolio: dict = None) -> dict:
    """
    Calculates current portfolio value, P&L, and per-holding metrics.

    Returns a dict with:
        holdings        — list of holdings with current values
        total_cost      — total amount invested
        total_value     — current market value
        total_gain_loss — unrealized P&L in dollars
        total_return_pct — unrealized P&L as percentage
        best_performer  — ticker with highest return %
        worst_performer — ticker with lowest return %
        as_of           — timestamp of calculation
    """
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
        ticker      = h["ticker"]
        shares      = h["shares"]
        buy_price   = h["buy_price"]
        curr_price  = prices.get(ticker, buy_price)

        cost        = round(shares * buy_price, 2)
        value       = round(shares * curr_price, 2)
        gain_loss   = round(value - cost, 2)
        return_pct  = round((gain_loss / cost) * 100, 2) if cost > 0 else 0

        # Days held
        try:
            buy_dt   = datetime.strptime(h["buy_date"], "%Y-%m-%d")
            days_held = (datetime.utcnow() - buy_dt).days
        except Exception:
            days_held = 0

        enriched.append({
            "ticker":       ticker,
            "shares":       shares,
            "buy_price":    buy_price,
            "current_price":curr_price,
            "cost":         cost,
            "value":        value,
            "gain_loss":    gain_loss,
            "return_pct":   return_pct,
            "buy_date":     h.get("buy_date", ""),
            "days_held":    days_held,
            "notes":        h.get("notes", ""),
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
# 3. Compare with Bot Recommendations
# ─────────────────────────────────────────────

def compare_with_recommendations(portfolio_value: dict, analysis: dict) -> list[dict]:
    """
    Compares your holdings against the latest Claude recommendations.

    Returns a list of alignment dicts — one per holding that has
    a matching recommendation.
    """
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

        # Alignment assessment
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
# 4. Demo
# ─────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("Portfolio Tracker — Demo")
    print("DISCLAIMER: Educational use only. Not financial advice.")
    print("=" * 60)

    # Add some sample holdings
    add_holding("AAPL",  10,  175.00, "2024-01-15")
    add_holding("MSFT",   5,  380.00, "2024-02-01")
    add_holding("NVDA",   3,  500.00, "2024-03-01")

    # Calculate value
    result = calculate_portfolio_value()

    print(f"\nPortfolio as of {result['as_of']}")
    print(f"{'Ticker':<8} {'Shares':>8} {'Buy Price':>10} {'Current':>10} {'Value':>10} {'P&L':>10} {'Return':>8}")
    print("-" * 70)

    for h in result["holdings"]:
        print(f"{h['ticker']:<8} {h['shares']:>8.2f} ${h['buy_price']:>9.2f} "
              f"${h['current_price']:>9.2f} ${h['value']:>9.2f} "
              f"${h['gain_loss']:>+9.2f} {h['return_pct']:>+7.2f}%")

    print("-" * 70)
    print(f"{'TOTAL':<8} {'':>8} {'':>10} {'':>10} ${result['total_value']:>9.2f} "
          f"${result['total_gain_loss']:>+9.2f} {result['total_return_pct']:>+7.2f}%")
    print(f"\nBest:  {result['best_performer']}")
    print(f"Worst: {result['worst_performer']}")
