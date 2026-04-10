"""
SEC Insider Trading Module
============================
Fetches insider trading data from the SEC EDGAR API.
No API key required — completely free.

Insider trading is one of the strongest signals in investing:
  - Executives BUYING their own stock = strong bullish signal
  - Executives SELLING their own stock = potential bearish signal
  - Cluster buying (multiple insiders buying) = very bullish

SEC Forms tracked:
  - Form 4  : Statement of changes in beneficial ownership (most important)
  - Form 3  : Initial statement of beneficial ownership
  - Form 144: Notice of proposed sale of securities

DISCLAIMER: For educational/portfolio purposes only. Not financial advice.
"""

import time
import logging
import requests
from datetime import datetime, timedelta
from typing import Optional

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# SEC requires a descriptive User-Agent with contact info
SEC_HEADERS = {
    "User-Agent":      "StockAnalysisBot research@example.com",
    "Accept-Encoding": "gzip, deflate",
    "Host":            "data.sec.gov",
}

SEC_SEARCH_HEADERS = {
    "User-Agent":      "StockAnalysisBot research@example.com",
    "Accept-Encoding": "gzip, deflate",
    "Host":            "efts.sec.gov",
}


# ─────────────────────────────────────────────
# 1. Get Company CIK from Ticker
# ─────────────────────────────────────────────

_cik_cache = {}

def get_cik(ticker: str) -> Optional[str]:
    """
    Looks up the SEC CIK (Central Index Key) for a ticker symbol.
    CIK is required to query SEC EDGAR for a specific company.
    Results are cached to avoid repeated lookups.
    """
    ticker = ticker.upper()
    if ticker in _cik_cache:
        return _cik_cache[ticker]

    try:
        url  = "https://www.sec.gov/files/company_tickers.json"
        resp = requests.get(url, headers={"User-Agent": "StockAnalysisBot research@example.com"}, timeout=10)
        resp.raise_for_status()
        data = resp.json()

        for entry in data.values():
            if entry.get("ticker", "").upper() == ticker:
                cik = str(entry["cik_str"]).zfill(10)
                _cik_cache[ticker] = cik
                logger.info(f"CIK for {ticker}: {cik}")
                return cik

        logger.warning(f"CIK not found for {ticker}")
        return None

    except Exception as e:
        logger.error(f"CIK lookup failed for {ticker}: {e}")
        return None


# ─────────────────────────────────────────────
# 2. Fetch Recent Form 4 Filings
# ─────────────────────────────────────────────

def fetch_insider_filings(ticker: str, days_back: int = 90) -> list[dict]:
    """
    Fetches recent Form 4 insider trading filings from SEC EDGAR
    for a given ticker.

    Form 4 must be filed within 2 business days of a transaction,
    making it a near real-time signal of insider activity.

    Returns a list of filing dicts with:
        filer_name, filer_role, transaction_date, transaction_type,
        shares, price_per_share, total_value, ownership_type
    """
    cik = get_cik(ticker)
    if not cik:
        return []

    try:
        # Get all filings for this company
        url  = f"https://data.sec.gov/submissions/CIK{cik}.json"
        resp = requests.get(url, headers=SEC_HEADERS, timeout=10)
        resp.raise_for_status()
        data = resp.json()

        recent      = data.get("filings", {}).get("recent", {})
        forms       = recent.get("form", [])
        dates       = recent.get("filingDate", [])
        accessions  = recent.get("accessionNumber", [])
        descriptions = recent.get("primaryDocument", [])

        cutoff = datetime.now() - timedelta(days=days_back)
        form4_filings = []

        for i, form in enumerate(forms):
            if form != "4":
                continue
            try:
                filing_date = datetime.strptime(dates[i], "%Y-%m-%d")
                if filing_date < cutoff:
                    continue
                form4_filings.append({
                    "accession":   accessions[i].replace("-", ""),
                    "date":        dates[i],
                    "cik":         cik,
                })
            except Exception:
                continue

        logger.info(f"Found {len(form4_filings)} Form 4 filings for {ticker} in last {days_back} days")

        # Parse each filing for transaction details
        transactions = []
        for filing in form4_filings[:10]:  # limit to 10 most recent
            txns = _parse_form4(filing["cik"], filing["accession"], filing["date"])
            transactions.extend(txns)
            time.sleep(0.15)  # SEC rate limit: max 10 requests/second

        return transactions

    except Exception as e:
        logger.error(f"Insider filings fetch failed for {ticker}: {e}")
        return []


def _parse_form4(cik: str, accession: str, filing_date: str) -> list[dict]:
    """
    Parses a single Form 4 filing to extract transaction details.
    """
    try:
        # Format accession number for URL
        acc_formatted = f"{accession[:10]}-{accession[10:12]}-{accession[12:]}"
        url = f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{accession}/{acc_formatted}-index.htm"

        # Get the filing index to find the XML file
        resp = requests.get(
            f"https://data.sec.gov/submissions/CIK{cik}.json",
            headers=SEC_HEADERS, timeout=10
        )

        # Use EDGAR full-text search for Form 4 XML
        xml_url = f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{accession}/form4.xml"
        resp    = requests.get(
            xml_url,
            headers={"User-Agent": "StockAnalysisBot research@example.com"},
            timeout=10
        )

        if resp.status_code != 200:
            return []

        return _extract_transactions_from_xml(resp.text, filing_date)

    except Exception:
        return []


def _extract_transactions_from_xml(xml_text: str, filing_date: str) -> list[dict]:
    """
    Extracts transaction data from Form 4 XML.
    Uses simple string parsing to avoid xml library dependencies.
    """
    transactions = []

    def extract_tag(text, tag):
        start = text.find(f"<{tag}>")
        end   = text.find(f"</{tag}>")
        if start != -1 and end != -1:
            return text[start + len(tag) + 2:end].strip()
        return None

    # Extract filer info
    filer_name = extract_tag(xml_text, "rptOwnerName") or "Unknown"
    filer_role = ""
    if "<isOfficer>1</isOfficer>" in xml_text:
        filer_role = extract_tag(xml_text, "officerTitle") or "Officer"
    elif "<isDirector>1</isDirector>" in xml_text:
        filer_role = "Director"
    elif "<isTenPercentOwner>1</isTenPercentOwner>" in xml_text:
        filer_role = "10% Owner"

    # Find all non-derivative transactions
    sections = xml_text.split("<nonDerivativeTransaction>")
    for section in sections[1:]:
        try:
            code  = extract_tag(section, "transactionCode")
            shares_str = extract_tag(section, "transactionShares")
            price_str  = extract_tag(section, "transactionPricePerShare")
            date_str   = extract_tag(section, "transactionDate") or filing_date
            ownership  = extract_tag(section, "ownershipNature") or ""

            if not shares_str:
                continue

            shares = abs(float(shares_str))
            price  = float(price_str) if price_str else 0.0
            total  = round(shares * price, 2)

            # Decode transaction type
            tx_type = _decode_transaction_code(code)
            if not tx_type:
                continue

            transactions.append({
                "filer_name":      filer_name,
                "filer_role":      filer_role,
                "transaction_date": date_str[:10] if date_str else filing_date,
                "transaction_type": tx_type,
                "transaction_code": code,
                "shares":          int(shares),
                "price_per_share": round(price, 2),
                "total_value":     total,
                "is_buy":          code in ("P", "A"),
            })

        except Exception:
            continue

    return transactions


def _decode_transaction_code(code: str) -> Optional[str]:
    """Converts SEC transaction codes to human-readable labels."""
    codes = {
        "P": "Open market purchase",
        "S": "Open market sale",
        "A": "Award / grant",
        "D": "Disposition to company",
        "F": "Tax withholding",
        "M": "Option exercise",
        "G": "Gift",
        "V": "Voluntary transaction",
    }
    return codes.get(code)


# ─────────────────────────────────────────────
# 3. Summarize Insider Activity
# ─────────────────────────────────────────────

def summarize_insider_activity(ticker: str, transactions: list[dict]) -> dict:
    """
    Summarizes insider transactions into a signal-ready dict for Claude.

    Returns:
        {
            ticker, total_transactions, buys, sells,
            net_shares_bought, total_buy_value, total_sell_value,
            signal, signal_strength, recent_transactions, summary
        }
    """
    if not transactions:
        return {
            "ticker":             ticker,
            "total_transactions": 0,
            "buys":               0,
            "sells":              0,
            "net_shares_bought":  0,
            "total_buy_value":    0,
            "total_sell_value":   0,
            "signal":             "NEUTRAL",
            "signal_strength":    "No insider data available in last 90 days",
            "recent_transactions": [],
            "summary":            "No insider trading data found for this period.",
        }

    buys       = [t for t in transactions if t["is_buy"]]
    sells      = [t for t in transactions if not t["is_buy"] and t["transaction_code"] in ("S", "D")]
    open_buys  = [t for t in transactions if t["transaction_code"] == "P"]  # open market only
    open_sells = [t for t in transactions if t["transaction_code"] == "S"]  # open market only

    total_buy_value  = sum(t["total_value"] for t in buys)
    total_sell_value = sum(t["total_value"] for t in sells)
    net_shares       = sum(t["shares"] for t in buys) - sum(t["shares"] for t in sells)

    # Signal logic — open market transactions are most meaningful
    signal = "NEUTRAL"
    if len(open_buys) >= 2 and len(open_buys) > len(open_sells):
        signal = "BULLISH"
    elif len(open_buys) == 1 and len(open_sells) == 0:
        signal = "SLIGHTLY BULLISH"
    elif len(open_sells) >= 2 and len(open_sells) > len(open_buys):
        signal = "BEARISH"
    elif len(open_sells) == 1 and len(open_buys) == 0:
        signal = "SLIGHTLY BEARISH"

    # Build strength description
    if signal in ("BULLISH", "SLIGHTLY BULLISH"):
        strength = f"{len(open_buys)} insider(s) bought on open market totaling ${total_buy_value:,.0f}"
    elif signal in ("BEARISH", "SLIGHTLY BEARISH"):
        strength = f"{len(open_sells)} insider(s) sold on open market totaling ${total_sell_value:,.0f}"
    else:
        strength = f"{len(transactions)} transactions found, no clear directional signal"

    # Recent transactions for Claude (last 5, most relevant first)
    sorted_txns = sorted(transactions, key=lambda x: x["transaction_date"], reverse=True)
    recent = [{
        "date":   t["transaction_date"],
        "who":    f"{t['filer_name']} ({t['filer_role']})",
        "action": t["transaction_type"],
        "shares": t["shares"],
        "value":  f"${t['total_value']:,.0f}" if t["total_value"] > 0 else "N/A",
    } for t in sorted_txns[:5]]

    summary = (
        f"Insider signal: {signal}. {strength}. "
        f"Total: {len(buys)} buys, {len(sells)} sells in last 90 days."
    )

    return {
        "ticker":              ticker,
        "total_transactions":  len(transactions),
        "buys":                len(buys),
        "sells":               len(sells),
        "open_market_buys":    len(open_buys),
        "open_market_sells":   len(open_sells),
        "net_shares_bought":   net_shares,
        "total_buy_value":     round(total_buy_value, 2),
        "total_sell_value":    round(total_sell_value, 2),
        "signal":              signal,
        "signal_strength":     strength,
        "recent_transactions": recent,
        "summary":             summary,
    }


# ─────────────────────────────────────────────
# 4. Main Entry Point
# ─────────────────────────────────────────────

def fetch_insider_data(ticker: str) -> dict:
    """
    Fetches and summarizes insider trading data for a single ticker.
    Main function to call from analysis_agent.py.
    """
    logger.info(f"Fetching insider trading data for {ticker}...")
    transactions = fetch_insider_filings(ticker)
    summary      = summarize_insider_activity(ticker, transactions)
    logger.info(f"Insider signal for {ticker}: {summary['signal']} — {summary['signal_strength']}")
    return summary


def fetch_insider_data_watchlist(tickers: list[str]) -> dict[str, dict]:
    """
    Fetches insider data for a list of tickers.
    Returns { ticker: insider_summary }
    """
    results = {}
    for ticker in tickers:
        results[ticker] = fetch_insider_data(ticker)
        time.sleep(0.5)  # Be polite to SEC servers
    return results


# ─────────────────────────────────────────────
# 5. Demo
# ─────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("SEC Insider Trading Module — Demo")
    print("DISCLAIMER: Educational use only. Not financial advice.")
    print("=" * 60)

    tickers = ["AAPL", "MSFT", "NVDA"]
    for ticker in tickers:
        data = fetch_insider_data(ticker)
        print(f"\n── {ticker} ──────────────────────────────────")
        print(f"  Signal         : {data['signal']}")
        print(f"  Strength       : {data['signal_strength']}")
        print(f"  Open mkt buys  : {data['open_market_buys']}")
        print(f"  Open mkt sells : {data['open_market_sells']}")
        print(f"  Summary        : {data['summary']}")
        if data["recent_transactions"]:
            print(f"  Recent activity:")
            for t in data["recent_transactions"]:
                print(f"    {t['date']}  {t['who']:40} {t['action']:25} {t['shares']:>10} shares  {t['value']}")
