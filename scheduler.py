"""
Stock Bot Scheduler
====================
Automatically runs the stock analysis pipeline on a schedule.
Saves each report with a timestamp and maintains a history log.

Requirements:
    pip install schedule

DISCLAIMER: For educational/portfolio purposes only. Not financial advice.
"""

import os
import json
import logging
import schedule
import time
from datetime import datetime
from pathlib import Path

from analysis_agent import run_pipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ── Config ────────────────────────────────────
WATCHLIST      = ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL","MAR","VRTX","CSZIX"]
REPORTS_DIR    = Path("reports")
RUN_TIME_DAILY = "09:00"   # 24-hour format, runs once per day at market open
# ─────────────────────────────────────────────


def run_and_save():
    """Runs the full pipeline and saves a timestamped report."""
    REPORTS_DIR.mkdir(exist_ok=True)
    logger.info(f"Scheduled run starting — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    analysis = run_pipeline(WATCHLIST, save_json=False)

    if "error" in analysis:
        logger.error(f"Run failed: {analysis['error']}")
        return

    # Save timestamped report
    timestamp  = datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath   = REPORTS_DIR / f"report_{timestamp}.json"
    with open(filepath, "w") as f:
        json.dump(analysis, f, indent=2, default=str)
    logger.info(f"Report saved → {filepath}")

    # Append summary line to history log
    log_path = REPORTS_DIR / "history.jsonl"
    with open(log_path, "a") as f:
        for rec in analysis.get("recommendations", []):
            f.write(json.dumps({
                "timestamp":      timestamp,
                "ticker":         rec["ticker"],
                "recommendation": rec["recommendation"],
                "confidence":     rec["confidence"],
            }) + "\n")
    logger.info("History log updated.")


def start_scheduler():
    """Starts the scheduler — runs once immediately then on schedule."""
    logger.info(f"Scheduler started. Daily run at {RUN_TIME_DAILY}.")
    logger.info(f"Watchlist: {', '.join(WATCHLIST)}")
    logger.info("Press Ctrl+C to stop.\n")

    # Run immediately on startup
    run_and_save()

    # Then schedule daily
    schedule.every().day.at(RUN_TIME_DAILY).do(run_and_save)

    while True:
        schedule.run_pending()
        time.sleep(60)


if __name__ == "__main__":
    print("=" * 60)
    print("  Stock Bot Scheduler")
    print("  DISCLAIMER: Educational use only. Not financial advice.")
    print("=" * 60)
    try:
        start_scheduler()
    except KeyboardInterrupt:
        logger.info("Scheduler stopped.")
