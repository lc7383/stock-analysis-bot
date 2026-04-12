"""
Linear Regression Prediction Model
=====================================
Predicts whether a stock price will be UP or DOWN tomorrow
using technical indicators as features.

This is Step 1 of the prediction pipeline:
    Step 1 — Linear Regression (this file)
    Step 2 — Random Forest
    Step 3 — LSTM Time Series

Features used:
    - RSI (14)
    - MACD and MACD Signal
    - Bollinger Band position
    - SMA 20 and SMA 50
    - Volume change
    - Price momentum (1, 5, 10 day returns)

Requirements:
    pip install scikit-learn numpy pandas yfinance

DISCLAIMER: For educational/portfolio purposes only.
            Predictions are not guaranteed to be accurate.
            This is not financial advice.
"""

import logging
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime
from typing import Optional

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import TimeSeriesSplit
    from sklearn.metrics import accuracy_score, classification_report
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logger.warning("scikit-learn not installed. Run: pip install scikit-learn")


# ─────────────────────────────────────────────
# 1. Feature Engineering
# ─────────────────────────────────────────────

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Builds a feature matrix from price and indicator data.

    Features:
        - RSI normalized to 0-1
        - MACD histogram (MACD - Signal)
        - Bollinger Band position (where price sits in the bands)
        - Price above SMA20 (binary)
        - Price above SMA50 (binary)
        - SMA20/SMA50 ratio (golden cross indicator)
        - Volume change % vs 20 day average
        - 1 day return
        - 5 day return
        - 10 day return
        - 20 day return
        - Volatility (20 day rolling std of returns)

    Target:
        - 1 if tomorrow's close is higher than today's
        - 0 if tomorrow's close is lower than today's
    """
    features = pd.DataFrame(index=df.index)

    close  = df["Close"]
    volume = df["Volume"]

    # RSI normalized
    if "RSI" in df.columns:
        features["rsi_norm"] = df["RSI"] / 100.0

    # MACD histogram
    if "MACD" in df.columns and "MACD_Signal" in df.columns:
        features["macd_hist"] = df["MACD"] - df["MACD_Signal"]
        features["macd_norm"] = df["MACD"] / close  # normalize by price

    # Bollinger Band position (0 = at lower band, 1 = at upper band)
    if "BB_Upper" in df.columns and "BB_Lower" in df.columns:
        bb_range = df["BB_Upper"] - df["BB_Lower"]
        bb_range = bb_range.replace(0, np.nan)
        features["bb_position"] = (close - df["BB_Lower"]) / bb_range

    # Moving average signals
    if "SMA_20" in df.columns:
        features["above_sma20"]   = (close > df["SMA_20"]).astype(int)
        features["price_sma20_ratio"] = close / df["SMA_20"]

    if "SMA_50" in df.columns:
        features["above_sma50"]   = (close > df["SMA_50"]).astype(int)
        features["price_sma50_ratio"] = close / df["SMA_50"]

    if "SMA_20" in df.columns and "SMA_50" in df.columns:
        features["sma_ratio"] = df["SMA_20"] / df["SMA_50"]  # golden cross signal

    # Price momentum
    features["return_1d"]  = close.pct_change(1)
    features["return_5d"]  = close.pct_change(5)
    features["return_10d"] = close.pct_change(10)
    features["return_20d"] = close.pct_change(20)

    # Volatility
    features["volatility_20d"] = features["return_1d"].rolling(20).std()

    # Volume features
    vol_ma20 = volume.rolling(20).mean()
    features["volume_ratio"] = volume / vol_ma20.replace(0, np.nan)

    # Target — 1 if next day close is higher, 0 if lower
    features["target"] = (close.shift(-1) > close).astype(int)

    # Drop rows with NaN values
    features = features.dropna()

    return features


# ─────────────────────────────────────────────
# 2. Train Model
# ─────────────────────────────────────────────

def train_model(features: pd.DataFrame) -> dict:
    """
    Trains a Logistic Regression model to predict price direction.

    Uses TimeSeriesSplit for cross validation — this is critical for
    financial data because you must never train on future data.

    Returns a dict with the trained model, scaler, metrics, and
    feature importance scores.
    """
    if not SKLEARN_AVAILABLE:
        return {"error": "scikit-learn not installed. Run: pip install scikit-learn"}

    if len(features) < 60:
        return {"error": "Not enough data — need at least 60 rows after feature engineering"}

    feature_cols = [c for c in features.columns if c != "target"]
    X = features[feature_cols].values
    y = features["target"].values

    # TimeSeriesSplit — respects temporal order, no data leakage
    tscv    = TimeSeriesSplit(n_splits=5)
    scores  = []
    reports = []

    scaler = StandardScaler()
    model  = LogisticRegression(max_iter=1000, random_state=42)

    for train_idx, test_idx in tscv.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled  = scaler.transform(X_test)

        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)

        scores.append(accuracy_score(y_test, y_pred))

    # Final model trained on all data
    X_scaled = scaler.fit_transform(X)
    model.fit(X_scaled, y)

    # Feature importance (coefficients)
    importance = dict(zip(feature_cols, np.abs(model.coef_[0])))
    importance = dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))

    avg_accuracy = np.mean(scores)
    logger.info(f"Model trained — CV accuracy: {avg_accuracy:.1%} (±{np.std(scores):.1%})")

    return {
        "model":           model,
        "scaler":          scaler,
        "feature_cols":    feature_cols,
        "cv_accuracy":     round(avg_accuracy * 100, 1),
        "cv_scores":       [round(s * 100, 1) for s in scores],
        "cv_std":          round(np.std(scores) * 100, 1),
        "feature_importance": importance,
        "baseline_accuracy": round(max(y.mean(), 1 - y.mean()) * 100, 1),
    }


# ─────────────────────────────────────────────
# 3. Make Prediction
# ─────────────────────────────────────────────

def predict_next_day(model_result: dict, features: pd.DataFrame) -> dict:
    """
    Uses the trained model to predict tomorrow's price direction.

    Returns a prediction dict with:
        direction    — "UP" or "DOWN"
        probability  — confidence 0-100%
        signal       — "BUY", "HOLD", or "SELL"
    """
    if "error" in model_result:
        return model_result

    model       = model_result["model"]
    scaler      = model_result["scaler"]
    feature_cols = model_result["feature_cols"]

    # Use the most recent row for prediction
    latest = features[feature_cols].iloc[-1:].values
    latest_scaled = scaler.transform(latest)

    prediction   = model.predict(latest_scaled)[0]
    probability  = model.predict_proba(latest_scaled)[0]

    up_prob   = round(float(probability[1]) * 100, 1)
    down_prob = round(float(probability[0]) * 100, 1)

    direction = "UP" if prediction == 1 else "DOWN"

    # Convert to trading signal
    if up_prob >= 60:
        signal = "BUY"
    elif down_prob >= 60:
        signal = "SELL"
    else:
        signal = "HOLD"

    return {
        "direction":   direction,
        "up_prob":     up_prob,
        "down_prob":   down_prob,
        "signal":      signal,
        "confidence":  max(up_prob, down_prob),
    }


# ─────────────────────────────────────────────
# 4. Full Pipeline for Single Ticker
# ─────────────────────────────────────────────

def run_prediction(ticker: str, period: str = "1y") -> dict:
    """
    Runs the full prediction pipeline for a single ticker:
        1. Fetch price history
        2. Add technical indicators
        3. Build features
        4. Train model
        5. Predict next day direction

    Returns a complete result dict ready for the dashboard.
    """
    logger.info(f"Running prediction for {ticker}...")

    # Fetch and prepare data
    try:
        df = yf.download(ticker, period=period, interval="1d", progress=False, auto_adjust=True)
        if df.empty:
            return {"error": f"No data for {ticker}", "ticker": ticker}
        df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
        df["Ticker"] = ticker
    except Exception as e:
        return {"error": str(e), "ticker": ticker}

    # Add indicators
    try:
        from data_collector import add_technical_indicators
        df = add_technical_indicators(df)
    except Exception as e:
        return {"error": f"Indicator calculation failed: {e}", "ticker": ticker}

    # Build features
    features = build_features(df)
    if len(features) < 60:
        return {"error": f"Not enough data after feature engineering ({len(features)} rows)", "ticker": ticker}

    # Train model
    model_result = train_model(features)
    if "error" in model_result:
        return {**model_result, "ticker": ticker}

    # Predict
    prediction = predict_next_day(model_result, features)

    # Current price info
    current_price = round(float(df["Close"].iloc[-1]), 2)
    prev_price    = round(float(df["Close"].iloc[-2]), 2)
    price_change  = round(current_price - prev_price, 2)
    price_change_pct = round((price_change / prev_price) * 100, 2)

    result = {
        "ticker":             ticker,
        "current_price":      current_price,
        "price_change":       price_change,
        "price_change_pct":   price_change_pct,
        "prediction":         prediction,
        "model_accuracy":     model_result["cv_accuracy"],
        "baseline_accuracy":  model_result["baseline_accuracy"],
        "cv_scores":          model_result["cv_scores"],
        "feature_importance": model_result["feature_importance"],
        "data_points":        len(features),
        "period":             period,
        "as_of":              datetime.utcnow().strftime("%Y-%m-%d"),
    }

    logger.info(
        f"{ticker}: {prediction['direction']} ({prediction['confidence']:.0f}% confidence) "
        f"| Model accuracy: {model_result['cv_accuracy']}% vs baseline {model_result['baseline_accuracy']}%"
    )

    return result


def run_predictions_watchlist(tickers: list[str], period: str = "1y") -> dict[str, dict]:
    """Runs predictions for all tickers in the watchlist."""
    return {ticker: run_prediction(ticker, period=period) for ticker in tickers}


# ─────────────────────────────────────────────
# 5. Demo
# ─────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("Linear Regression Prediction Model — Demo")
    print("DISCLAIMER: Educational use only. Not financial advice.")
    print("=" * 60)

    tickers = ["AAPL", "MSFT", "NVDA"]
    for ticker in tickers:
        result = run_prediction(ticker, period="1y")
        if "error" in result:
            print(f"\n{ticker}: ERROR — {result['error']}")
            continue

        pred = result["prediction"]
        print(f"\n── {ticker} ──────────────────────────────────────")
        print(f"  Current price   : ${result['current_price']} ({result['price_change_pct']:+.2f}%)")
        print(f"  Tomorrow        : {pred['direction']} ({pred['confidence']:.0f}% confidence)")
        print(f"  Signal          : {pred['signal']}")
        print(f"  Model accuracy  : {result['model_accuracy']}% (baseline: {result['baseline_accuracy']}%)")
        print(f"  Top features    :")
        for feat, imp in list(result["feature_importance"].items())[:5]:
            print(f"    {feat:<25} {imp:.4f}")
