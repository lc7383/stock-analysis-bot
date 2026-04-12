"""
Random Forest Prediction Model
================================
Predicts whether a stock price will be UP or DOWN tomorrow
using an ensemble of decision trees.

Improvements over Linear Regression (Step 1):
    - Handles non-linear relationships between indicators
    - More robust to outliers and noisy data
    - Provides better feature importance scores
    - Generally higher accuracy on financial data
    - Less likely to overfit than a single decision tree

This is Step 2 of the prediction pipeline:
    Step 1 — Linear Regression  (prediction_lr.py)
    Step 2 — Random Forest      (this file)
    Step 3 — LSTM Time Series   (coming next)

Same features as Step 1:
    - RSI, MACD, Bollinger Bands, SMA 20/50
    - Volume change, Price momentum
    - Volatility

Additional Random Forest features:
    - Lagged features (yesterday's indicators)
    - Day of week (markets behave differently Mon vs Fri)
    - Distance from 52-week high/low

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

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import TimeSeriesSplit
    from sklearn.metrics import accuracy_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logger.warning("scikit-learn not installed. Run: pip install scikit-learn")


# ─────────────────────────────────────────────
# 1. Feature Engineering (enhanced)
# ─────────────────────────────────────────────

def build_features_rf(df: pd.DataFrame) -> pd.DataFrame:
    """
    Builds an enhanced feature matrix for Random Forest.
    Includes all Linear Regression features plus:
        - Lagged indicator values (t-1, t-2, t-5)
        - Day of week encoding
        - Distance from 52-week high and low
        - Price acceleration (rate of change of returns)
        - RSI divergence (RSI trend vs price trend)
        - Volume surge indicator
    """
    features = pd.DataFrame(index=df.index)
    close    = df["Close"]
    volume   = df["Volume"]

    # ── Core indicators ───────────────────────
    if "RSI" in df.columns:
        features["rsi_norm"]      = df["RSI"] / 100.0
        features["rsi_overbought"] = (df["RSI"] > 70).astype(int)
        features["rsi_oversold"]   = (df["RSI"] < 30).astype(int)

    if "MACD" in df.columns and "MACD_Signal" in df.columns:
        features["macd_hist"]      = df["MACD"] - df["MACD_Signal"]
        features["macd_norm"]      = df["MACD"] / close.replace(0, np.nan)
        features["macd_cross_up"]  = (
            (df["MACD"] > df["MACD_Signal"]) &
            (df["MACD"].shift(1) <= df["MACD_Signal"].shift(1))
        ).astype(int)
        features["macd_cross_down"] = (
            (df["MACD"] < df["MACD_Signal"]) &
            (df["MACD"].shift(1) >= df["MACD_Signal"].shift(1))
        ).astype(int)

    if "BB_Upper" in df.columns and "BB_Lower" in df.columns:
        bb_range               = (df["BB_Upper"] - df["BB_Lower"]).replace(0, np.nan)
        features["bb_position"] = (close - df["BB_Lower"]) / bb_range
        features["bb_squeeze"]  = bb_range / close  # low = squeeze (breakout likely)
        features["bb_above"]    = (close > df["BB_Upper"]).astype(int)
        features["bb_below"]    = (close < df["BB_Lower"]).astype(int)

    if "SMA_20" in df.columns:
        features["above_sma20"]        = (close > df["SMA_20"]).astype(int)
        features["price_sma20_ratio"]  = close / df["SMA_20"].replace(0, np.nan)

    if "SMA_50" in df.columns:
        features["above_sma50"]        = (close > df["SMA_50"]).astype(int)
        features["price_sma50_ratio"]  = close / df["SMA_50"].replace(0, np.nan)

    if "SMA_20" in df.columns and "SMA_50" in df.columns:
        features["sma_ratio"]          = df["SMA_20"] / df["SMA_50"].replace(0, np.nan)
        features["golden_cross"]       = (df["SMA_20"] > df["SMA_50"]).astype(int)

    # ── Price momentum ────────────────────────
    features["return_1d"]  = close.pct_change(1)
    features["return_3d"]  = close.pct_change(3)
    features["return_5d"]  = close.pct_change(5)
    features["return_10d"] = close.pct_change(10)
    features["return_20d"] = close.pct_change(20)

    # Price acceleration (is momentum speeding up or slowing down?)
    features["momentum_accel"] = features["return_5d"] - features["return_10d"]

    # ── Volatility ────────────────────────────
    features["volatility_5d"]  = features["return_1d"].rolling(5).std()
    features["volatility_20d"] = features["return_1d"].rolling(20).std()
    features["vol_ratio"]      = features["volatility_5d"] / features["volatility_20d"].replace(0, np.nan)

    # ── Volume features ───────────────────────
    vol_ma20               = volume.rolling(20).mean().replace(0, np.nan)
    features["vol_ratio_20"] = volume / vol_ma20
    features["vol_surge"]    = (features["vol_ratio_20"] > 2.0).astype(int)
    features["vol_trend"]    = volume.pct_change(5)

    # ── 52-week high/low distance ─────────────
    high_52w               = close.rolling(252).max()
    low_52w                = close.rolling(252).min()
    features["dist_52w_high"] = (close - high_52w) / high_52w.replace(0, np.nan)
    features["dist_52w_low"]  = (close - low_52w)  / low_52w.replace(0, np.nan)

    # ── Day of week (markets behave differently) ──
    features["day_mon"] = (pd.to_datetime(df.index).dayofweek == 0).astype(int)
    features["day_fri"] = (pd.to_datetime(df.index).dayofweek == 4).astype(int)

    # ── Lagged features (yesterday's values) ──
    for col in ["rsi_norm", "macd_hist", "bb_position", "return_1d", "vol_ratio_20"]:
        if col in features.columns:
            features[f"{col}_lag1"] = features[col].shift(1)
            features[f"{col}_lag2"] = features[col].shift(2)

    # ── RSI divergence ────────────────────────
    # Price making new highs but RSI not = bearish divergence
    if "rsi_norm" in features.columns:
        price_higher = (close > close.shift(5)).astype(int)
        rsi_higher   = (features["rsi_norm"] > features["rsi_norm"].shift(5)).astype(int)
        features["rsi_divergence"] = price_higher - rsi_higher

    # ── Target ───────────────────────────────
    features["target"] = (close.shift(-1) > close).astype(int)

    # Drop NaN rows
    features = features.dropna()

    return features


# ─────────────────────────────────────────────
# 2. Train Random Forest Model
# ─────────────────────────────────────────────

def train_random_forest(features: pd.DataFrame) -> dict:
    """
    Trains a Random Forest classifier using TimeSeriesSplit.

    Random Forest parameters:
        n_estimators=200  — 200 decision trees in the ensemble
        max_depth=6       — prevents overfitting on financial data
        min_samples_leaf=10 — requires meaningful sample size per leaf
        class_weight=balanced — handles class imbalance

    Returns trained model, scaler, metrics, and feature importance.
    """
    if not SKLEARN_AVAILABLE:
        return {"error": "scikit-learn not installed. Run: pip install scikit-learn"}

    if len(features) < 60:
        return {"error": f"Not enough data ({len(features)} rows). Need at least 60."}

    feature_cols = [c for c in features.columns if c != "target"]
    X = features[feature_cols].values
    y = features["target"].values

    tscv   = TimeSeriesSplit(n_splits=5)
    scores = []

    scaler = StandardScaler()
    model  = RandomForestClassifier(
        n_estimators=200,
        max_depth=6,
        min_samples_leaf=10,
        max_features="sqrt",
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )

    for train_idx, test_idx in tscv.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled  = scaler.transform(X_test)

        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        scores.append(accuracy_score(y_test, y_pred))

    # Final model on all data
    X_scaled = scaler.fit_transform(X)
    model.fit(X_scaled, y)

    # Feature importance (Random Forest gives true importance, not just coefficients)
    importance = dict(zip(feature_cols, model.feature_importances_))
    importance = dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))

    avg_accuracy = np.mean(scores)
    logger.info(f"Random Forest trained — CV accuracy: {avg_accuracy:.1%} (±{np.std(scores):.1%})")

    return {
        "model":              model,
        "scaler":             scaler,
        "feature_cols":       feature_cols,
        "cv_accuracy":        round(avg_accuracy * 100, 1),
        "cv_scores":          [round(s * 100, 1) for s in scores],
        "cv_std":             round(np.std(scores) * 100, 1),
        "feature_importance": importance,
        "baseline_accuracy":  round(max(y.mean(), 1 - y.mean()) * 100, 1),
        "model_type":         "Random Forest",
        "n_estimators":       200,
    }


# ─────────────────────────────────────────────
# 3. Make Prediction
# ─────────────────────────────────────────────

def predict_next_day_rf(model_result: dict, features: pd.DataFrame) -> dict:
    """
    Uses the trained Random Forest to predict tomorrow's direction.
    Same output format as Linear Regression for easy comparison.
    """
    if "error" in model_result:
        return model_result

    model        = model_result["model"]
    scaler       = model_result["scaler"]
    feature_cols = model_result["feature_cols"]

    latest        = features[feature_cols].iloc[-1:].values
    latest_scaled = scaler.transform(latest)

    prediction  = model.predict(latest_scaled)[0]
    probability = model.predict_proba(latest_scaled)[0]

    up_prob   = round(float(probability[1]) * 100, 1)
    down_prob = round(float(probability[0]) * 100, 1)
    direction = "UP" if prediction == 1 else "DOWN"

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
        "model_type":  "Random Forest",
    }


# ─────────────────────────────────────────────
# 4. Full Pipeline
# ─────────────────────────────────────────────

def run_prediction_rf(ticker: str, period: str = "1y", regime: dict = None) -> dict:
    """
    Runs the full Random Forest prediction pipeline for a single ticker.
    """
    logger.info(f"Running Random Forest prediction for {ticker}...")

    # Fetch price data
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

    # Build enhanced features
    features = build_features_rf(df)
    if len(features) < 60:
        return {"error": f"Not enough data ({len(features)} rows). Use 1 year or more.", "ticker": ticker}

    # Train model
    model_result = train_random_forest(features)
    if "error" in model_result:
        return {**model_result, "ticker": ticker}

    # Predict
    prediction = predict_next_day_rf(model_result, features)

    # Apply volatility regime adjustment
    if regime is None:
        from prediction_lr import get_market_regime
        regime = get_market_regime()

    from prediction_lr import adjust_confidence_for_regime
    prediction = adjust_confidence_for_regime(prediction, regime)

    # Price info
    current_price    = round(float(df["Close"].iloc[-1]), 2)
    prev_price       = round(float(df["Close"].iloc[-2]), 2)
    price_change     = round(current_price - prev_price, 2)
    price_change_pct = round((price_change / prev_price) * 100, 2)

    result = {
        "ticker":             ticker,
        "current_price":      current_price,
        "price_change":       price_change,
        "price_change_pct":   price_change_pct,
        "prediction":         prediction,
        "regime":             regime,
        "model_accuracy":     model_result["cv_accuracy"],
        "model_std":          model_result["cv_std"],
        "baseline_accuracy":  model_result["baseline_accuracy"],
        "cv_scores":          model_result["cv_scores"],
        "feature_importance": model_result["feature_importance"],
        "data_points":        len(features),
        "period":             period,
        "model_type":         "Random Forest",
        "as_of":              datetime.utcnow().strftime("%Y-%m-%d"),
    }

    logger.info(
        f"{ticker} RF: {prediction['direction']} ({prediction['confidence']:.0f}% adj confidence) "
        f"| Accuracy: {model_result['cv_accuracy']}% ± {model_result['cv_std']}%"
    )

    return result


def run_predictions_rf_watchlist(tickers: list[str], period: str = "1y") -> dict[str, dict]:
    """Runs Random Forest predictions for all tickers. Fetches regime once."""
    from prediction_lr import get_market_regime
    logger.info("Fetching market regime for Random Forest predictions...")
    regime = get_market_regime()
    return {
        ticker: run_prediction_rf(ticker, period=period, regime=regime)
        for ticker in tickers
    }


# ─────────────────────────────────────────────
# 5. Compare LR vs RF
# ─────────────────────────────────────────────

def compare_models(ticker: str, period: str = "1y") -> dict:
    """
    Runs both Linear Regression and Random Forest on the same ticker
    and returns a side-by-side comparison.
    Useful for seeing which model is more confident and where they agree.
    """
    from prediction_lr import run_prediction, get_market_regime

    regime = get_market_regime()

    lr_result = run_prediction(ticker, period=period, regime=regime)
    rf_result = run_prediction_rf(ticker, period=period, regime=regime)

    if "error" in lr_result or "error" in rf_result:
        return {
            "ticker": ticker,
            "error":  lr_result.get("error") or rf_result.get("error"),
        }

    lr_pred = lr_result["prediction"]
    rf_pred = rf_result["prediction"]

    # Agreement signal — strongest when both models agree
    agree = lr_pred["direction"] == rf_pred["direction"]
    if agree:
        combined_confidence = round((lr_pred["confidence"] + rf_pred["confidence"]) / 2, 1)
        combined_signal     = rf_pred["signal"]  # RF signal when they agree
    else:
        combined_confidence = 50.0
        combined_signal     = "HOLD"  # Disagreement = hold

    return {
        "ticker":               ticker,
        "regime":               regime,
        "lr_direction":         lr_pred["direction"],
        "lr_confidence":        lr_pred["confidence"],
        "lr_signal":            lr_pred["signal"],
        "lr_accuracy":          lr_result["model_accuracy"],
        "rf_direction":         rf_pred["direction"],
        "rf_confidence":        rf_pred["confidence"],
        "rf_signal":            rf_pred["signal"],
        "rf_accuracy":          rf_result["model_accuracy"],
        "models_agree":         agree,
        "combined_signal":      combined_signal,
        "combined_confidence":  combined_confidence,
        "current_price":        rf_result["current_price"],
        "price_change_pct":     rf_result["price_change_pct"],
        "top_features":         dict(list(rf_result["feature_importance"].items())[:10]),
    }


def compare_models_watchlist(tickers: list[str], period: str = "1y") -> dict[str, dict]:
    """Runs model comparison for all tickers."""
    return {ticker: compare_models(ticker, period=period) for ticker in tickers}


# ─────────────────────────────────────────────
# 6. Demo
# ─────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 65)
    print("Random Forest Prediction Model — Demo")
    print("DISCLAIMER: Educational use only. Not financial advice.")
    print("=" * 65)

    tickers = ["AAPL", "MSFT", "NVDA"]

    print("\n── Model Comparison (LR vs RF) ─────────────────────────")
    for ticker in tickers:
        result = compare_models(ticker, period="1y")
        if "error" in result:
            print(f"\n{ticker}: ERROR — {result['error']}")
            continue

        agree  = "✓ AGREE" if result["models_agree"] else "✗ DISAGREE"
        print(f"\n── {ticker} ────────────────────────────────────────")
        print(f"  Price          : ${result['current_price']} ({result['price_change_pct']:+.2f}%)")
        print(f"  LR prediction  : {result['lr_direction']} ({result['lr_confidence']:.0f}%) — {result['lr_signal']}")
        print(f"  RF prediction  : {result['rf_direction']} ({result['rf_confidence']:.0f}%) — {result['rf_signal']}")
        print(f"  Models         : {agree}")
        print(f"  Combined signal: {result['combined_signal']} ({result['combined_confidence']:.0f}% confidence)")
        print(f"  Regime         : {result['regime']['regime']}")
