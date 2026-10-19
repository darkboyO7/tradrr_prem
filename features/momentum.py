"""Trading Intelligence Platform - Momentum Feature Engine.

Calculates Relative Strength Index (RSI), MACD, Rate of Change (ROC),
Stochastic Oscillator (%K, %D), and Williams %R.
"""

import numpy as np
import pandas as pd


def compute_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Calculate Relative Strength Index using Wilder's exponential smoothing."""
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)

    avg_gain = gain.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()

    rs = avg_gain / avg_loss.replace(0.0, 1e-8)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi.fillna(50.0)


def compute_stochastic(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    k_period: int = 14,
    d_period: int = 3,
) -> tuple[pd.Series, pd.Series]:
    """Calculate Stochastic Oscillator %K and %D."""
    lowest_low = low.rolling(window=k_period, min_periods=1).min()
    highest_high = high.rolling(window=k_period, min_periods=1).max()
    denom = (highest_high - lowest_low).replace(0.0, 1e-8)

    fast_k = 100.0 * (close - lowest_low) / denom
    fast_d = fast_k.rolling(window=d_period, min_periods=1).mean()
    return fast_k, fast_d


def compute_momentum_features(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate momentum indicators: RSI, MACD, ROC, Stochastic, and Williams %R."""
    feats = pd.DataFrame(index=df.index)
    c = df["close"]
    h = df["high"]
    l = df["low"]

    # 1. RSI across multiple lookbacks
    for p in [7, 14, 21]:
        feats[f"rsi_{p}"] = compute_rsi(c, period=p)
        feats[f"rsi_{p}_overbought"] = np.where(feats[f"rsi_{p}"] >= 70.0, 1, 0)
        feats[f"rsi_{p}_oversold"] = np.where(feats[f"rsi_{p}"] <= 30.0, 1, 0)

    # 2. MACD (12, 26, 9)
    fast_ema = c.ewm(span=12, min_periods=1, adjust=False).mean()
    slow_ema = c.ewm(span=26, min_periods=1, adjust=False).mean()
    macd_line = fast_ema - slow_ema
    macd_signal = macd_line.ewm(span=9, min_periods=1, adjust=False).mean()
    macd_hist = macd_line - macd_signal

    feats["macd_line"] = macd_line
    feats["macd_signal"] = macd_signal
    feats["macd_hist"] = macd_hist
    feats["macd_hist_slope_3"] = macd_hist - macd_hist.shift(3).fillna(0.0)

    # 3. Rate of Change (ROC)
    for p in [5, 10, 20]:
        feats[f"roc_{p}"] = ((c - c.shift(p)) / c.shift(p).replace(0.0, 1e-8)) * 100.0
        feats[f"roc_{p}"] = feats[f"roc_{p}"].fillna(0.0)

    # 4. Stochastic Oscillator (14, 3)
    k_14, d_14 = compute_stochastic(h, l, c, k_period=14, d_period=3)
    feats["stoch_k_14"] = k_14
    feats["stoch_d_14"] = d_14

    # 5. Williams %R (14)
    highest_h14 = h.rolling(window=14, min_periods=1).max()
    lowest_l14 = l.rolling(window=14, min_periods=1).min()
    feats["williams_r_14"] = -100.0 * (highest_h14 - c) / (highest_h14 - lowest_l14).replace(0.0, 1e-8)

    return feats
