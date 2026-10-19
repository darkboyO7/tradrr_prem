"""Trading Intelligence Platform - Volatility Feature Engine.

Calculates Average True Range (ATR), normalized ATR, rolling return volatility,
Parkinson & Garman-Klass estimators, Bollinger Bands, and Volatility Regime categorization.
"""

import numpy as np
import pandas as pd


def compute_atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """Calculate Average True Range using Wilder's smoothing."""
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    return atr.bfill().fillna(0.0)


def compute_parkinson_volatility(high: pd.Series, low: pd.Series, window: int = 20) -> pd.Series:
    """Calculate Parkinson high-low volatility estimator."""
    hl_ratio = np.log(high / low.replace(0.0, 1e-8)) ** 2
    factor = 1.0 / (4.0 * np.log(2.0))
    parkinson = np.sqrt(factor * hl_ratio.rolling(window=window, min_periods=1).mean())
    return parkinson.fillna(0.0)


def compute_garman_klass_volatility(
    open_: pd.Series,
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    window: int = 20,
) -> pd.Series:
    """Calculate Garman-Klass volatility estimator incorporating OHLC."""
    hl = 0.5 * (np.log(high / low.replace(0.0, 1e-8)) ** 2)
    co = (2.0 * np.log(2.0) - 1.0) * (np.log(close / open_.replace(0.0, 1e-8)) ** 2)
    gk = np.sqrt((hl - co).rolling(window=window, min_periods=1).mean().clip(lower=0.0))
    return gk.fillna(0.0)


def compute_volatility_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute comprehensive volatility metrics and regime classifications."""
    feats = pd.DataFrame(index=df.index)
    c = df["close"]
    o = df["open"]
    h = df["high"]
    l = df["low"]

    # 1. ATR (14 and 28 period)
    atr_14 = compute_atr(h, l, c, period=14)
    atr_28 = compute_atr(h, l, c, period=28)
    feats["atr_14"] = atr_14
    feats["atr_28"] = atr_28
    feats["natr_14"] = (atr_14 / c.replace(0.0, 1e-8)) * 100.0  # Percentage ATR

    # 2. Rolling standard deviation of returns
    log_ret = np.log(c / c.shift(1)).fillna(0.0)
    for w in [10, 20, 50]:
        feats[f"rolling_vol_{w}"] = log_ret.rolling(window=w, min_periods=1).std().fillna(0.0)

    # 3. High-Low Parkinson & Garman-Klass estimators
    feats["parkinson_vol_20"] = compute_parkinson_volatility(h, l, window=20)
    feats["garman_klass_vol_20"] = compute_garman_klass_volatility(o, h, l, c, window=20)

    # 4. Bollinger Bands (20, 2)
    sma_20 = c.rolling(window=20, min_periods=1).mean()
    std_20 = c.rolling(window=20, min_periods=1).std().fillna(0.0)
    bb_upper = sma_20 + 2.0 * std_20
    bb_lower = sma_20 - 2.0 * std_20
    bb_bandwidth = (bb_upper - bb_lower) / sma_20.replace(0.0, 1e-8)
    bb_pct_b = (c - bb_lower) / (bb_upper - bb_lower).replace(0.0, 1e-8)

    feats["bb_upper_20"] = bb_upper
    feats["bb_lower_20"] = bb_lower
    feats["bb_bandwidth_20"] = bb_bandwidth
    feats["bb_pct_b_20"] = bb_pct_b

    # Squeeze: Bandwidth in the lowest 20th percentile of last 50 bars
    feats["bb_squeeze"] = np.where(
        bb_bandwidth <= bb_bandwidth.rolling(50, min_periods=1).quantile(0.20), 1, 0
    )

    # 5. Volatility Regime
    # Compare short-term ATR (14) against long-term baseline ATR (100)
    atr_long = compute_atr(h, l, c, period=100)
    vol_ratio = (atr_14 / atr_long.replace(0.0, 1e-8)).fillna(1.0)
    feats["volatility_ratio"] = vol_ratio

    # Regime encoding: 0 = Low, 1 = Normal, 2 = High, 3 = Extreme
    regime = np.where(
        vol_ratio < 0.75, 0,
        np.where(vol_ratio < 1.25, 1,
        np.where(vol_ratio < 2.0, 2, 3))
    )
    feats["volatility_regime"] = regime

    return feats
