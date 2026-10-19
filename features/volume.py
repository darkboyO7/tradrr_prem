"""Trading Intelligence Platform - Volume & Liquidity Feature Engine.

Calculates volume moving averages, Relative Volume (RVOL), volume spikes,
volume trends, On-Balance Volume (OBV), and VWAP distance metrics.
"""

import numpy as np
import pandas as pd


def compute_obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    """Calculate cumulative On-Balance Volume (OBV)."""
    direction = np.sign(close.diff().fillna(0.0))
    obv = (direction * volume).cumsum()
    return obv


def compute_volume_features(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate volume dynamics, relative volume, spikes, and OBV indicators."""
    feats = pd.DataFrame(index=df.index)
    v = df["volume"]
    c = df["close"]

    # 1. Volume moving averages
    for w in [5, 10, 20, 50]:
        v_sma = v.rolling(window=w, min_periods=1).mean()
        feats[f"volume_sma_{w}"] = v_sma

    # 2. Relative Volume (RVOL)
    v_sma_20 = feats["volume_sma_20"].replace(0.0, 1e-8)
    rvol = v / v_sma_20
    feats["rvol_20"] = rvol

    # 3. Volume spikes
    feats["volume_spike_2x"] = np.where(rvol >= 2.0, 1, 0)
    feats["volume_spike_3x"] = np.where(rvol >= 3.0, 1, 0)

    # 4. Volume Trend (Ratio of short-term volume to medium-term volume)
    feats["volume_trend_5_20"] = (feats["volume_sma_5"] / v_sma_20).fillna(1.0)

    # 5. On-Balance Volume (OBV) & OBV EMA
    obv = compute_obv(c, v)
    feats["obv"] = obv
    feats["obv_ema_20"] = obv.ewm(span=20, min_periods=1, adjust=False).mean()
    feats["obv_slope_5"] = (obv - obv.shift(5)).fillna(0.0)

    # 6. VWAP distance (if vwap present in input DataFrame, else computed)
    if "vwap" in df.columns:
        vwap = df["vwap"]
    elif "quote_volume" in df.columns:
        vwap = (df["quote_volume"] / v.replace(0.0, 1e-8)).fillna(c)
    else:
        vwap = c

    feats["vwap_dist_pct"] = ((c - vwap) / vwap.replace(0.0, 1e-8)) * 100.0

    return feats
