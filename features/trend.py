"""Trading Intelligence Platform - Trend & Moving Average Feature Engine.

Calculates SMA, EMA (9, 20, 50, 100, 200), trend slopes, EMA stack alignment,
Average Directional Index (ADX / +DI / -DI), and swing structure (HH, HL, LH, LL).
"""

from typing import Any

import numpy as np
import pandas as pd


def compute_adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.DataFrame:
    """Calculate Average Directional Index (ADX) with +DI and -DI."""
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.Series(np.maximum(tr1.values, np.maximum(tr2.values, tr3.values)), index=close.index)

    up_move = high - high.shift(1)
    down_move = low.shift(1) - low

    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    plus_dm_s = pd.Series(plus_dm, index=high.index)
    minus_dm_s = pd.Series(minus_dm, index=high.index)

    # Wilder's Smoothing
    tr_smooth = tr.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    denom = tr_smooth.replace(0.0, 1e-8)
    plus_di = 100.0 * (plus_dm_s.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean() / denom)
    minus_di = 100.0 * (minus_dm_s.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean() / denom)

    di_sum = (plus_di + minus_di).replace(0.0, 1e-8)
    dx = 100.0 * (plus_di - minus_di).abs() / di_sum
    adx = dx.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()

    return pd.DataFrame({
        f"plus_di_{period}": plus_di.fillna(0.0),
        f"minus_di_{period}": minus_di.fillna(0.0),
        f"adx_{period}": adx.fillna(0.0),
    }, index=high.index)


def compute_swing_structure(high: pd.Series, low: pd.Series, lookback: int = 5) -> pd.DataFrame:
    """Classify market structure into Higher Highs, Higher Lows, Lower Highs, Lower Lows."""
    recent_high = high.rolling(window=lookback, min_periods=lookback).max()
    recent_low = low.rolling(window=lookback, min_periods=lookback).min()

    prev_high = recent_high.shift(lookback)
    prev_low = recent_low.shift(lookback)

    is_hh = np.where(recent_high > prev_high, 1, 0)
    is_hl = np.where(recent_low > prev_low, 1, 0)
    is_lh = np.where(recent_high < prev_high, 1, 0)
    is_ll = np.where(recent_low < prev_low, 1, 0)
    score = is_hh + is_hl - is_lh - is_ll

    return pd.DataFrame({
        "is_higher_high": is_hh,
        "is_higher_low": is_hl,
        "is_lower_high": is_lh,
        "is_lower_low": is_ll,
        "trend_structure_score": score,
    }, index=high.index)


def compute_trend_features(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate moving averages, slope, alignment, and directional indices."""
    data: dict[str, Any] = {}
    c = df["close"]
    h = df["high"]
    l = df["low"]

    # 1. Simple Moving Averages
    for w in [10, 20, 50, 100, 200]:
        sma = c.rolling(window=w, min_periods=1).mean()
        data[f"sma_{w}"] = sma
        data[f"dist_sma_{w}_pct"] = (c - sma) / sma

    # 2. Exponential Moving Averages
    for w in [9, 20, 50, 100, 200]:
        ema = c.ewm(span=w, min_periods=1, adjust=False).mean()
        data[f"ema_{w}"] = ema
        data[f"dist_ema_{w}_pct"] = (c - ema) / ema

    # 3. EMA Slopes (lookback 5 bars)
    for w in [9, 20, 50]:
        ema_series = data[f"ema_{w}"]
        ema_prev5 = ema_series.shift(5)
        slope = (ema_series - ema_prev5) / ema_prev5.replace(0.0, 1e-8)
        data[f"ema_{w}_slope_5"] = slope.fillna(0.0)

    # 4. Moving Average Alignment (Bullish Stack: EMA9 > EMA20 > EMA50)
    e9 = data["ema_9"]
    e20 = data["ema_20"]
    e50 = data["ema_50"]
    data["ema_bullish_stack"] = np.where((e9 > e20) & (e20 > e50), 1, 0)
    data["ema_bearish_stack"] = np.where((e9 < e20) & (e20 < e50), 1, 0)

    # 5. ADX, +DI, -DI
    adx_df = compute_adx(h, l, c, period=14)
    for col in adx_df.columns:
        data[col] = adx_df[col]

    # 6. Swing Structure
    swing_df = compute_swing_structure(h, l, lookback=5)
    for col in swing_df.columns:
        data[col] = swing_df[col]

    return pd.DataFrame(data, index=df.index)
