"""Trading Intelligence Platform - Market Structure & Level Detection Engine.

Detects pivot highs/lows, dynamic support & resistance levels, breakout / breakdown signals,
and consolidation ranges without future lookahead bias.
"""

import numpy as np
import pandas as pd


def compute_market_structure(df: pd.DataFrame, lookback: int = 20) -> pd.DataFrame:
    """Calculate support/resistance boundaries, breakouts, breakdowns, and consolidation metrics.

    Uses strictly past-window information (e.g. rolling max/min over past `lookback` bars shifted by 1)
    so that the current bar's close is compared against established historical boundaries with ZERO lookahead.
    """
    feats = pd.DataFrame(index=df.index)
    c = df["close"]
    h = df["high"]
    l = df["low"]

    # Rolling resistance (highest high of past `lookback` bars, shifted by 1 so current bar is excluded)
    resistance = h.shift(1).rolling(window=lookback, min_periods=lookback).max()
    # Rolling support (lowest low of past `lookback` bars, shifted by 1)
    support = l.shift(1).rolling(window=lookback, min_periods=lookback).min()

    feats[f"resistance_{lookback}"] = resistance
    feats[f"support_{lookback}"] = support

    # Normalized distance to levels
    feats[f"dist_to_resistance_pct_{lookback}"] = (resistance - c) / c.replace(0.0, 1e-8)
    feats[f"dist_to_support_pct_{lookback}"] = (c - support) / c.replace(0.0, 1e-8)

    # Breakout & Breakdown events (Current bar closes beyond the past window level)
    feats[f"breakout_{lookback}"] = np.where(c > resistance, 1, 0)
    feats[f"breakdown_{lookback}"] = np.where(c < support, 1, 0)

    # Range width of structure
    structure_range = (resistance - support).replace(0.0, 1e-8)
    feats[f"structure_range_pct_{lookback}"] = structure_range / c

    # Consolidation detection: Range width is in the lowest 25% of recent memory
    range_quantile = structure_range.rolling(50, min_periods=lookback).quantile(0.25)
    feats[f"is_consolidating_{lookback}"] = np.where(structure_range <= range_quantile, 1, 0)

    # Position within range (0 = at support, 1 = at resistance)
    range_pos = (c - support) / structure_range
    feats[f"range_position_{lookback}"] = range_pos.clip(lower=0.0, upper=1.0)

    return feats
