"""Trading Intelligence Platform - Price & Return Feature Engine.

Calculates returns, log returns, range dynamics, candle bodies, wicks,
gaps, and rolling cumulative returns.
"""

from typing import Any

import numpy as np
import pandas as pd


def compute_price_features(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate vectorized price, return, and candle geometry features.

    Expects columns: open, high, low, close, volume.
    Returns DataFrame containing engineered price feature columns.
    """
    c = df["close"]
    o = df["open"]
    h = df["high"]
    l = df["low"]
    prev_c = c.shift(1)

    data: dict[str, Any] = {}

    # 1. Simple returns & Log returns
    data["return_1"] = c.pct_change(1).fillna(0.0)
    data["log_return_1"] = np.log(c / prev_c).fillna(0.0)

    # 2. Rolling returns
    for window in [3, 5, 10, 20, 50]:
        data[f"return_{window}"] = c.pct_change(window).fillna(0.0)
        data[f"log_return_{window}"] = np.log(c / c.shift(window)).fillna(0.0)

    # 3. Candle geometry
    full_range = (h - l).replace(0.0, 1e-8)
    body = (c - o).abs()
    data["candle_range"] = h - l
    data["candle_range_pct"] = (h - l) / c
    data["candle_body"] = body
    data["candle_body_ratio"] = body / full_range

    # 4. Upper & lower wicks
    upper_wick = h - np.maximum(o, c)
    lower_wick = np.minimum(o, c) - l
    data["upper_wick"] = upper_wick
    data["lower_wick"] = lower_wick
    data["upper_wick_ratio"] = upper_wick / full_range
    data["lower_wick_ratio"] = lower_wick / full_range

    # 5. Directional candle polarity (1 = bullish, -1 = bearish, 0 = neutral)
    data["candle_direction"] = np.where(c > o, 1, np.where(c < o, -1, 0))

    # 6. Price gap between previous close and current open
    gap = o - prev_c
    gap_pct = gap / prev_c.replace(0.0, 1e-8)
    data["gap"] = gap.fillna(0.0)
    data["gap_pct"] = gap_pct.fillna(0.0)

    return pd.DataFrame(data, index=df.index)
