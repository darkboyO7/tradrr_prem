"""Trading Intelligence Platform - Market Microstructure Feature Engine.

Calculates high-frequency order book imbalance, micro-price premiums,
spread expansion ratios, and liquidity vacuum warning flags.
"""

from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd


def compute_microstructure_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extract quantitative microstructure features from order book snapshots or tick records.

    Expected optional columns:
      - best_bid, best_ask
      - bid_depth_top5, ask_depth_top5
      - order_book_imbalance
      - micro_price, mid_price
      - spread, spread_bps
    """
    data: dict[str, Any] = {}

    # 1. Spread and Basis Points
    if "best_bid" in df.columns and "best_ask" in df.columns:
        bb = df["best_bid"]
        ba = df["best_ask"]
        mid = (ba + bb) / 2.0
        spread = ba - bb
        data["mid_price"] = mid
        data["spread"] = spread
        spread_bps = np.where(mid > 0, (spread / mid) * 10000.0, 0.0)
        data["spread_bps"] = spread_bps
    elif "spread_bps" in df.columns:
        spread_bps = df["spread_bps"]
        data["spread_bps"] = spread_bps
    else:
        spread_bps = pd.Series(5.0, index=df.index)  # default baseline 5 bps
        data["spread_bps"] = spread_bps

    # 2. Order Book Imbalance (OBI)
    if "bid_depth_top5" in df.columns and "ask_depth_top5" in df.columns:
        b_qty = df["bid_depth_top5"]
        a_qty = df["ask_depth_top5"]
        total_d = b_qty + a_qty
        obi = np.where(total_d > 0, (b_qty - a_qty) / total_d, 0.0)
        data["obi_top5"] = obi
        data["depth_ratio"] = np.where(a_qty > 0, b_qty / a_qty, 1.0)
    elif "order_book_imbalance" in df.columns:
        obi = df["order_book_imbalance"]
        data["obi_top5"] = obi
        data["depth_ratio"] = 1.0
    else:
        obi = pd.Series(0.0, index=df.index)
        data["obi_top5"] = obi
        data["depth_ratio"] = 1.0

    # 3. Micro-Price Premium in Basis Points
    if "micro_price" in df.columns and "mid_price" in df.columns:
        mp = df["micro_price"]
        mid = df["mid_price"]
        data["micro_price_bps"] = np.where(mid > 0, ((mp - mid) / mid) * 10000.0, 0.0)
    else:
        data["micro_price_bps"] = data["obi_top5"] * (data["spread_bps"] * 0.5)

    # 4. Spread Expansion & Liquidity Vacuum Flag
    s_bps_series = data["spread_bps"] if isinstance(data["spread_bps"], pd.Series) else pd.Series(data["spread_bps"], index=df.index)
    rolling_median_spread = s_bps_series.rolling(20, min_periods=1).median()
    expansion_ratio = np.where(
        rolling_median_spread > 0, s_bps_series / rolling_median_spread, 1.0
    )
    data["spread_expansion_ratio"] = expansion_ratio
    # Liquidity vacuum occurs when spread blows out > 2.5x normal or exceeds 35 bps
    data["liquidity_vacuum"] = (
        (expansion_ratio >= 2.5) | (s_bps_series >= 35.0)
    ).astype(int)

    return pd.DataFrame(data, index=df.index)
