"""Trading Intelligence Platform - Market Microstructure Feature Engine.

Calculates high-frequency order book imbalance, micro-price premiums,
spread expansion ratios, and liquidity vacuum warning flags.
"""

from typing import Dict, List, Optional
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
    out = pd.DataFrame(index=df.index)

    # 1. Spread and Basis Points
    if "best_bid" in df.columns and "best_ask" in df.columns:
        bb = df["best_bid"]
        ba = df["best_ask"]
        mid = (ba + bb) / 2.0
        spread = ba - bb
        out["mid_price"] = mid
        out["spread"] = spread
        out["spread_bps"] = np.where(mid > 0, (spread / mid) * 10000.0, 0.0)
    elif "spread_bps" in df.columns:
        out["spread_bps"] = df["spread_bps"]
    else:
        out["spread_bps"] = 5.0  # default baseline 5 bps

    # 2. Order Book Imbalance (OBI)
    if "bid_depth_top5" in df.columns and "ask_depth_top5" in df.columns:
        b_qty = df["bid_depth_top5"]
        a_qty = df["ask_depth_top5"]
        total_d = b_qty + a_qty
        out["obi_top5"] = np.where(total_d > 0, (b_qty - a_qty) / total_d, 0.0)
        out["depth_ratio"] = np.where(a_qty > 0, b_qty / a_qty, 1.0)
    elif "order_book_imbalance" in df.columns:
        out["obi_top5"] = df["order_book_imbalance"]
        out["depth_ratio"] = 1.0
    else:
        out["obi_top5"] = 0.0
        out["depth_ratio"] = 1.0

    # 3. Micro-Price Premium in Basis Points
    if "micro_price" in df.columns and "mid_price" in df.columns:
        mp = df["micro_price"]
        mid = df["mid_price"]
        out["micro_price_bps"] = np.where(mid > 0, ((mp - mid) / mid) * 10000.0, 0.0)
    else:
        out["micro_price_bps"] = out["obi_top5"] * (out["spread_bps"] * 0.5)

    # 4. Spread Expansion & Liquidity Vacuum Flag
    rolling_median_spread = out["spread_bps"].rolling(20, min_periods=1).median()
    out["spread_expansion_ratio"] = np.where(
        rolling_median_spread > 0, out["spread_bps"] / rolling_median_spread, 1.0
    )
    # Liquidity vacuum occurs when spread blows out > 2.5x normal or exceeds 35 bps
    out["liquidity_vacuum"] = (
        (out["spread_expansion_ratio"] >= 2.5) | (out["spread_bps"] >= 35.0)
    ).astype(int)

    return out
