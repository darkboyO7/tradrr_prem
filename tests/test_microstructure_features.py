"""Tests for Market Microstructure and Order Book Imbalance Features."""

import numpy as np
import pandas as pd
import pytest

from features.microstructure import compute_microstructure_features
from features.pipeline import FeaturePipeline


def test_compute_microstructure_features_direct():
    """Verify calculation of OBI, spread basis points, and liquidity vacuum."""
    df = pd.DataFrame({
        "best_bid": [65000.0, 65000.0, 65000.0],
        "best_ask": [65005.0, 65010.0, 65300.0],  # Last row spread = 300 (blowout)
        "bid_depth_top5": [10.0, 2.0, 1.0],
        "ask_depth_top5": [2.0, 10.0, 1.0],
        "micro_price": [65004.0, 65002.0, 65150.0],
    })

    feats = compute_microstructure_features(df)

    assert "obi_top5" in feats.columns
    assert "spread_bps" in feats.columns
    assert "liquidity_vacuum" in feats.columns

    # Row 0: Heavy bid dominance -> OBI = (10 - 2) / 12 = 0.6667
    assert feats["obi_top5"].iloc[0] > 0.6
    # Row 1: Heavy ask dominance -> OBI = (2 - 10) / 12 = -0.6667
    assert feats["obi_top5"].iloc[1] < -0.6
    # Row 2: Spread blowout -> Liquidity vacuum flag triggered
    assert feats["liquidity_vacuum"].iloc[2] == 1


def test_feature_pipeline_with_microstructure():
    """Verify FeaturePipeline automatically integrates microstructure block when present."""
    pipeline = FeaturePipeline()
    df = pd.DataFrame({
        "timestamp": pd.date_range("2026-01-01", periods=30, freq="1min", tz="UTC"),
        "open": np.full(30, 65000.0),
        "high": np.full(30, 65100.0),
        "low": np.full(30, 64900.0),
        "close": np.full(30, 65050.0),
        "volume": np.full(30, 15.0),
        "order_book_imbalance": np.linspace(-0.5, 0.5, 30),
        "spread_bps": np.full(30, 4.5),
    })

    all_feats = pipeline.compute_all_features(df)
    assert "obi_top5" in all_feats.columns
    assert "spread_bps" in all_feats.columns
    assert "liquidity_vacuum" in all_feats.columns
