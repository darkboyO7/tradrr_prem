"""Tests for Candlestick Pattern Recognition and Historical Similarity Engine."""

import numpy as np
import pandas as pd

from analysis.patterns import CandlestickPatternDetector, HistoricalPatternSimilarityEngine


def test_bullish_engulfing_detection():
    # Bar 0: Red candle (open 105, close 100)
    # Bar 1: Green candle engulfing (open 99, close 106)
    df = pd.DataFrame({
        "open": [105.0, 99.0],
        "high": [106.0, 107.0],
        "low": [99.5, 98.5],
        "close": [100.0, 106.0],
        "volume": [10.0, 20.0],
    })

    scanned = CandlestickPatternDetector.scan_patterns(df)
    assert "pattern_bullish_engulfing" in scanned.columns
    assert scanned["pattern_bullish_engulfing"].iloc[1] == 1


def test_hammer_detection():
    # Hammer: small body at the top, long lower wick (>= 2x body)
    # Open 100, close 101 (body 1), high 101.5 (upper wick 0.5), low 97.0 (lower wick 3.0)
    df = pd.DataFrame({
        "open": [102.0, 100.0],
        "high": [103.0, 101.5],
        "low": [101.0, 97.0],
        "close": [101.5, 101.0],
        "volume": [10.0, 15.0],
    })

    scanned = CandlestickPatternDetector.scan_patterns(df)
    assert "pattern_hammer" in scanned.columns
    assert scanned["pattern_hammer"].iloc[1] == 1


def test_inside_bar_detection():
    # Bar 0: mother bar (high 110, low 90)
    # Bar 1: inside bar (high 105, low 95)
    df = pd.DataFrame({
        "open": [92.0, 96.0],
        "high": [110.0, 105.0],
        "low": [90.0, 95.0],
        "close": [108.0, 100.0],
        "volume": [50.0, 30.0],
    })

    scanned = CandlestickPatternDetector.scan_patterns(df)
    assert "pattern_inside_bar" in scanned.columns
    assert scanned["pattern_inside_bar"].iloc[1] == 1


def test_historical_pattern_similarity_search():
    np.random.seed(123)
    # Create 120 bars of synthetic data
    bars = 120
    returns = np.random.normal(loc=0.0005, scale=0.01, size=bars)
    # Inject a distinct specific shape at index 20..30 and replicate it at index 100..110
    shape = np.array([0.02, 0.015, -0.01, -0.02, 0.03, 0.02, -0.005, 0.01, 0.025, 0.015])
    returns[20:30] = shape
    returns[100:110] = shape

    closes = 100.0 * np.exp(np.cumsum(returns))
    highs = closes * 1.002
    lows = closes * 0.998
    opens = closes * 0.999
    volumes = [100.0] * bars

    df = pd.DataFrame({
        "close": closes,
        "open": opens,
        "high": highs,
        "low": lows,
        "volume": volumes,
        "return_1": pd.Series(closes).pct_change().fillna(0.0),
        "candle_range_pct": (highs - lows) / closes,
        "candle_body_ratio": np.abs(closes - opens) / (highs - lows),
    })

    engine = HistoricalPatternSimilarityEngine(feature_columns=["return_1", "candle_range_pct"])

    # Query last 10 bars
    matches = engine.find_similar_patterns(df, query_window=10, top_k=3, forward_horizons=[1, 3, 5])

    assert len(matches) > 0
    top = matches[0]
    assert top.similarity_score > 0.0
    assert 1 in top.forward_returns
    assert 5 in top.forward_returns

    consensus = engine.compute_similarity_consensus(matches, horizon=5)
    assert consensus["sample_size"] == len(matches)
    assert "win_rate" in consensus
    assert "direction" in consensus
