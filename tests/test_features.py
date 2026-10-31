"""Tests for Feature Engineering Engines and Multi-Timeframe Feature Pipeline."""

from datetime import UTC, datetime, timedelta

import numpy as np
import pandas as pd

from features import (
    FeaturePipeline,
    compute_market_structure,
    compute_momentum_features,
    compute_price_features,
    compute_trend_features,
    compute_volatility_features,
    compute_volume_features,
)


def generate_synthetic_ohlcv(bars: int = 150) -> pd.DataFrame:
    """Generate realistic synthetic OHLCV data for feature testing."""
    np.random.seed(42)
    start_time = datetime(2026, 1, 1, 0, 0, 0, tzinfo=UTC)
    timestamps = [start_time + timedelta(minutes=i * 5) for i in range(bars)]

    # Geometric random walk
    returns = np.random.normal(loc=0.0002, scale=0.005, size=bars)
    price = 100.0 * np.exp(np.cumsum(returns))

    opens = price * (1 + np.random.uniform(-0.001, 0.001, bars))
    highs = np.maximum(opens, price) * (1 + np.random.uniform(0.0005, 0.004, bars))
    lows = np.minimum(opens, price) * (1 - np.random.uniform(0.0005, 0.004, bars))
    closes = price
    volumes = np.random.uniform(10.0, 500.0, bars)

    df = pd.DataFrame({
        "timestamp": timestamps,
        "open": opens,
        "high": highs,
        "low": lows,
        "close": closes,
        "volume": volumes,
        "quote_volume": closes * volumes,
        "trade_count": np.random.randint(5, 50, bars),
    })
    return df


def test_price_features():
    df = generate_synthetic_ohlcv(50)
    feats = compute_price_features(df)

    assert "return_1" in feats.columns
    assert "log_return_1" in feats.columns
    assert "candle_body" in feats.columns
    assert "candle_range" in feats.columns
    assert "upper_wick" in feats.columns
    assert "lower_wick" in feats.columns
    assert "candle_direction" in feats.columns

    # High - Low must be >= 0
    assert (feats["candle_range"] >= 0).all()
    # Body ratio must be between 0 and 1
    assert (feats["candle_body_ratio"] >= 0).all()
    assert (feats["candle_body_ratio"] <= 1.0001).all()


def test_trend_features():
    df = generate_synthetic_ohlcv(100)
    feats = compute_trend_features(df)

    assert "sma_10" in feats.columns
    assert "sma_50" in feats.columns
    assert "ema_9" in feats.columns
    assert "ema_20" in feats.columns
    assert "ema_bullish_stack" in feats.columns
    assert "plus_di_14" in feats.columns
    assert "minus_di_14" in feats.columns
    assert "adx_14" in feats.columns
    assert "trend_structure_score" in feats.columns

    # ADX must be non-negative
    assert (feats["adx_14"] >= 0).all()


def test_momentum_features():
    df = generate_synthetic_ohlcv(60)
    feats = compute_momentum_features(df)

    assert "rsi_14" in feats.columns
    assert "macd_line" in feats.columns
    assert "macd_signal" in feats.columns
    assert "macd_hist" in feats.columns
    assert "roc_10" in feats.columns
    assert "stoch_k_14" in feats.columns
    assert "stoch_d_14" in feats.columns

    # RSI must be strictly bounded between 0 and 100
    assert (feats["rsi_14"] >= 0.0).all()
    assert (feats["rsi_14"] <= 100.0).all()

    # Stochastic %K must be bounded between 0 and 100
    assert (feats["stoch_k_14"] >= 0.0).all()
    assert (feats["stoch_k_14"] <= 100.0).all()


def test_volatility_features():
    df = generate_synthetic_ohlcv(80)
    feats = compute_volatility_features(df)

    assert "atr_14" in feats.columns
    assert "natr_14" in feats.columns
    assert "rolling_vol_20" in feats.columns
    assert "parkinson_vol_20" in feats.columns
    assert "bb_upper_20" in feats.columns
    assert "bb_lower_20" in feats.columns
    assert "bb_squeeze" in feats.columns
    assert "volatility_regime" in feats.columns

    # ATR must be strictly positive
    assert (feats["atr_14"] > 0).all()
    # Upper band must be >= Lower band
    assert (feats["bb_upper_20"] >= feats["bb_lower_20"]).all()
    # Volatility regime must be in {0, 1, 2, 3}
    assert set(feats["volatility_regime"].unique()).issubset({0, 1, 2, 3})


def test_volume_features():
    df = generate_synthetic_ohlcv(60)
    feats = compute_volume_features(df)

    assert "volume_sma_20" in feats.columns
    assert "rvol_20" in feats.columns
    assert "volume_spike_2x" in feats.columns
    assert "obv" in feats.columns
    assert "vwap_dist_pct" in feats.columns

    # RVOL must be >= 0
    assert (feats["rvol_20"] >= 0).all()


def test_market_structure_features():
    df = generate_synthetic_ohlcv(60)
    feats = compute_market_structure(df, lookback=20)

    assert "resistance_20" in feats.columns
    assert "support_20" in feats.columns
    assert "breakout_20" in feats.columns
    assert "breakdown_20" in feats.columns
    assert "range_position_20" in feats.columns

    # Range position must be in [0, 1]
    valid_pos = feats["range_position_20"].dropna()
    assert (valid_pos >= 0.0).all()
    assert (valid_pos <= 1.0).all()


def test_feature_pipeline_all_features():
    df = generate_synthetic_ohlcv(120)
    pipeline = FeaturePipeline()

    full_features = pipeline.compute_all_features(df)
    assert len(full_features) == 120
    # Original columns present
    assert "close" in full_features.columns
    # Check diverse features exist
    assert "return_1" in full_features.columns
    assert "ema_50" in full_features.columns
    assert "rsi_14" in full_features.columns
    assert "atr_14" in full_features.columns
    assert "rvol_20" in full_features.columns
    assert "resistance_20" in full_features.columns


def test_multi_timeframe_feature_merger_no_lookahead():
    """Verify that HTF features are merged strictly after the HTF bar closes."""
    pipeline = FeaturePipeline()

    # Base 5m bars: 12:00, 12:05, ..., 13:30 (19 bars)
    ltf_timestamps = pd.date_range("2026-01-01 12:00:00", "2026-01-01 13:30:00", freq="5min", tz="UTC")
    ltf_df = pd.DataFrame({
        "timestamp": ltf_timestamps,
        "close": [100.0 + i for i in range(len(ltf_timestamps))],
    })

    # Higher timeframe 1h bars:
    # Bar 1 starts at 11:00 (closes at 12:00) with rsi_14 = 40.0
    # Bar 2 starts at 12:00 (closes at 13:00) with rsi_14 = 75.0
    htf_timestamps = pd.date_range("2026-01-01 11:00:00", "2026-01-01 12:00:00", freq="1h", tz="UTC")
    htf_1h_feats = pd.DataFrame({
        "timestamp": htf_timestamps,
        "rsi_14": [40.0, 75.0],
    })

    merged = pipeline.merge_multi_timeframe_features(ltf_df, {"1h": htf_1h_feats})

    assert "htf_1h_rsi_14" in merged.columns

    # Lookahead test:
    # At 12:00, 12:05, ..., 12:55 (before 13:00):
    # The 12:00 1h bar is NOT yet closed!
    # The system must ONLY see Bar 1 (which started at 11:00, ended at 12:00, rsi=40.0)!
    row_12_00 = merged[merged["timestamp"] == pd.Timestamp("2026-01-01 12:00:00", tz="UTC")].iloc[0]
    row_12_30 = merged[merged["timestamp"] == pd.Timestamp("2026-01-01 12:30:00", tz="UTC")].iloc[0]
    row_12_55 = merged[merged["timestamp"] == pd.Timestamp("2026-01-01 12:55:00", tz="UTC")].iloc[0]

    assert row_12_00["htf_1h_rsi_14"] == 40.0
    assert row_12_30["htf_1h_rsi_14"] == 40.0
    assert row_12_55["htf_1h_rsi_14"] == 40.0

    # At 13:00:
    # The 12:00-13:00 1h bar completes, so the new RSI of 75.0 becomes available!
    row_13_00 = merged[merged["timestamp"] == pd.Timestamp("2026-01-01 13:00:00", tz="UTC")].iloc[0]
    row_13_05 = merged[merged["timestamp"] == pd.Timestamp("2026-01-01 13:05:00", tz="UTC")].iloc[0]

    assert row_13_00["htf_1h_rsi_14"] == 75.0
    assert row_13_05["htf_1h_rsi_14"] == 75.0


def test_incremental_feature_calculator_parity():
    """Verify incremental single-bar calculation matches full batch pipeline."""
    from features.incremental import IncrementalFeatureCalculator

    df = generate_synthetic_ohlcv(100)
    pipe = FeaturePipeline()
    batch_feats = pipe.compute_all_features(df)

    penultimate_row = batch_feats.iloc[-2]
    active_bar = df.iloc[-1]
    price = float(active_bar["close"])

    inc_dict = IncrementalFeatureCalculator.update_active_bar(
        price=price,
        high=float(active_bar["high"]),
        low=float(active_bar["low"]),
        open_price=float(active_bar["open"]),
        volume=float(active_bar["volume"]),
        history_df=df,
        penultimate_row=penultimate_row,
    )

    actual_row = batch_feats.iloc[-1]

    # Verify key indicator parity within strict tolerance
    assert abs(inc_dict["return_1"] - float(actual_row["return_1"])) < 1e-6
    assert abs(inc_dict["ema_9"] - float(actual_row["ema_9"])) < 1e-4
    assert abs(inc_dict["ema_20"] - float(actual_row["ema_20"])) < 1e-4
    assert abs(inc_dict["ema_50"] - float(actual_row["ema_50"])) < 1e-4
    assert abs(inc_dict["atr_14"] - float(actual_row["atr_14"])) < 1e-4
    assert abs(inc_dict["candle_range_pct"] - float(actual_row["candle_range_pct"])) < 1e-6
    assert abs(inc_dict["candle_body_ratio"] - float(actual_row["candle_body_ratio"])) < 1e-6

