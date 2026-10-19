"""Tests for Deterministic Replay Engine and Zero-Drift Parity Verification."""

import numpy as np
import pandas as pd
import pytest

from backtesting.replay import DeterministicReplayEngine, ReplayParityReport


@pytest.fixture
def sample_ohlcv_dataframe():
    """Generates synthetic OHLCV bars with a clear trend to trigger signals and trades."""
    np.random.seed(42)
    bars = 100
    returns = np.random.normal(0.0008, 0.008, bars)
    # Inject a strong bull trend section
    returns[20:50] = 0.005
    closes = 50000.0 * np.exp(np.cumsum(returns))
    highs = closes * 1.005
    lows = closes * 0.995
    opens = closes * 0.999
    volumes = np.random.uniform(50, 200, bars)

    return pd.DataFrame({
        "timestamp": pd.date_range("2026-01-01", periods=bars, freq="1h", tz="UTC"),
        "open": opens,
        "high": highs,
        "low": lows,
        "close": closes,
        "volume": volumes,
    })


def test_deterministic_replay_zero_drift(sample_ohlcv_dataframe):
    engine = DeterministicReplayEngine(initial_capital=100000.0, seed=42)
    report = engine.execute_parity_replay(
        df=sample_ohlcv_dataframe,
        symbol="BTC-USD",
        timeframe="1h",
        tolerance_bps=100.0,
    )

    assert isinstance(report, ReplayParityReport)
    assert report.total_bars == 100
    assert report.zero_lookahead_verified is True
    assert report.parity_verified is True
    assert report.drift_bps <= 100.0


def test_deterministic_replay_insufficient_bars():
    short_df = pd.DataFrame({
        "open": [100.0] * 10,
        "high": [105.0] * 10,
        "low": [95.0] * 10,
        "close": [101.0] * 10,
        "volume": [10.0] * 10,
    })
    engine = DeterministicReplayEngine()
    with pytest.raises(ValueError) as exc_info:
        engine.execute_parity_replay(short_df)

    assert "at least 20" in str(exc_info.value)
