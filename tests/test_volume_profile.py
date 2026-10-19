"""Tests for Intraday Volume Profiler."""

from datetime import datetime, timezone
import numpy as np
import pandas as pd
import pytest

from execution.volume_profile import IntradayVolumeProfile, VolumeProfiler


def test_canonical_volume_profile():
    """Verify that canonical prior profile has 24 valid normalized bins and cumulative curve."""
    profiler = VolumeProfiler()
    profile = profiler.get_canonical_profile("BTC-USD")

    assert isinstance(profile, IntradayVolumeProfile)
    assert profile.symbol == "BTC-USD"
    assert profile.num_bins == 24
    assert len(profile.bin_fractions) == 24
    assert len(profile.cumulative_fractions) == 24

    # Bin fractions must sum to 1.0
    assert sum(profile.bin_fractions) == pytest.approx(1.0, rel=1e-4)

    # Cumulative fractions must start > 0 and end at 1.0
    assert profile.cumulative_fractions[0] > 0
    assert profile.cumulative_fractions[-1] == pytest.approx(1.0, rel=1e-4)
    # Monotonically non-decreasing
    for i in range(1, 24):
        assert profile.cumulative_fractions[i] >= profile.cumulative_fractions[i - 1]

    # Peak hour and trough hour must be valid
    assert 0 <= profile.peak_bin_hour < 24
    assert 0 <= profile.trough_bin_hour < 24


def test_slice_fractions_for_window():
    """Verify slicing calculation over a custom execution window."""
    profiler = VolumeProfiler()
    profile = profiler.get_canonical_profile("ETH-USD")

    now = datetime(2024, 1, 1, 14, 0, tzinfo=timezone.utc)
    fractions = profiler.get_slice_fractions_for_window(
        profile=profile,
        start_time=now,
        duration_hours=2.0,
        num_slices=4,
    )

    assert len(fractions) == 4
    assert sum(fractions) == pytest.approx(1.0, rel=1e-4)
    for f in fractions:
        assert f > 0


def test_empirical_volume_profile_blending():
    """Verify empirical profile estimation from historical hourly candles."""
    profiler = VolumeProfiler()

    # Generate 72 hours of synthetic candle data (3 days)
    rng = np.random.default_rng(42)
    dates = pd.date_range("2024-01-01", periods=72, freq="1h", tz="UTC")
    # Higher volume during hours 13-17 UTC
    vols = [rng.uniform(500, 1000) if 13 <= d.hour <= 17 else rng.uniform(100, 300) for d in dates]
    df = pd.DataFrame({"volume": vols}, index=dates)

    profile = profiler.estimate_profile_from_history("SOL-USD", df)
    assert profile.symbol == "SOL-USD"
    assert sum(profile.bin_fractions) == pytest.approx(1.0, rel=1e-4)
    # Peak hour should reflect the higher simulated volume
    assert 12 <= profile.peak_bin_hour <= 18
