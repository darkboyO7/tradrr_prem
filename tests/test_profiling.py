"""Tests for Statistical Expectancy and Return Profiling Engine."""

import numpy as np
import pandas as pd
import pytest

from analysis.profiling import StatisticalProfiler


def test_calculate_forward_returns():
    df = pd.DataFrame({
        "close": [100.0, 102.0, 105.0, 103.0, 108.0],
    })

    fwd_df = StatisticalProfiler.calculate_forward_returns(df, horizons=[1, 2])
    assert "fwd_return_1" in fwd_df.columns
    assert "fwd_return_2" in fwd_df.columns

    # 1 bar forward from index 0: (102 - 100) / 100 = +0.02
    assert pytest.approx(fwd_df["fwd_return_1"].iloc[0], rel=1e-4) == 0.02
    # 2 bars forward from index 0: (105 - 100) / 100 = +0.05
    assert pytest.approx(fwd_df["fwd_return_2"].iloc[0], rel=1e-4) == 0.05


def test_profile_condition_expectancy():
    # Construct 40 bars with a clear positive condition
    np.random.seed(42)
    n = 60
    closes = 100.0 * np.exp(np.cumsum(np.random.normal(0.001, 0.01, n)))
    flags = [1 if i % 2 == 0 else 0 for i in range(n)]

    df = pd.DataFrame({
        "close": closes,
        "is_setup": flags,
    })

    mask = df["is_setup"] == 1
    profiles = StatisticalProfiler.profile_condition(df, mask, condition_name="TestSetup", horizons=[1, 3])

    assert 1 in profiles
    assert 3 in profiles

    prof1 = profiles[1]
    assert prof1.condition_name == "TestSetup"
    assert prof1.sample_size > 0
    assert 0.0 <= prof1.win_rate <= 1.0
    assert 0.0 <= prof1.loss_rate <= 1.0
    assert isinstance(prof1.expectancy, float)
    assert isinstance(prof1.profit_factor, float)
    assert isinstance(prof1.t_stat, float)
