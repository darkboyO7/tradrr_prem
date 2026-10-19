"""Tests for Statistical Arbitrage, Kalman Filter, and Cointegration Pairs Trading Engine."""

import numpy as np
import pandas as pd
import pytest

from analysis.stat_arb import (
    CointegrationAnalyzer,
    CointegrationResult,
    KalmanFilterHedgeRatio,
    PairsSpreadMonitor,
    PairTradingSignal,
)


def test_kalman_filter_hedge_ratio_tracking():
    np.random.seed(42)
    n = 200
    # True relationship: y = 15.0 + 2.0 * x + noise
    x = 100.0 + np.cumsum(np.random.normal(0, 1.0, n))
    noise = np.random.normal(0, 0.5, n)
    y = 15.0 + 2.0 * x + noise

    kf = KalmanFilterHedgeRatio(initial_beta=1.0, initial_alpha=0.0)
    series_y = pd.Series(y)
    series_x = pd.Series(x)

    df_filtered = kf.filter_series(series_y, series_x)

    assert len(df_filtered) == n
    # After warm-up, beta should converge close to 2.0
    final_beta = df_filtered["beta"].iloc[-1]
    assert 1.8 <= final_beta <= 2.2
    assert abs(df_filtered["spread"].iloc[-1]) < 5.0


def test_cointegration_analyzer_cointegrated_pair():
    np.random.seed(123)
    n = 300
    # Generate stationary AR(1) spread: S_t = 0.7 * S_{t-1} + noise
    spread = np.zeros(n)
    for i in range(1, n):
        spread[i] = 0.75 * spread[i - 1] + np.random.normal(0, 1.0)

    # Generate random walk X and cointegrated Y = 2.5 * X + S
    x = 50.0 + np.cumsum(np.random.normal(0, 0.8, n))
    y = 2.5 * x + spread

    analyzer = CointegrationAnalyzer()
    res = analyzer.test_cointegration(pd.Series(y), pd.Series(x), asset_y="ASSET_A", asset_x="ASSET_B")

    assert isinstance(res, CointegrationResult)
    assert res.is_cointegrated is True
    assert res.adf_t_statistic < res.critical_values["5%"]
    assert 2.3 <= res.static_hedge_ratio <= 2.7
    assert res.half_life_bars < 30.0  # Fast mean-reversion
    assert res.r_squared > 0.85


def test_cointegration_analyzer_spurious_non_cointegrated_pair():
    np.random.seed(999)
    n = 200
    # Two independent random walks without cointegrating relationship
    x = np.cumsum(np.random.normal(0, 1.0, n)) + 100.0
    y = np.cumsum(np.random.normal(0, 1.0, n)) + 50.0

    analyzer = CointegrationAnalyzer()
    res = analyzer.test_cointegration(pd.Series(y), pd.Series(x), asset_y="Y", asset_x="X")

    assert isinstance(res, CointegrationResult)
    # Independent random walks should fail cointegration test
    assert res.is_cointegrated is False
    assert res.adf_t_statistic > res.critical_values["5%"]


def test_pairs_spread_monitor_signals():
    monitor = PairsSpreadMonitor(
        pair_name="ETH/BTC",
        asset_y="ETH",
        asset_x="BTC",
        z_entry=2.0,
        z_exit=0.5,
        z_stop=3.5,
        window=30,
    )

    np.random.seed(42)
    # Prime history with normal baseline spread (std ~ 5.0)
    for _ in range(30):
        sig = monitor.process_observation(price_y=3000.0 + np.random.normal(0, 5.0), price_x=60000.0)

    assert sig.action in ("HOLD", "UNWIND_SPREAD")

    # Inject moderate spread spike (~2.5 sigma) -> should generate SHORT_SPREAD
    spike_sig = monitor.process_observation(price_y=3012.5, price_x=60000.0)
    assert 2.0 <= spike_sig.z_score < 3.5
    assert spike_sig.action == "SHORT_SPREAD"
    assert spike_sig.weight_y > 0
    assert spike_sig.weight_x < 0

    # Inject extreme divergence (> 3.5 sigma) -> should trigger STOP_LOSS_EXIT
    extreme_sig = monitor.process_observation(price_y=3050.0, price_x=60000.0)
    assert extreme_sig.action == "STOP_LOSS_EXIT"


def test_pairs_spread_monitor_cusum_structural_break():
    monitor = PairsSpreadMonitor(
        pair_name="SOL/ETH",
        asset_y="SOL",
        asset_x="ETH",
        cusum_threshold=4.0,
        window=20,
    )

    # Establish baseline
    for _ in range(25):
        monitor.process_observation(price_y=150.0, price_x=3000.0)

    # Inject persistent one-sided drift to trigger CUSUM break
    broken = False
    for step in range(15):
        sig = monitor.process_observation(price_y=150.0 + (step * 8.0), price_x=3000.0)
        if sig.is_structurally_broken:
            broken = True
            assert sig.action == "STOP_LOSS_EXIT"
            break

    assert broken is True
