"""Tests for Real-Time Slippage & Liquidity Cost Surface Calibration Engine."""

from datetime import datetime, timezone
import pytest
from execution.cost_surface_calibrator import (
    CalibratedCostParameters,
    CostSurfaceCalibrator,
    CostSurfaceGrid,
    CostSurfacePoint,
    SlippagePredictionRequest,
    SlippagePredictionResult,
)
from execution.tca import TradeExecutionRecord


def test_default_parameters_prior_calibration():
    """Verify default prior parameters when insufficient empirical trade observations exist."""
    calibrator = CostSurfaceCalibrator()
    params = calibrator.get_parameters("BTC-USD", "coinbase")

    assert params.symbol == "BTC-USD"
    assert params.venue == "coinbase"
    assert abs(params.calibrated_eta - 0.142) < 1e-3
    assert abs(params.calibrated_gamma - 0.314) < 1e-3
    assert params.sample_trades_count == 0
    assert params.r_squared == 0.0


def test_trade_ingestion_and_empirical_calibration():
    """Verify empirical OLS recalibration of eta and gamma from trade telemetry."""
    calibrator = CostSurfaceCalibrator(min_samples_for_recal=5)

    # Ingest 8 trades with increasing participation and impact
    now = datetime.now(timezone.utc)
    for i in range(1, 9):
        qty = float(i) * 0.5
        record = TradeExecutionRecord(
            order_id=f"ord_cal_{i}",
            symbol="BTC-USD",
            side="BUY",
            venue="coinbase",
            order_quantity=qty,
            filled_quantity=qty,
            decision_price=65000.0,
            arrival_price=65000.0,
            arrival_bid=64999.0,
            arrival_ask=65001.0,
            execution_price=65000.0 + (i * 2.5),  # Increasing temp impact
            terminal_price=65000.0 + (i * 1.2),   # Increasing perm impact
            decision_time=now,
            arrival_time=now,
            execution_time=now,

        )
        calibrator.ingest_trade_execution(record, realized_volatility=0.55, market_interval_volume=10.0)

    params = calibrator.calibrate("BTC-USD", "coinbase")
    assert params.sample_trades_count == 8
    assert params.calibrated_eta > 0.0
    assert params.calibrated_gamma > 0.0
    assert params.r_squared >= 0.0


def test_slippage_prediction_and_ofi_asymmetry():
    """Verify slippage forecasting and asymmetric order flow imbalance (OFI) adjustment."""
    calibrator = CostSurfaceCalibrator()

    # 1. Prediction with neutral OFI
    req_neutral = SlippagePredictionRequest(
        symbol="BTC-USD",
        venue="coinbase",
        side="BUY",
        order_quantity=2.0,
        order_duration_seconds=60.0,
        reference_price=65000.0,
        realized_volatility=0.50,
        order_flow_imbalance=0.0,
    )
    res_neutral = calibrator.predict_slippage(req_neutral)
    assert res_neutral.symbol == "BTC-USD"
    assert res_neutral.expected_total_slippage_bps > 0.0
    assert res_neutral.expected_total_slippage_usd > 0.0
    assert abs(res_neutral.expected_total_slippage_usd - ((res_neutral.expected_total_slippage_bps / 10000.0) * 130000.0)) < 0.1

    # 2. Prediction with adverse OFI (+0.80 for BUY)
    req_adverse = SlippagePredictionRequest(
        symbol="BTC-USD",
        venue="coinbase",
        side="BUY",
        order_quantity=2.0,
        order_duration_seconds=60.0,
        reference_price=65000.0,
        realized_volatility=0.50,
        order_flow_imbalance=0.80,  # Intense buy pressure
    )
    res_adverse = calibrator.predict_slippage(req_adverse)

    # 3. Prediction with favorable OFI (-0.80 for BUY)
    req_favorable = SlippagePredictionRequest(
        symbol="BTC-USD",
        venue="coinbase",
        side="BUY",
        order_quantity=2.0,
        order_duration_seconds=60.0,
        reference_price=65000.0,
        realized_volatility=0.50,
        order_flow_imbalance=-0.80,  # Selling inventory available
    )
    res_favorable = calibrator.predict_slippage(req_favorable)

    # Adverse OFI must produce higher expected slippage than favorable OFI
    assert res_adverse.expected_total_slippage_bps > res_neutral.expected_total_slippage_bps
    assert res_neutral.expected_total_slippage_bps > res_favorable.expected_total_slippage_bps


def test_cost_surface_grid_generation_and_monotonicity():
    """Verify cost surface grid dimensions and structural monotonicity across participation and volatility."""
    calibrator = CostSurfaceCalibrator()
    grid = calibrator.generate_cost_surface("ETH-USD", "coinbase")

    assert grid.symbol == "ETH-USD"
    assert grid.venue == "coinbase"
    assert len(grid.surface_matrix_bps) == len(grid.volatility_bins)
    assert len(grid.surface_matrix_bps[0]) == len(grid.participation_bins)

    # Monotonicity checks:
    # 1. Across participation rate (row-wise): as POV increases, slippage must strictly increase
    for row in grid.surface_matrix_bps:
        for j in range(len(row) - 1):
            assert row[j] < row[j + 1], f"Expected monotonic increase along POV: {row[j]} >= {row[j+1]}"

    # 2. Across volatility (column-wise): as volatility increases, slippage must strictly increase
    n_cols = len(grid.participation_bins)
    for col_idx in range(n_cols):
        col_vals = [grid.surface_matrix_bps[row_idx][col_idx] for row_idx in range(len(grid.volatility_bins))]
        for i in range(len(col_vals) - 1):
            assert col_vals[i] < col_vals[i + 1], f"Expected monotonic increase along Vol: {col_vals[i]} >= {col_vals[i+1]}"
