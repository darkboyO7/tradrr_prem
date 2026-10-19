"""Tests for Almgren-Chriss Execution Cost and Market Impact Model."""

import pytest

from execution.cost_model import AlmgrenChrissCostModel, ExecutionCostEstimate


def test_cost_model_component_breakdown():
    """Verify that all cost components are calculated and sum to total shortfall."""
    model = AlmgrenChrissCostModel(default_taker_fee=0.0010)

    est = model.estimate_cost(
        symbol="BTC-USD",
        side="BUY",
        quantity=1.0,
        current_price=65000.0,
        bid_ask_spread=6.50,  # 1.0 bps spread
        average_daily_volume=10_000.0,
        daily_volatility=0.03,
        execution_horizon_hours=0.5,
    )

    assert isinstance(est, ExecutionCostEstimate)
    assert est.symbol == "BTC-USD"
    assert est.notional_value_usd == 65000.0

    # Components must be positive
    assert est.half_spread_cost_usd > 0
    assert est.temporary_impact_usd > 0
    assert est.permanent_impact_usd > 0
    assert est.venue_fees_usd == 65.0  # 10 bps on 65,000

    # Sum of dollar components equals total cost
    expected_sum = (
        est.half_spread_cost_usd
        + est.temporary_impact_usd
        + est.permanent_impact_usd
        + est.venue_fees_usd
    )
    assert est.total_cost_usd == pytest.approx(expected_sum, rel=1e-3)
    assert est.total_shortfall_bps > 10.0  # fees + spread + impact


def test_cost_model_size_scaling_impact():
    """Verify non-linear market impact scaling with increasing order size."""
    model = AlmgrenChrissCostModel()

    # Small order (0.1 BTC)
    small_order = model.estimate_cost(
        symbol="BTC-USD", side="BUY", quantity=0.1, current_price=65000.0, average_daily_volume=1000.0
    )

    # Large order (50.0 BTC)
    large_order = model.estimate_cost(
        symbol="BTC-USD", side="BUY", quantity=50.0, current_price=65000.0, average_daily_volume=1000.0
    )

    # Large order must incur significantly higher bps of temporary and permanent impact
    assert large_order.temporary_impact_bps > small_order.temporary_impact_bps
    assert large_order.permanent_impact_bps > small_order.permanent_impact_bps
    assert large_order.total_shortfall_bps > small_order.total_shortfall_bps


def test_cost_model_strategy_recommendation():
    """Verify execution strategy transitions from IMMEDIATE_SWEEP to TWAP to VWAP."""
    model = AlmgrenChrissCostModel()

    # Low participation (< 1%) -> IMMEDIATE_SWEEP
    small = model.estimate_cost("BTC-USD", "BUY", quantity=0.05, current_price=65000.0, average_daily_volume=5000.0)
    assert small.recommended_strategy == "IMMEDIATE_SWEEP"
    assert small.suggested_slices_count == 1

    # Medium participation (1% - 5%) -> TWAP_SLICING
    med = model.estimate_cost("BTC-USD", "BUY", quantity=5.0, current_price=65000.0, average_daily_volume=5000.0, execution_horizon_hours=1.0)
    assert med.recommended_strategy == "TWAP_SLICING"
    assert med.suggested_slices_count >= 3

    # High participation (> 5%) -> VWAP_ALGORITHM
    huge = model.estimate_cost("BTC-USD", "BUY", quantity=100.0, current_price=65000.0, average_daily_volume=1000.0, execution_horizon_hours=1.0)
    assert huge.recommended_strategy == "VWAP_ALGORITHM"
    assert huge.suggested_slices_count >= 6
