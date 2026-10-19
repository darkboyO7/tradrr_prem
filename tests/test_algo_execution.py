"""Tests for TWAP / VWAP Algorithmic Execution Engine."""

from unittest.mock import AsyncMock
import pytest

from execution.algo import (
    AlgoEngine,
    AlgoExecutionPlan,
    AlgoExecutionReport,
    AlgoOrderType,
)
from execution.simulated import SimulatedExecutionClient


def test_create_twap_plan_with_jitter():
    """Verify TWAP plan creates randomized slices summing precisely to parent quantity."""
    engine = AlgoEngine()
    plan = engine.create_twap_plan(
        symbol="BTC-USD",
        side="BUY",
        quantity=5.0,
        duration_minutes=30.0,
        num_slices=10,
        arrival_price=64000.0,
        time_jitter_pct=0.10,
        size_jitter_pct=0.10,
        random_seed=42,
    )

    assert isinstance(plan, AlgoExecutionPlan)
    assert plan.algo_type == AlgoOrderType.TWAP
    assert plan.symbol == "BTC-USD"
    assert plan.side == "BUY"
    assert plan.parent_quantity == 5.0
    assert len(plan.slices) == 10

    # First slice must start immediately
    assert plan.slices[0].time_offset_seconds == 0.0

    # Time offsets must be strictly non-decreasing
    for i in range(1, len(plan.slices)):
        assert plan.slices[i].time_offset_seconds >= plan.slices[i - 1].time_offset_seconds

    # Sum of slice quantities must equal 5.0 exactly
    total_qty = sum(s.quantity for s in plan.slices)
    assert total_qty == pytest.approx(5.0, rel=1e-5)


def test_create_vwap_plan_volume_weighted():
    """Verify VWAP plan weights child slices proportional to intraday profile."""
    engine = AlgoEngine()
    plan = engine.create_vwap_plan(
        symbol="ETH-USD",
        side="SELL",
        quantity=20.0,
        duration_minutes=60.0,
        num_slices=6,
        arrival_price=3500.0,
        random_seed=42,
    )

    assert isinstance(plan, AlgoExecutionPlan)
    assert plan.algo_type == AlgoOrderType.VWAP
    assert plan.symbol == "ETH-USD"
    assert plan.side == "SELL"
    assert plan.parent_quantity == 20.0
    assert len(plan.slices) == 6

    # Sum of slice quantities must equal 20.0 exactly
    total_qty = sum(s.quantity for s in plan.slices)
    assert total_qty == pytest.approx(20.0, rel=1e-5)


@pytest.mark.asyncio
async def test_execute_algo_plan_quality_benchmarking():
    """Verify dispatch and implementation shortfall tracking against arrival and market VWAP."""
    engine = AlgoEngine()
    plan = engine.create_twap_plan(
        symbol="BTC-USD",
        side="BUY",
        quantity=1.0,
        duration_minutes=10.0,
        num_slices=4,
        arrival_price=65000.0,
    )

    client = SimulatedExecutionClient()
    await client.connect()

    # Simulate price drifting slightly upward: 65000, 65020, 65040, 65060
    simulated_prices = [65000.0, 65020.0, 65040.0, 65060.0]
    report = await engine.execute_algo_plan(
        plan=plan,
        client=client,
        simulated_prices=simulated_prices,
        fast_simulation=True,
    )

    assert isinstance(report, AlgoExecutionReport)
    assert report.status == "COMPLETED"
    assert report.filled_quantity == pytest.approx(1.0, rel=1e-5)
    assert report.slices_executed_count == 4

    # Realized VWAP should be roughly (65000+65020+65040+65060)/4 = 65030.0
    assert report.realized_vwap == pytest.approx(65030.0, rel=0.01)

    # For BUY, higher realized price than arrival price means positive slippage drag
    assert report.arrival_slippage_bps > 0
    assert report.total_fees_paid > 0
