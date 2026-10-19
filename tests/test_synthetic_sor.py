"""Tests for Cross-Venue Synthetic Smart Order Router (SOR) & Dynamic Liquidity Splitting Engine."""

import pytest

from execution.synthetic_sor import (
    DynamicLiquidityRoutingPlan,
    SOROptimizationObjective,
    SyntheticChildAllocation,
    SyntheticSmartOrderRouter,
    VenueBookDepth,
    VenueLiquidityType,
)


@pytest.fixture
def multi_venue_sor():
    """Setup SOR with Binance, Coinbase, and Kraken depth books for BTC-USD."""
    router = SyntheticSmartOrderRouter(step_granularity=0.1)

    # Binance: Tightest top of book (2.0 at 65000), but deeper levels jump to 65020 (3.0)
    # Bids: top of book (2.0 at 64995), deeper levels drop to 64970 (3.0)
    router.register_venue_depth(
        VenueBookDepth(
            venue="Binance",
            symbol="BTC-USD",
            bids=[(64995.0, 2.0), (64970.0, 3.0), (64950.0, 5.0)],
            asks=[(65000.0, 2.0), (65020.0, 3.0), (65030.0, 5.0)],
            maker_fee_bps=-1.0,
            taker_fee_bps=6.0,
            latency_ms=4.0,
        )
    )

    # Coinbase: 4.0 at 65004, deeper at 65008
    # Bids: 4.0 at 64991, deeper at 64985
    router.register_venue_depth(
        VenueBookDepth(
            venue="Coinbase",
            symbol="BTC-USD",
            bids=[(64991.0, 4.0), (64985.0, 6.0), (64980.0, 10.0)],
            asks=[(65004.0, 4.0), (65008.0, 6.0), (65015.0, 10.0)],
            maker_fee_bps=-0.5,
            taker_fee_bps=12.0,
            latency_ms=15.0,
        )
    )

    # Kraken: 1.5 at 65001 (cheaper than Binance 65020 or Coinbase 65008), 4.0 at 65005
    # Bids: 1.5 at 64994, 4.0 at 64988
    router.register_venue_depth(
        VenueBookDepth(
            venue="Kraken",
            symbol="BTC-USD",
            bids=[(64994.0, 1.5), (64988.0, 4.0), (64975.0, 8.0)],
            asks=[(65001.0, 1.5), (65005.0, 4.0), (65020.0, 8.0)],
            maker_fee_bps=-1.2,
            taker_fee_bps=8.0,
            latency_ms=10.0,
        )
    )

    return router


def test_convex_multi_venue_liquidity_split(multi_venue_sor):
    """Test convex marginal water-filling across multiple venues."""
    router = multi_venue_sor

    # Plan BUY of 5.0 BTC
    plan = router.plan_liquidity_split(
        symbol="BTC-USD",
        side="BUY",
        parent_quantity=5.0,
        objective=SOROptimizationObjective.MIN_TOTAL_NET_COST,
    )

    assert isinstance(plan, DynamicLiquidityRoutingPlan)
    assert plan.symbol == "BTC-USD"
    assert plan.side == "BUY"
    assert plan.parent_quantity == 5.0
    assert plan.is_fully_allocated is True
    assert plan.unallocated_quantity == 0.0

    # Must allocate across multiple venues rather than naively 100% on one
    assert len(plan.child_allocations) >= 2
    allocated_sum = sum(c.allocated_quantity for c in plan.child_allocations)
    assert allocated_sum == pytest.approx(5.0, rel=1e-3)

    # Blended price should be competitive
    assert 65000.0 <= plan.blended_vwap <= 65015.0
    assert plan.blended_net_price >= plan.blended_vwap  # Due to taker fees

    # Multi-venue convex split should produce positive savings vs naive single-venue dump
    assert plan.expected_savings_bps >= 0.0


def test_maker_rebate_harvesting_objective(multi_venue_sor):
    """Test maker rebate harvesting policy allocating to passive liquidity."""
    router = multi_venue_sor

    plan = router.plan_liquidity_split(
        symbol="BTC-USD",
        side="BUY",
        parent_quantity=3.0,
        objective=SOROptimizationObjective.MAX_MAKER_HARVESTING,
    )

    assert plan.objective == SOROptimizationObjective.MAX_MAKER_HARVESTING
    for child in plan.child_allocations:
        assert child.liquidity_type == VenueLiquidityType.MAKER
        # Maker fees are negative (rebates)
        assert child.expected_fee_bps < 0.0


def test_sell_order_multi_venue_allocation(multi_venue_sor):
    """Test SELL parent order allocation maximizing revenue and net proceeds."""
    router = multi_venue_sor

    plan = router.plan_liquidity_split(
        symbol="BTC-USD",
        side="SELL",
        parent_quantity=4.0,
        objective=SOROptimizationObjective.MIN_TOTAL_NET_COST,
    )

    assert plan.side == "SELL"
    assert plan.is_fully_allocated is True
    assert len(plan.child_allocations) >= 2
    # For SELL, net unit price should be lower than gross VWAP due to fee deduction
    assert plan.blended_net_price <= plan.blended_vwap


def test_latency_lead_lag_priority(multi_venue_sor):
    """Test that dispatch priority accounts for arrival latency differences."""
    router = multi_venue_sor

    plan = router.plan_liquidity_split(
        symbol="BTC-USD",
        side="BUY",
        parent_quantity=4.0,
    )

    alloc_by_venue = {c.venue.lower(): c for c in plan.child_allocations}
    # Coinbase has 15ms latency, Binance has 4ms latency
    # To synchronize arrival, Coinbase should have higher or equal dispatch priority (staggered)
    if "coinbase" in alloc_by_venue and "binance" in alloc_by_venue:
        assert alloc_by_venue["coinbase"].estimated_arrival_latency_ms > alloc_by_venue["binance"].estimated_arrival_latency_ms


def test_validation_and_serialization(multi_venue_sor):
    """Test input validations and dictionary conversion."""
    router = multi_venue_sor

    # Validation errors
    with pytest.raises(ValueError, match="Invalid side"):
        router.plan_liquidity_split("BTC-USD", "HOLD", 1.0)

    with pytest.raises(ValueError, match="positive"):
        router.plan_liquidity_split("BTC-USD", "BUY", -1.0)

    with pytest.raises(ValueError, match="No liquidity venues"):
        router.plan_liquidity_split("UNKNOWN-COIN", "BUY", 1.0)

    plan = router.plan_liquidity_split("BTC-USD", "BUY", 2.0)
    d = plan.to_dict()
    assert d["symbol"] == "BTC-USD"
    assert "child_allocations" in d
    assert len(d["child_allocations"]) > 0
    assert "blended_vwap" in d
