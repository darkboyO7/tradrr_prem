"""Tests for Dynamic Collateral Rehypothecation & Multi-Venue Liquidity Balancing Engine."""

import pytest
from risk.collateral_rehypothecation import (
    CollateralRebalancePlan,
    CollateralRehypothecationEngine,
    CollateralTransferLeg,
    TransferStatus,
    TransferUrgency,
    VenueCollateralAsset,
    VenueCollateralState,
)


def test_venue_collateral_metrics_and_haircuts():
    """Verify venue collateral haircut computation, margin utilization, and health classification."""
    assets = {
        "BTC": VenueCollateralAsset(asset="BTC", quantity=2.0, mark_price=60000.0, haircut_pct=0.10),
        "SOL": VenueCollateralAsset(asset="SOL", quantity=100.0, mark_price=150.0, haircut_pct=0.25),
    }
    # BTC nominal: 120,000, effective: 108,000
    # SOL nominal: 15,000, effective: 11,250
    # Total effective from assets: 119,250
    # Cash: 10,000
    # Total effective: 129,250
    # Margin required: 64,625 -> utilization = 50.0% (HEALTHY)
    venue = VenueCollateralState(
        venue_id="binance",
        venue_type="EXCHANGE",
        assets=assets,
        cash_usd=10000.0,
        margin_required_usd=64625.0,
        maintenance_margin_required_usd=30000.0,
        target_utilization_pct=65.0,
        warning_utilization_pct=80.0,
        critical_utilization_pct=90.0,
    )

    assert venue.total_nominal_equity_usd == 145000.0
    assert abs(venue.total_effective_collateral_usd - 129250.0) < 1e-4
    assert abs(venue.margin_utilization_pct - 50.0) < 0.1
    assert venue.health_classification == "HEALTHY"
    assert venue.free_collateral_usd == 129250.0 - 64625.0
    assert venue.liquidation_distance_usd == 129250.0 - 30000.0


def test_rebalance_plan_generation_emergency():
    """Verify rebalance plan creation when a venue is in critical margin call status."""
    # Venue 1: Critical margin compression (utilization > 90%)
    v_compressed = VenueCollateralState(
        venue_id="deribit",
        venue_type="EXCHANGE",
        cash_usd=5000.0,
        margin_required_usd=92000.0,
        maintenance_margin_required_usd=60000.0,
        target_utilization_pct=65.0,
        warning_utilization_pct=80.0,
        critical_utilization_pct=90.0,
    )
    # Effective collateral = 5,000 -> utilization = 92,000 / 5,000 = 1840% (extreme deficit)

    # Venue 2: Capital surplus (Coinbase institutional custody)
    v_surplus = VenueCollateralState(
        venue_id="coinbase",
        venue_type="EXCHANGE",
        cash_usd=300000.0,
        margin_required_usd=20000.0,
        target_utilization_pct=65.0,
        unencumbered_reserve_buffer_usd=20000.0,
    )

    engine = CollateralRehypothecationEngine(venues={"deribit": v_compressed, "coinbase": v_surplus})
    plan = engine.generate_rebalance_plan()

    assert plan.is_emergency is True
    assert plan.total_transfers_count >= 1
    assert plan.recommended_action == "DISPATCH_TRANSFERS"
    assert len(plan.transfer_legs) > 0

    first_leg = plan.transfer_legs[0]
    assert first_leg.source_venue == "coinbase"
    assert first_leg.target_venue == "deribit"
    assert first_leg.urgency == TransferUrgency.EMERGENCY_LIQUIDATION_PREVENTION
    assert first_leg.amount_usd > 10000.0


def test_execute_transfer_plan_simulation_and_state_transition():
    """Verify end-to-end execution of a rebalance plan and resulting venue state transition."""
    v_src = VenueCollateralState(
        venue_id="vault_fireblocks",
        venue_type="CUSTODIAN_VAULT",
        cash_usd=100000.0,
        margin_required_usd=0.0,
        transfer_fee_flat_usd=25.0,
    )
    v_dst = VenueCollateralState(
        venue_id="binance_perp",
        venue_type="EXCHANGE",
        cash_usd=10000.0,
        margin_required_usd=15000.0,  # 15,000 / 10,000 = 150% util
        target_utilization_pct=65.0,
    )

    engine = CollateralRehypothecationEngine(venues={"vault_fireblocks": v_src, "binance_perp": v_dst})
    plan = engine.generate_rebalance_plan()
    assert len(plan.transfer_legs) == 1

    # Execute plan live (dry_run=False)
    executed = engine.execute_transfer_plan(plan, dry_run=False)
    assert len(executed) == 1
    assert executed[0].status == TransferStatus.CONFIRMED

    # Source should have deducted amount + fee
    assert v_src.cash_usd < 100000.0
    # Destination should have received capital
    assert v_dst.cash_usd > 10000.0
    # Post-rebalance destination utilization should be within healthy levels (< 70%)
    assert v_dst.margin_utilization_pct <= 70.0


def test_source_venue_insufficient_solvency_handling():
    """Ensure engine rejects or fails transfers if source venue lacks adequate unencumbered cash."""
    v_src = VenueCollateralState(
        venue_id="src_dry",
        cash_usd=50.0,
        transfer_fee_flat_usd=15.0,
    )
    v_dst = VenueCollateralState(
        venue_id="dst_hungry",
        cash_usd=100.0,
    )

    engine = CollateralRehypothecationEngine(venues={"src_dry": v_src, "dst_hungry": v_dst})
    # Fabricate an oversized transfer leg manually
    bad_leg = CollateralTransferLeg(
        source_venue="src_dry",
        target_venue="dst_hungry",
        asset="USD",
        quantity=5000.0,
        amount_usd=5000.0,
        estimated_fee_usd=15.0,
        estimated_latency_seconds=300.0,
        urgency=TransferUrgency.CRITICAL_MARGIN_CALL,
    )
    plan = CollateralRebalancePlan(transfer_legs=[bad_leg])

    executed = engine.execute_transfer_plan(plan, dry_run=False)
    assert len(executed) == 1
    assert executed[0].status == TransferStatus.FAILED
    assert v_src.cash_usd == 50.0  # Balance untouched


def test_crypto_asset_collateral_transfer_and_creation():
    """Verify non-cash crypto asset collateral transfer to venue that previously had zero balance."""
    btc_asset = VenueCollateralAsset(asset="BTC", quantity=5.0, mark_price=65000.0, haircut_pct=0.10)
    v_src = VenueCollateralState(
        venue_id="prime_broker",
        assets={"BTC": btc_asset},
        cash_usd=1000.0,
        transfer_fee_flat_usd=20.0,
    )
    v_dst = VenueCollateralState(
        venue_id="kraken",
        assets={},
        cash_usd=500.0,
    )

    engine = CollateralRehypothecationEngine(venues={"prime_broker": v_src, "kraken": v_dst})

    crypto_leg = CollateralTransferLeg(
        source_venue="prime_broker",
        target_venue="kraken",
        asset="BTC",
        quantity=1.5,
        amount_usd=1.5 * 65000.0,
        estimated_fee_usd=20.0,
        estimated_latency_seconds=600.0,
        urgency=TransferUrgency.ROUTINE,
    )
    plan = CollateralRebalancePlan(transfer_legs=[crypto_leg])

    executed = engine.execute_transfer_plan(plan, dry_run=False)
    assert executed[0].status == TransferStatus.CONFIRMED
    assert v_src.assets["BTC"].quantity == 3.5
    assert "BTC" in v_dst.assets
    assert v_dst.assets["BTC"].quantity == 1.5
    assert v_dst.assets["BTC"].mark_price == 65000.0
