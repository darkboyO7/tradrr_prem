"""Tests for Decentralized MEV & Front-Running Attack Surface Defense Gateway."""

import pytest

from compliance.mev_defense import (
    ExecutionRelayType,
    FlashbotsBundleSubmission,
    MEVDefenseAssessment,
    MEVDefenseGateway,
    MEVRiskLevel,
    PendingMempoolTx,
)


@pytest.fixture
def mev_gateway():
    """Setup MEV defense gateway with simulated mempool transactions."""
    gw = MEVDefenseGateway(base_fee_gwei=30.0, private_relay_threshold_score=50.0)

    # Populate sample competitor mempool txs
    sample_mempool = [
        PendingMempoolTx(
            tx_hash=f"0x{i:064x}",
            from_address="0x1111111111111111111111111111111111111111",
            to_address="0xUniswapRouter",
            gas_price_gwei=35.0,
            max_priority_fee_gwei=float(2.0 + i * 0.5),
            value_eth=1.0,
            method_id="0x38ed1739",
            timestamp_ns=1_000_000_000 + i * 1000,
        )
        for i in range(10)
    ]
    gw.update_mempool_snapshot(sample_mempool)
    return gw


def test_benign_small_order_mev_risk(mev_gateway):
    """Test small order relative to pool liquidity results in LOW risk and public mempool routing."""
    gw = mev_gateway

    # Order of $1,000 in a $10,000,000 pool with tight 15 bps slippage
    assessment = gw.assess_order_mev_risk(
        order_id="ORD_SMALL_01",
        symbol="ETH-USDC",
        notional_usd=1000.0,
        pool_liquidity_usd=10_000_000.0,
        requested_slippage_bps=15.0,
    )

    assert isinstance(assessment, MEVDefenseAssessment)
    assert assessment.risk_level == MEVRiskLevel.LOW
    assert assessment.risk_score < 25.0
    assert assessment.private_bundle_required is False
    assert assessment.recommended_relay == ExecutionRelayType.PUBLIC_MEMPOOL
    assert assessment.recommended_slippage_bps <= 15.0


def test_critical_large_order_sandwich_vulnerability(mev_gateway):
    """Test large order with loose slippage triggers CRITICAL risk and MEV-Share bundle."""
    gw = mev_gateway

    # $500,000 order in a $2,000,000 pool (25% of pool!) with 150 bps slippage
    assessment = gw.assess_order_mev_risk(
        order_id="ORD_WHALE_01",
        symbol="PEPE-ETH",
        notional_usd=500_000.0,
        pool_liquidity_usd=2_000_000.0,
        requested_slippage_bps=150.0,
    )

    assert assessment.risk_level == MEVRiskLevel.CRITICAL
    assert assessment.risk_score >= 75.0
    assert assessment.private_bundle_required is True
    assert assessment.recommended_relay == ExecutionRelayType.MEV_SHARE_BUNDLE
    assert assessment.estimated_potential_sandwich_loss_usd > 1000.0
    assert "CRITICAL MEV EXPOSURE" in assessment.reason

    # Recommended slippage is auto-tightened to prevent excessive extraction
    assert assessment.recommended_slippage_bps < assessment.requested_slippage_bps


def test_flashbots_bundle_generation(mev_gateway):
    """Test construction of atomic private bundle for Flashbots builders."""
    gw = mev_gateway

    raw_txs = [
        "0x02f87301018402faf080...",
        "0x02f87301028402faf081...",
    ]

    bundle = gw.build_flashbots_bundle(
        raw_signed_txs=raw_txs,
        current_block_number=19_000_000,
        block_lead=1,
    )

    assert isinstance(bundle, FlashbotsBundleSubmission)
    assert bundle.target_block_number == 19_000_001
    assert len(bundle.raw_transactions) == 2
    assert bundle.bundle_id.startswith("bndl_")
    assert bundle.max_timestamp > bundle.min_timestamp

    b_dict = bundle.to_dict()
    assert b_dict["transactions_count"] == 2
    assert b_dict["target_block_number"] == 19_000_001


def test_priority_fee_recommendation_from_mempool(mev_gateway):
    """Test priority fee calibrating to 80th percentile of pending mempool."""
    gw = mev_gateway

    assessment = gw.assess_order_mev_risk(
        order_id="ORD_GAS_01",
        symbol="BTC-USD",
        notional_usd=50_000.0,
        pool_liquidity_usd=5_000_000.0,
        requested_slippage_bps=30.0,
    )

    # 80th percentile of range [2.0, 6.5] is around 5.6 + 1.0 = ~6.6
    assert assessment.recommended_priority_fee_gwei > 4.0


def test_validation_errors(mev_gateway):
    """Test input validations and edge cases."""
    gw = mev_gateway

    with pytest.raises(ValueError, match="notional_usd must be positive"):
        gw.assess_order_mev_risk("ORD_ERR", "ETH", -10.0, 1000.0)

    with pytest.raises(ValueError, match="pool_liquidity_usd must be positive"):
        gw.assess_order_mev_risk("ORD_ERR", "ETH", 10.0, 0.0)

    with pytest.raises(ValueError, match="raw_signed_txs cannot be empty"):
        gw.build_flashbots_bundle([], 100)
