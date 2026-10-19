"""Tests for Real-Time Decentralized Cross-Chain Liquidity & Bridge Settlement Monitor."""

import pytest

from compliance.cross_chain_monitor import (
    BridgeProtocol,
    BridgeTransferRecord,
    BridgeTransferStatus,
    ChainFinalityProfile,
    CrossChainBridgeMonitor,
    CrossChainMonitorReport,
    CrossChainRouteQuote,
    SupportedChain,
)


def test_chain_finality_and_reorg_safety():
    """Verify block confirmations, latency estimations, and reorg safety evaluation."""
    monitor = CrossChainBridgeMonitor()

    # Update Ethereum block height with 40 unfinalized blocks (>= 32 safety buffer)
    eth_prof = monitor.update_chain_height(
        chain=SupportedChain.ETHEREUM,
        current_block=19500050,
        finalized_block=19500010,  # 40 unfinalized
    )
    assert eth_prof.is_reorg_safe is True
    assert eth_prof.unfinalized_blocks == 40
    assert eth_prof.estimated_finality_latency_seconds == 40 * 12.0

    # Update Arbitrum with insufficient confirmations
    arb_prof = monitor.update_chain_height(
        chain=SupportedChain.ARBITRUM,
        current_block=185000050,
        finalized_block=185000030,  # 20 unfinalized < 64 safety buffer
    )
    assert arb_prof.is_reorg_safe is False
    assert arb_prof.unfinalized_blocks == 20


def test_cross_chain_transfer_lifecycle():
    """Verify in-flight bridge transfer registration, confirmation progression, and final settlement."""
    monitor = CrossChainBridgeMonitor()
    monitor.update_token_price(SupportedChain.ARBITRUM, "ETH", 3200.0)

    # 1. Register transfer
    rec = monitor.register_transfer(
        transfer_id="XFER_001",
        source_tx_hash="0xabc123",
        source_chain=SupportedChain.ARBITRUM,
        destination_chain=SupportedChain.ETHEREUM,
        token_symbol="ETH",
        amount=5.0,
        bridge_protocol=BridgeProtocol.ACROSS,
        source_block_height=1000,
    )

    assert rec.status == BridgeTransferStatus.INITIATED
    assert rec.notional_usd == 16000.0  # 5 * 3200
    assert rec.required_confirmations == 64

    # 2. Advance confirmations partially
    rec_part = monitor.update_transfer_confirmations("XFER_001", new_block_height=1030)
    assert rec_part.current_confirmations == 30
    assert rec_part.status == BridgeTransferStatus.INITIATED

    # 3. Advance past required confirmations
    rec_conf = monitor.update_transfer_confirmations("XFER_001", new_block_height=1070)
    assert rec_conf.current_confirmations == 70
    assert rec_conf.status == BridgeTransferStatus.CONFIRMED_SOURCE

    # 4. Settle transfer
    rec_settled = monitor.complete_transfer("XFER_001", destination_tx_hash="0xdef456")
    assert rec_settled is not None
    assert rec_settled.status == BridgeTransferStatus.SETTLED_FINAL
    assert rec_settled.destination_tx_hash == "0xdef456"
    assert rec_settled.settled_at_utc is not None


def test_bridge_arbitrage_quote_evaluation():
    """Verify price discrepancy scanning and net profitability after gas and bridge protocol fees."""
    monitor = CrossChainBridgeMonitor()

    # Price on Arbitrum = 3200, Price on Ethereum = 3225 (gross profit = 25 USD / ETH)
    monitor.update_token_price(SupportedChain.ARBITRUM, "ETH", 3200.0)
    monitor.update_token_price(SupportedChain.ETHEREUM, "ETH", 3225.0)

    quote = monitor.evaluate_route_quote(
        source_chain=SupportedChain.ARBITRUM,
        destination_chain=SupportedChain.ETHEREUM,
        token_symbol="ETH",
        bridge_protocol=BridgeProtocol.ACROSS,
        trade_amount=10.0,
    )

    assert isinstance(quote, CrossChainRouteQuote)
    assert quote.source_price_usd == 3200.0
    assert quote.destination_price_usd == 3225.0
    assert quote.price_discrepancy_bps > 70.0
    # Gross profit = 10 * 25 = 250 USD
    # Fees = gas (~0.50) + bridge fee (~12.80)
    assert quote.net_arbitrage_profit_usd > 200.0
    assert quote.is_arbitrage_profitable is True
    assert quote.expected_latency_seconds > 0.0


def test_monitor_report_generation():
    """Verify system-wide cross-chain monitor report synthesis."""
    monitor = CrossChainBridgeMonitor()
    monitor.update_token_price(SupportedChain.ETHEREUM, "ETH", 3250.0)
    monitor.update_token_price(SupportedChain.ARBITRUM, "ETH", 3200.0)

    report = monitor.generate_monitor_report()
    assert isinstance(report, CrossChainMonitorReport)
    assert len(report.chains) == 6
    assert report.active_transfers_count >= 0
    assert report.total_volume_bridged_usd >= 0.0
