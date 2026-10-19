"""Trading Intelligence Platform - Real-Time Decentralized Cross-Chain Liquidity & Bridge Settlement Monitor.

Tracks multi-chain bridge routes (Ethereum, Arbitrum, Optimism, Solana, Polygon), monitoring:
- Finality latency across rollups and L1 settlement checkpoints
- Chain reorganization safety buffers and unfinalized block confirmations
- Bridge transfer fee structures (base fee + variable bridge fee + gas costs)
- Cross-chain liquidity pool depth and synthetic price discrepancies (Bridge Arbitrage)
- Automated transaction status tracking (PENDING, SETTLED, DISCREPANT, TIMED_OUT)
"""

from datetime import datetime, timezone
from enum import Enum
import hashlib
import math
from typing import Any, Dict, List, Optional, Tuple
from pydantic import BaseModel, Field

from core.logging import get_logger

logger = get_logger("cross_chain_monitor")


class SupportedChain(str, Enum):
    """Institutional classification of supported public chains and L2 rollups."""

    ETHEREUM = "ETHEREUM"
    ARBITRUM = "ARBITRUM"
    OPTIMISM = "OPTIMISM"
    SOLANA = "SOLANA"
    POLYGON = "POLYGON"
    BASE = "BASE"


class BridgeProtocol(str, Enum):
    """Supported cross-chain interoperability bridges."""

    ACROSS = "ACROSS"
    STARGATE = "STARGATE"
    HOP = "HOP"
    CCTP = "CCTP"
    WORMHOLE = "WORMHOLE"


class BridgeTransferStatus(str, Enum):
    """Status lifecycle of a cross-chain liquidity transfer."""

    INITIATED = "INITIATED"
    CONFIRMED_SOURCE = "CONFIRMED_SOURCE"
    RELAYED_IN_FLIGHT = "RELAYED_IN_FLIGHT"
    SETTLED_FINAL = "SETTLED_FINAL"
    REORG_HALTED = "REORG_HALTED"
    TIMED_OUT = "TIMED_OUT"


class ChainFinalityProfile(BaseModel):
    """Real-time finality and block confirmation parameters per chain."""

    chain: SupportedChain
    current_block_height: int
    finalized_block_height: int
    unfinalized_blocks: int
    avg_block_time_seconds: float
    reorg_safety_buffer_blocks: int
    is_reorg_safe: bool
    estimated_finality_latency_seconds: float


class CrossChainRouteQuote(BaseModel):
    """Real-time route quote across a specific bridge protocol."""

    route_id: str
    source_chain: SupportedChain
    destination_chain: SupportedChain
    token_symbol: str
    bridge_protocol: BridgeProtocol
    source_price_usd: float
    destination_price_usd: float
    price_discrepancy_bps: float
    available_liquidity_usd: float
    estimated_gas_cost_usd: float
    bridge_fee_usd: float
    total_fee_usd: float
    expected_latency_seconds: float
    net_arbitrage_profit_usd: float
    is_arbitrage_profitable: bool


class BridgeTransferRecord(BaseModel):
    """Individual in-flight or completed bridge transaction."""

    transfer_id: str
    source_tx_hash: str
    destination_tx_hash: Optional[str] = None
    source_chain: SupportedChain
    destination_chain: SupportedChain
    token_symbol: str
    amount: float
    notional_usd: float
    bridge_protocol: BridgeProtocol
    source_block_height: int
    current_confirmations: int
    required_confirmations: int
    status: BridgeTransferStatus
    initiated_at_utc: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    settled_at_utc: Optional[datetime] = None


class CrossChainMonitorReport(BaseModel):
    """System-wide telemetry report across supported chains, bridges, and arbitrage routes."""

    chains: List[ChainFinalityProfile] = []
    active_transfers_count: int
    settled_transfers_count: int
    total_volume_bridged_usd: float
    arbitrage_opportunities: List[CrossChainRouteQuote] = []
    top_profitable_route_id: Optional[str] = None
    highest_net_profit_usd: float = 0.0
    timestamp_utc: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class CrossChainBridgeMonitor:
    """Institutional cross-chain liquidity tracker, reorg safety validator, and bridge arbitrage scanner."""

    def __init__(self) -> None:
        self._chain_profiles: Dict[SupportedChain, ChainFinalityProfile] = {}
        self._active_transfers: Dict[str, BridgeTransferRecord] = {}
        self._settled_transfers: Dict[str, BridgeTransferRecord] = {}
        self._token_prices: Dict[Tuple[SupportedChain, str], float] = {}
        self._liquidity_depths: Dict[Tuple[BridgeProtocol, SupportedChain, SupportedChain, str], float] = {}

        # Initialize default finality configurations
        self._init_default_chains()

    def _init_default_chains(self) -> None:
        """Setup baseline chain parameters."""
        defaults = [
            (SupportedChain.ETHEREUM, 19500000, 19499968, 12.0, 32),
            (SupportedChain.ARBITRUM, 185000000, 184999000, 0.25, 64),
            (SupportedChain.OPTIMISM, 115000000, 114999000, 2.0, 50),
            (SupportedChain.POLYGON, 55000000, 54999700, 2.1, 128),
            (SupportedChain.SOLANA, 260000000, 259999900, 0.40, 32),
            (SupportedChain.BASE, 12000000, 11999950, 2.0, 50),
        ]
        for chain, cur_blk, fin_blk, blk_time, safety_buf in defaults:
            unfin = cur_blk - fin_blk
            safe = unfin >= safety_buf
            latency = unfin * blk_time
            self._chain_profiles[chain] = ChainFinalityProfile(
                chain=chain,
                current_block_height=cur_blk,
                finalized_block_height=fin_blk,
                unfinalized_blocks=unfin,
                avg_block_time_seconds=blk_time,
                reorg_safety_buffer_blocks=safety_buf,
                is_reorg_safe=safe,
                estimated_finality_latency_seconds=round(latency, 2),
            )

    def update_chain_height(
        self,
        chain: SupportedChain,
        current_block: int,
        finalized_block: int,
    ) -> ChainFinalityProfile:
        """Update block height telemetry for a chain and evaluate reorg safety."""
        profile = self._chain_profiles[chain]
        unfin = max(0, current_block - finalized_block)
        safe = unfin >= profile.reorg_safety_buffer_blocks
        latency = unfin * profile.avg_block_time_seconds

        updated = ChainFinalityProfile(
            chain=chain,
            current_block_height=current_block,
            finalized_block_height=finalized_block,
            unfinalized_blocks=unfin,
            avg_block_time_seconds=profile.avg_block_time_seconds,
            reorg_safety_buffer_blocks=profile.reorg_safety_buffer_blocks,
            is_reorg_safe=safe,
            estimated_finality_latency_seconds=round(latency, 2),
        )
        self._chain_profiles[chain] = updated
        return updated

    def update_token_price(self, chain: SupportedChain, token: str, price_usd: float) -> None:
        """Set observe token spot price on a chain."""
        self._token_prices[(chain, token.upper())] = float(price_usd)

    def update_bridge_liquidity(
        self,
        protocol: BridgeProtocol,
        source: SupportedChain,
        destination: SupportedChain,
        token: str,
        depth_usd: float,
    ) -> None:
        """Set available bridge liquidity depth."""
        self._liquidity_depths[(protocol, source, destination, token.upper())] = float(depth_usd)

    # -------------------------------------------------------------------------
    # Cross-Chain Transfer Tracking
    # -------------------------------------------------------------------------

    def register_transfer(
        self,
        transfer_id: str,
        source_tx_hash: str,
        source_chain: SupportedChain,
        destination_chain: SupportedChain,
        token_symbol: str,
        amount: float,
        bridge_protocol: BridgeProtocol,
        source_block_height: int,
    ) -> BridgeTransferRecord:
        """Register an in-flight cross-chain bridge transfer."""
        px = self._token_prices.get((source_chain, token_symbol.upper()), 1.0)
        notional = amount * px
        required_conf = self._chain_profiles[source_chain].reorg_safety_buffer_blocks

        record = BridgeTransferRecord(
            transfer_id=transfer_id,
            source_tx_hash=source_tx_hash,
            source_chain=source_chain,
            destination_chain=destination_chain,
            token_symbol=token_symbol.upper(),
            amount=amount,
            notional_usd=round(notional, 2),
            bridge_protocol=bridge_protocol,
            source_block_height=source_block_height,
            current_confirmations=0,
            required_confirmations=required_conf,
            status=BridgeTransferStatus.INITIATED,
        )
        self._active_transfers[transfer_id] = record
        return record

    def update_transfer_confirmations(
        self,
        transfer_id: str,
        new_block_height: int,
    ) -> Optional[BridgeTransferRecord]:
        """Advance block confirmation count and update status."""
        if transfer_id not in self._active_transfers:
            return None

        record = self._active_transfers[transfer_id]
        confs = max(0, new_block_height - record.source_block_height)
        record.current_confirmations = confs

        if confs >= record.required_confirmations:
            record.status = BridgeTransferStatus.CONFIRMED_SOURCE
        return record

    def complete_transfer(self, transfer_id: str, destination_tx_hash: str) -> Optional[BridgeTransferRecord]:
        """Settle a bridge transfer successfully."""
        if transfer_id not in self._active_transfers:
            return None

        record = self._active_transfers.pop(transfer_id)
        record.destination_tx_hash = destination_tx_hash
        record.status = BridgeTransferStatus.SETTLED_FINAL
        record.settled_at_utc = datetime.now(timezone.utc)
        self._settled_transfers[transfer_id] = record
        return record

    # -------------------------------------------------------------------------
    # Bridge Arbitrage Evaluation
    # -------------------------------------------------------------------------

    def evaluate_route_quote(
        self,
        source_chain: SupportedChain,
        destination_chain: SupportedChain,
        token_symbol: str,
        bridge_protocol: BridgeProtocol,
        trade_amount: float = 10.0,
    ) -> CrossChainRouteQuote:
        """Scan price discrepancy between chains and evaluate net arbitrage yield after fees."""
        sym = token_symbol.upper()
        p_src = self._token_prices.get((source_chain, sym), 100.0)
        p_dst = self._token_prices.get((destination_chain, sym), 100.0)

        liq = self._liquidity_depths.get((bridge_protocol, source_chain, destination_chain, sym), 500_000.0)

        # Spread in bps
        price_diff_bps = ((p_dst - p_src) / p_src) * 10000.0

        # Fee models
        gas_usd = 5.0 if source_chain == SupportedChain.ETHEREUM else 0.50
        bridge_fee_pct = 0.0004  # 4 bps protocol fee
        trade_notional = trade_amount * p_src
        bridge_fee_usd = trade_notional * bridge_fee_pct
        total_fee_usd = gas_usd + bridge_fee_usd

        gross_profit_usd = (trade_amount * p_dst) - (trade_amount * p_src)
        net_profit_usd = gross_profit_usd - total_fee_usd

        latency_src = self._chain_profiles[source_chain].estimated_finality_latency_seconds
        latency_dst = self._chain_profiles[destination_chain].estimated_finality_latency_seconds
        exp_latency = latency_src + latency_dst + 30.0  # 30s relayer window

        rid = f"ROUTE_{bridge_protocol.value}_{source_chain.value}_{destination_chain.value}_{sym}"

        return CrossChainRouteQuote(
            route_id=rid,
            source_chain=source_chain,
            destination_chain=destination_chain,
            token_symbol=sym,
            bridge_protocol=bridge_protocol,
            source_price_usd=round(p_src, 4),
            destination_price_usd=round(p_dst, 4),
            price_discrepancy_bps=round(price_diff_bps, 2),
            available_liquidity_usd=round(liq, 2),
            estimated_gas_cost_usd=round(gas_usd, 2),
            bridge_fee_usd=round(bridge_fee_usd, 2),
            total_fee_usd=round(total_fee_usd, 2),
            expected_latency_seconds=round(exp_latency, 1),
            net_arbitrage_profit_usd=round(net_profit_usd, 2),
            is_arbitrage_profitable=net_profit_usd > 0.0,
        )

    def generate_monitor_report(self) -> CrossChainMonitorReport:
        """Synthesize overall cross-chain monitor report."""
        chains = list(self._chain_profiles.values())
        tot_vol = sum(t.notional_usd for t in self._settled_transfers.values())

        # Scan active pairs for arbitrage
        opportunities: List[CrossChainRouteQuote] = []
        sample_pairs = [
            (SupportedChain.ETHEREUM, SupportedChain.ARBITRUM, "ETH", BridgeProtocol.ACROSS),
            (SupportedChain.ETHEREUM, SupportedChain.OPTIMISM, "ETH", BridgeProtocol.HOP),
            (SupportedChain.ARBITRUM, SupportedChain.BASE, "USDC", BridgeProtocol.CCTP),
            (SupportedChain.ETHEREUM, SupportedChain.SOLANA, "SOL", BridgeProtocol.WORMHOLE),
        ]

        for src, dst, tok, proto in sample_pairs:
            quote = self.evaluate_route_quote(src, dst, tok, proto)
            if quote.is_arbitrage_profitable:
                opportunities.append(quote)

        opportunities.sort(key=lambda q: q.net_arbitrage_profit_usd, reverse=True)
        top_route = opportunities[0].route_id if opportunities else None
        top_profit = opportunities[0].net_arbitrage_profit_usd if opportunities else 0.0

        return CrossChainMonitorReport(
            chains=chains,
            active_transfers_count=len(self._active_transfers),
            settled_transfers_count=len(self._settled_transfers),
            total_volume_bridged_usd=round(tot_vol, 2),
            arbitrage_opportunities=opportunities,
            top_profitable_route_id=top_route,
            highest_net_profit_usd=top_profit,
        )
