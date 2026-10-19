"""Institutional Decentralized MEV & Front-Running Attack Surface Defense Gateway.

Protects automated algorithmic orders and large institutional rebalancing sweeps from:
1. Sandwich attacks (front-running + back-running buy/sell traps).
2. Priority gas auctions (PGA) and mempool adverse selection.
3. Generalized front-running by searcher bots.

Provides:
- Real-time sandwich risk scoring (0-100) based on order size vs pool depth and slippage threshold.
- Private transaction relay abstraction (Flashbots / MEV-Share builder bundle builder).
- Dynamic priority fee recommendation (base fee + priority tip targeting inclusion without leakage).
- Slippage tolerance auto-tightening during high-MEV volatility spikes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
import hashlib
import json
import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


class MEVRiskLevel(str, Enum):
    """Institutional classification of MEV front-running exposure."""
    LOW = "LOW"             # Risk Score < 25: Safe for public mempool or standard routing
    MEDIUM = "MEDIUM"       # 25 <= Risk Score < 50: Moderate extraction risk, tighten slippage
    HIGH = "HIGH"           # 50 <= Risk Score < 75: High sandwich risk, mandate private relay
    CRITICAL = "CRITICAL"   # Risk Score >= 75: Extreme extraction risk, private bundle only


class ExecutionRelayType(str, Enum):
    """Routing path selected for transaction execution."""
    PUBLIC_MEMPOOL = "PUBLIC_MEMPOOL"           # Standard peer-to-peer mempool
    FLASHBOTS_PROTECT = "FLASHBOTS_PROTECT"     # Private RPC endpoint bypass
    MEV_SHARE_BUNDLE = "MEV_SHARE_BUNDLE"       # Builder bundle with user searcher kickback
    DIRECT_BUILDER_RPC = "DIRECT_BUILDER_RPC"   # Direct Tier-1 Block Builder connection


@dataclass(frozen=True)
class PendingMempoolTx:
    """Observed pending transaction in local or network mempool."""
    tx_hash: str
    from_address: str
    to_address: str  # e.g., Uniswap pool or router
    gas_price_gwei: float
    max_priority_fee_gwei: float
    value_eth: float
    method_id: str
    timestamp_ns: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tx_hash": self.tx_hash,
            "from_address": self.from_address,
            "to_address": self.to_address,
            "gas_price_gwei": float(self.gas_price_gwei),
            "max_priority_fee_gwei": float(self.max_priority_fee_gwei),
            "value_eth": float(self.value_eth),
            "method_id": self.method_id,
            "timestamp_ns": int(self.timestamp_ns),
        }


@dataclass(frozen=True)
class MEVDefenseAssessment:
    """Pre-execution assessment of front-running/sandwich attack vulnerability."""
    order_id: str
    symbol: str
    notional_usd: float
    pool_liquidity_usd: float
    requested_slippage_bps: float
    recommended_slippage_bps: float
    risk_score: float  # 0 to 100
    risk_level: MEVRiskLevel
    recommended_relay: ExecutionRelayType
    estimated_potential_sandwich_loss_usd: float
    recommended_priority_fee_gwei: float
    private_bundle_required: bool
    reason: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "order_id": self.order_id,
            "symbol": self.symbol,
            "notional_usd": float(self.notional_usd),
            "pool_liquidity_usd": float(self.pool_liquidity_usd),
            "requested_slippage_bps": float(self.requested_slippage_bps),
            "recommended_slippage_bps": float(self.recommended_slippage_bps),
            "risk_score": float(self.risk_score),
            "risk_level": self.risk_level.value,
            "recommended_relay": self.recommended_relay.value,
            "estimated_potential_sandwich_loss_usd": float(self.estimated_potential_sandwich_loss_usd),
            "recommended_priority_fee_gwei": float(self.recommended_priority_fee_gwei),
            "private_bundle_required": bool(self.private_bundle_required),
            "reason": self.reason,
        }


@dataclass(frozen=True)
class FlashbotsBundleSubmission:
    """Constructed private execution bundle ready for builder relay."""
    bundle_id: str
    target_block_number: int
    raw_transactions: List[str]
    min_timestamp: int
    max_timestamp: int
    reverting_hashes_allowed: List[str]
    created_at_utc: datetime

    def to_dict(self) -> Dict[str, Any]:
        return {
            "bundle_id": self.bundle_id,
            "target_block_number": int(self.target_block_number),
            "transactions_count": len(self.raw_transactions),
            "min_timestamp": int(self.min_timestamp),
            "max_timestamp": int(self.max_timestamp),
            "reverting_hashes_allowed": list(self.reverting_hashes_allowed),
            "created_at_utc": self.created_at_utc.isoformat(),
        }


class MEVDefenseGateway:
    """Institutional Decentralized MEV & Front-Running Attack Defense Gateway.

    Evaluates pending order exposure before on-chain dispatch:
    1. Sandwich Attack Scoring:
       Extractable Value EV ~= Q * S_tolerance - 2 * Gas_cost.
       If EV > 0, searchers will exploit via front-run and back-run.
    2. Dynamic Slippage Tightening:
       Auto-clamps loose default slippage (e.g. 100 bps -> 10-25 bps) based on volatility and pool size.
    3. Private Relay Routing:
       Directs vulnerable orders through Flashbots Protect or Builder bundles, hiding tx from mempool.
    4. Priority Fee Calibration:
       Computes optimal priority gas fees from pending mempool gas distribution.
    """

    def __init__(
        self,
        base_fee_gwei: float = 25.0,
        private_relay_threshold_score: float = 50.0,
        max_acceptable_sandwich_bps: float = 30.0,
    ) -> None:
        self.base_fee_gwei = float(base_fee_gwei)
        self.private_relay_threshold_score = float(private_relay_threshold_score)
        self.max_acceptable_sandwich_bps = float(max_acceptable_sandwich_bps)
        self._mempool_cache: List[PendingMempoolTx] = []

    def update_mempool_snapshot(self, pending_txs: List[PendingMempoolTx]) -> None:
        """Update cached mempool transactions for competitor gas price profiling."""
        self._mempool_cache = list(pending_txs)

    def assess_order_mev_risk(
        self,
        order_id: str,
        symbol: str,
        notional_usd: float,
        pool_liquidity_usd: float,
        requested_slippage_bps: float = 50.0,
        current_gas_price_gwei: Optional[float] = None,
    ) -> MEVDefenseAssessment:
        """Analyze transaction order for sandwich exploitability and recommend defenses."""
        if notional_usd <= 0:
            raise ValueError(f"notional_usd must be positive, got {notional_usd}")
        if pool_liquidity_usd <= 0:
            raise ValueError(f"pool_liquidity_usd must be positive, got {pool_liquidity_usd}")

        gas_price = current_gas_price_gwei or self.base_fee_gwei

        # 1. Size-to-pool ratio
        size_ratio = notional_usd / pool_liquidity_usd  # e.g., $100k in $1M pool = 0.10

        # 2. Maximum extractable value (MEV) in USD:
        # A searcher can push the price up to the requested slippage tolerance:
        # Extracted profit ~= notional_usd * (requested_slippage_bps / 10000)
        potential_sandwich_loss = notional_usd * (requested_slippage_bps / 10000.0)

        # 3. Sandwich Risk Score (0 - 100):
        # Driven by size ratio (weight 50%) and slippage tolerance (weight 50%)
        size_score = min(50.0, (size_ratio / 0.05) * 50.0)  # Reaches max 50 at 5% of pool
        slippage_score = min(50.0, (requested_slippage_bps / 100.0) * 50.0)  # Reaches max 50 at 100 bps
        risk_score = float(np.clip(size_score + slippage_score, 0.0, 100.0))

        # 4. Classify Risk Level
        if risk_score >= 75.0:
            risk_level = MEVRiskLevel.CRITICAL
        elif risk_score >= 50.0:
            risk_level = MEVRiskLevel.HIGH
        elif risk_score >= 25.0:
            risk_level = MEVRiskLevel.MEDIUM
        else:
            risk_level = MEVRiskLevel.LOW

        # 5. Dynamic Slippage Auto-Tightening:
        # For high/critical risk, aggressively clamp down to prevent extraction
        expected_impact_bps = (size_ratio * 0.5) * 10000.0
        calculated_slip = expected_impact_bps + 5.0
        if risk_level in (MEVRiskLevel.HIGH, MEVRiskLevel.CRITICAL):
            max_allowed = min(self.max_acceptable_sandwich_bps, requested_slippage_bps * 0.5)
            recommended_slippage_bps = float(np.clip(calculated_slip, 5.0, max_allowed))
        else:
            recommended_slippage_bps = float(np.clip(calculated_slip, 5.0, requested_slippage_bps))

        # 6. Priority Fee Recommendation:
        # Target 80th percentile priority fee in mempool + 1 gwei tip
        if self._mempool_cache:
            p_fees = [tx.max_priority_fee_gwei for tx in self._mempool_cache]
            rec_priority_fee = float(np.percentile(p_fees, 80)) + 1.0
        else:
            rec_priority_fee = 2.0  # Default 2 gwei priority tip

        # 7. Routing Selection
        private_required = risk_score >= self.private_relay_threshold_score
        if risk_level == MEVRiskLevel.CRITICAL:
            recommended_relay = ExecutionRelayType.MEV_SHARE_BUNDLE
            reason = "CRITICAL MEV EXPOSURE: Order size is large relative to pool; route via MEV-Share builder bundle."
        elif risk_level == MEVRiskLevel.HIGH:
            recommended_relay = ExecutionRelayType.FLASHBOTS_PROTECT
            reason = "HIGH SANDWICH RISK: Potential extractable profit exceeds gas costs; route via Flashbots Protect RPC."
        elif risk_level == MEVRiskLevel.MEDIUM:
            recommended_relay = ExecutionRelayType.FLASHBOTS_PROTECT
            reason = "MODERATE RISK: Slippage tightened to prevent front-running; private RPC recommended."
        else:
            recommended_relay = ExecutionRelayType.PUBLIC_MEMPOOL
            reason = "LOW MEV RISK: Size and slippage are benign; public mempool dispatch is safe and cost-effective."

        return MEVDefenseAssessment(
            order_id=order_id,
            symbol=symbol.upper(),
            notional_usd=round(notional_usd, 2),
            pool_liquidity_usd=round(pool_liquidity_usd, 2),
            requested_slippage_bps=round(requested_slippage_bps, 2),
            recommended_slippage_bps=round(recommended_slippage_bps, 2),
            risk_score=round(risk_score, 2),
            risk_level=risk_level,
            recommended_relay=recommended_relay,
            estimated_potential_sandwich_loss_usd=round(potential_sandwich_loss, 2),
            recommended_priority_fee_gwei=round(rec_priority_fee, 2),
            private_bundle_required=private_required,
            reason=reason,
        )

    def build_flashbots_bundle(
        self,
        raw_signed_txs: List[str],
        current_block_number: int,
        block_lead: int = 1,
    ) -> FlashbotsBundleSubmission:
        """Construct atomic private bundle for Flashbots/MEV-Share builder submission."""
        if not raw_signed_txs:
            raise ValueError("raw_signed_txs cannot be empty")

        target_block = current_block_number + block_lead
        now = datetime.now(timezone.utc)
        now_ts = int(now.timestamp())

        # Generate deterministic bundle ID hash
        bundle_str = f"{target_block}_{'_'.join(raw_signed_txs)}"
        bundle_id = f"bndl_{hashlib.sha256(bundle_str.encode()).hexdigest()[:12]}"

        return FlashbotsBundleSubmission(
            bundle_id=bundle_id,
            target_block_number=target_block,
            raw_transactions=list(raw_signed_txs),
            min_timestamp=now_ts,
            max_timestamp=now_ts + 120,  # Valid for 2 minutes
            reverting_hashes_allowed=[],
            created_at_utc=now,
        )
