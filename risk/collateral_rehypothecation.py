"""Trading Intelligence Platform - Dynamic Collateral Rehypothecation & Multi-Venue Liquidity Balancing Engine.

Institutional multi-venue capital optimization, automated margin compression relief,
haircut-adjusted liquidity transfers, and minimum-cost collateral routing across
exchanges, prime brokers, and clearing networks.
"""

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional
import uuid
import numpy as np
from pydantic import BaseModel, Field

from core.logging import get_logger

logger = get_logger("collateral_rehypothecation")


class TransferUrgency(str, Enum):
    """Urgency level of a collateral rehypothecation transfer."""

    ROUTINE = "ROUTINE"                                         # Routine capital balance leveling
    ELEVATED = "ELEVATED"                                       # Venue approaching warning margin utilization
    CRITICAL_MARGIN_CALL = "CRITICAL_MARGIN_CALL"               # Urgent transfer to prevent formal margin call
    EMERGENCY_LIQUIDATION_PREVENTION = "EMERGENCY_PREVENTION"  # Immediate intervention to avoid liquidation breach


class TransferStatus(str, Enum):
    """Execution status of an inter-venue collateral transfer."""

    PENDING = "PENDING"
    IN_FLIGHT = "IN_FLIGHT"
    CONFIRMED = "CONFIRMED"
    FAILED = "FAILED"
    REJECTED = "REJECTED"


class VenueCollateralAsset(BaseModel):
    """Individual collateral holding at a specific venue."""

    asset: str
    quantity: float
    mark_price: float
    haircut_pct: float = 0.0

    @property
    def nominal_value_usd(self) -> float:
        return self.quantity * self.mark_price

    @property
    def effective_collateral_usd(self) -> float:
        return self.nominal_value_usd * max(0.0, 1.0 - self.haircut_pct)


class VenueCollateralState(BaseModel):
    """Comprehensive margin and collateral standing of a trading venue or custodian."""

    venue_id: str
    venue_type: str = "EXCHANGE"  # "EXCHANGE", "PRIME_BROKER", "CUSTODIAN_VAULT", "CLEARING_NETWORK"
    assets: Dict[str, VenueCollateralAsset] = {}
    cash_usd: float = 0.0
    margin_required_usd: float = 0.0
    maintenance_margin_required_usd: float = 0.0
    target_utilization_pct: float = 65.0
    warning_utilization_pct: float = 80.0
    critical_utilization_pct: float = 90.0
    unencumbered_reserve_buffer_usd: float = 10000.0
    transfer_latency_seconds: float = 300.0  # Default 5 mins for on-chain/clearing transfer
    transfer_fee_flat_usd: float = 15.0

    @property
    def total_nominal_equity_usd(self) -> float:
        return self.cash_usd + sum(a.nominal_value_usd for a in self.assets.values())

    @property
    def total_effective_collateral_usd(self) -> float:
        return self.cash_usd + sum(a.effective_collateral_usd for a in self.assets.values())

    @property
    def margin_utilization_pct(self) -> float:
        eff = self.total_effective_collateral_usd
        if eff <= 0:
            return 100.0 if self.margin_required_usd > 0 else 0.0
        return (self.margin_required_usd / eff) * 100.0

    @property
    def free_collateral_usd(self) -> float:
        return max(0.0, self.total_effective_collateral_usd - self.margin_required_usd)

    @property
    def unencumbered_excess_usd(self) -> float:
        """Capital available to lend/transfer without breaching target utilization or reserve buffer."""
        eff = self.total_effective_collateral_usd
        if eff <= 0:
            return 0.0
        # Collateral required to maintain target utilization:
        # required_eff = margin_required / (target_util / 100)
        target_fraction = max(0.10, self.target_utilization_pct / 100.0)
        needed_collateral = self.margin_required_usd / target_fraction
        surplus_over_target = eff - needed_collateral
        surplus_over_buffer = self.free_collateral_usd - self.unencumbered_reserve_buffer_usd
        return max(0.0, min(surplus_over_target, surplus_over_buffer))

    @property
    def liquidation_distance_usd(self) -> float:
        """Dollar buffer remaining before maintenance margin liquidation threshold is hit."""
        return max(0.0, self.total_effective_collateral_usd - self.maintenance_margin_required_usd)

    @property
    def health_classification(self) -> str:
        util = self.margin_utilization_pct
        if util >= self.critical_utilization_pct:
            return "CRITICAL_MARGIN_CALL"
        if util >= self.warning_utilization_pct:
            return "WARNING"
        if util >= self.target_utilization_pct:
            return "ELEVATED"
        return "HEALTHY"


class CollateralTransferLeg(BaseModel):
    """Specific asset transfer leg routed between two venues."""

    transfer_id: str = Field(default_factory=lambda: f"xfer_{uuid.uuid4().hex[:8]}")
    source_venue: str
    target_venue: str
    asset: str
    quantity: float
    amount_usd: float
    estimated_fee_usd: float
    estimated_latency_seconds: float
    urgency: TransferUrgency
    status: TransferStatus = TransferStatus.PENDING
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    executed_at: Optional[datetime] = None


class CollateralRebalancePlan(BaseModel):
    """Institutional collateral rehypothecation and liquidity rebalancing plan."""

    plan_id: str = Field(default_factory=lambda: f"rebal_{uuid.uuid4().hex[:8]}")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    total_transfers_count: int = 0
    total_capital_rebalanced_usd: float = 0.0
    total_transfer_fees_usd: float = 0.0
    venue_initial_utilizations: Dict[str, float] = {}
    venue_projected_utilizations: Dict[str, float] = {}
    transfer_legs: List[CollateralTransferLeg] = []
    recommended_action: str = "NO_ACTION"
    is_emergency: bool = False


class CollateralRehypothecationEngine:
    """Institutional multi-venue collateral optimizer and automated liquidity rebalancer."""

    DEFAULT_HAIRCUTS: Dict[str, float] = {
        "USD": 0.0,
        "USDC": 0.005,
        "USDT": 0.01,
        "BTC": 0.10,
        "ETH": 0.15,
        "SOL": 0.25,
        "DEFAULT": 0.35,
    }

    def __init__(
        self,
        venues: Optional[Dict[str, VenueCollateralState]] = None,
        custom_haircuts: Optional[Dict[str, float]] = None,
    ) -> None:
        self.venues: Dict[str, VenueCollateralState] = venues or {}
        self.haircuts = {**self.DEFAULT_HAIRCUTS, **(custom_haircuts or {})}
        self.transfer_history: List[CollateralTransferLeg] = []

    def register_venue(self, venue: VenueCollateralState) -> None:
        """Register or update a trading venue state."""
        self.venues[venue.venue_id] = venue

    def get_haircut(self, asset: str) -> float:
        """Retrieve applicable haircut rate for an asset."""
        return self.haircuts.get(asset.upper(), self.haircuts["DEFAULT"])

    def evaluate_all_venues(self) -> Dict[str, Dict[str, Any]]:
        """Compute comprehensive margin health summary across all registered venues."""
        summary = {}
        for vid, v in self.venues.items():
            summary[vid] = {
                "venue_id": vid,
                "venue_type": v.venue_type,
                "total_nominal_equity_usd": round(v.total_nominal_equity_usd, 2),
                "total_effective_collateral_usd": round(v.total_effective_collateral_usd, 2),
                "margin_required_usd": round(v.margin_required_usd, 2),
                "margin_utilization_pct": round(v.margin_utilization_pct, 2),
                "free_collateral_usd": round(v.free_collateral_usd, 2),
                "unencumbered_excess_usd": round(v.unencumbered_excess_usd, 2),
                "liquidation_distance_usd": round(v.liquidation_distance_usd, 2),
                "health_status": v.health_classification,
            }
        return summary

    def generate_rebalance_plan(self) -> CollateralRebalancePlan:
        """Synthesize an optimal multi-venue collateral rebalancing plan.

        Identifies compressed venues (utilization > target_utilization_pct) and donor venues
        with surplus unencumbered collateral. Formulates prioritized transfer legs routing
        liquid stablecoins/USD first to minimize haircut drag and transfer latency.
        """
        if not self.venues:
            return CollateralRebalancePlan()

        initial_utils = {vid: round(v.margin_utilization_pct, 2) for vid, v in self.venues.items()}
        projected_venues: Dict[str, Dict[str, Any]] = {}
        for vid, v in self.venues.items():
            projected_venues[vid] = {
                "cash_usd": v.cash_usd,
                "margin_req": v.margin_required_usd,
                "target_util": v.target_utilization_pct,
                "effective_collateral": v.total_effective_collateral_usd,
                "fee_flat": v.transfer_fee_flat_usd,
                "latency": v.transfer_latency_seconds,
            }

        # Classify venues into Deficit (needing collateral) and Surplus (capable of supplying)
        deficit_venues: List[str] = []
        surplus_venues: List[str] = []

        is_emergency = False

        for vid, v in self.venues.items():
            util = v.margin_utilization_pct
            if util >= v.critical_utilization_pct:
                is_emergency = True
                deficit_venues.append(vid)
            elif util > v.target_utilization_pct:
                deficit_venues.append(vid)
            elif v.unencumbered_excess_usd > 100.0:
                surplus_venues.append(vid)

        # Sort deficit venues by descending utilization (most urgent first)
        deficit_venues.sort(key=lambda vid: self.venues[vid].margin_utilization_pct, reverse=True)
        # Sort surplus venues by descending unencumbered excess (most liquid donors first)
        surplus_venues.sort(key=lambda vid: self.venues[vid].unencumbered_excess_usd, reverse=True)

        transfer_legs: List[CollateralTransferLeg] = []
        total_rebalanced_usd = 0.0
        total_fees_usd = 0.0

        for def_id in deficit_venues:
            def_v = self.venues[def_id]
            curr_eff = projected_venues[def_id]["effective_collateral"]
            margin_req = projected_venues[def_id]["margin_req"]
            target_frac = max(0.1, def_v.target_utilization_pct / 100.0)

            # Target effective collateral needed to bring utilization down to target:
            desired_eff = margin_req / target_frac
            needed_collateral_usd = max(0.0, desired_eff - curr_eff)

            if needed_collateral_usd <= 10.0:
                continue

            # Determine transfer urgency
            curr_util = (margin_req / curr_eff * 100.0) if curr_eff > 0 else 100.0
            if curr_util >= def_v.critical_utilization_pct:
                urgency = TransferUrgency.EMERGENCY_LIQUIDATION_PREVENTION
            elif curr_util >= def_v.warning_utilization_pct:
                urgency = TransferUrgency.CRITICAL_MARGIN_CALL
            elif curr_util > def_v.target_utilization_pct:
                urgency = TransferUrgency.ELEVATED
            else:
                urgency = TransferUrgency.ROUTINE

            # Source collateral from available surplus venues
            for sur_id in surplus_venues:
                if needed_collateral_usd <= 10.0:
                    break
                sur_v = self.venues[sur_id]
                donor_avail_usd = projected_venues[sur_id]["cash_usd"]

                if donor_avail_usd <= 100.0:
                    continue

                transfer_amount = min(needed_collateral_usd, donor_avail_usd * 0.85)  # Preserve 15% safety buffer
                if transfer_amount < 50.0:
                    continue

                fee = sur_v.transfer_fee_flat_usd
                latency = sur_v.transfer_latency_seconds

                # Create transfer leg
                leg = CollateralTransferLeg(
                    source_venue=sur_id,
                    target_venue=def_id,
                    asset="USD",
                    quantity=transfer_amount,
                    amount_usd=transfer_amount,
                    estimated_fee_usd=fee,
                    estimated_latency_seconds=latency,
                    urgency=urgency,
                    status=TransferStatus.PENDING,
                )
                transfer_legs.append(leg)

                # Update projected balances
                projected_venues[sur_id]["cash_usd"] -= (transfer_amount + fee)
                projected_venues[sur_id]["effective_collateral"] -= (transfer_amount + fee)
                projected_venues[def_id]["cash_usd"] += transfer_amount
                projected_venues[def_id]["effective_collateral"] += transfer_amount

                needed_collateral_usd -= transfer_amount
                total_rebalanced_usd += transfer_amount
                total_fees_usd += fee

        # Compute projected utilizations
        projected_utils = {}
        for vid in self.venues:
            p_eff = projected_venues[vid]["effective_collateral"]
            p_req = projected_venues[vid]["margin_req"]
            if p_eff <= 0:
                p_util = 100.0 if p_req > 0 else 0.0
            else:
                p_util = round((p_req / p_eff) * 100.0, 2)
            projected_utils[vid] = p_util

        action = "DISPATCH_TRANSFERS" if transfer_legs else "HOLD_BALANCES"

        return CollateralRebalancePlan(
            total_transfers_count=len(transfer_legs),
            total_capital_rebalanced_usd=round(total_rebalanced_usd, 2),
            total_transfer_fees_usd=round(total_fees_usd, 2),
            venue_initial_utilizations=initial_utils,
            venue_projected_utilizations=projected_utils,
            transfer_legs=transfer_legs,
            recommended_action=action,
            is_emergency=is_emergency,
        )

    def execute_transfer_plan(
        self,
        plan: CollateralRebalancePlan,
        dry_run: bool = False,
    ) -> List[CollateralTransferLeg]:
        """Execute or dry-run simulated execution of all transfer legs in the rebalance plan."""
        executed_legs: List[CollateralTransferLeg] = []

        for leg in plan.transfer_legs:
            if leg.source_venue not in self.venues or leg.target_venue not in self.venues:
                leg.status = TransferStatus.REJECTED
                executed_legs.append(leg)
                continue

            src = self.venues[leg.source_venue]
            dst = self.venues[leg.target_venue]

            # Verify source solvency
            if leg.asset == "USD":
                if src.cash_usd < (leg.amount_usd + leg.estimated_fee_usd):
                    leg.status = TransferStatus.FAILED
                    executed_legs.append(leg)
                    continue

                if not dry_run:
                    src.cash_usd -= (leg.amount_usd + leg.estimated_fee_usd)
                    dst.cash_usd += leg.amount_usd
            else:
                if leg.asset not in src.assets or src.assets[leg.asset].quantity < leg.quantity:
                    leg.status = TransferStatus.FAILED
                    executed_legs.append(leg)
                    continue

                if not dry_run:
                    src.assets[leg.asset].quantity -= leg.quantity
                    src.cash_usd -= leg.estimated_fee_usd
                    if leg.asset in dst.assets:
                        dst.assets[leg.asset].quantity += leg.quantity
                    else:
                        haircut = self.get_haircut(leg.asset)
                        mark = src.assets[leg.asset].mark_price
                        dst.assets[leg.asset] = VenueCollateralAsset(
                            asset=leg.asset,
                            quantity=leg.quantity,
                            mark_price=mark,
                            haircut_pct=haircut,
                        )

            leg.status = TransferStatus.CONFIRMED
            leg.executed_at = datetime.now(timezone.utc)
            executed_legs.append(leg)
            self.transfer_history.append(leg)
            logger.info(
                f"Collateral transfer executed: {leg.source_venue} -> {leg.target_venue} "
                f"[{leg.quantity} {leg.asset} = ${leg.amount_usd:.2f}], urgency={leg.urgency}"
            )

        return executed_legs
