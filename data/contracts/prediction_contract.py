"""Trading Intelligence Platform - Canonical Prediction Contract (v1.0.0).

Defines an explicit, immutable, versioned schema (PRED-CONTRACT-v1.0.0) for every prediction
generated across supported markets (NSE, BSE, Quotex).

Ensures:
1. Canonical market and instrument identification (NSE:SYM, BSE:SYM, QUOTEX:SYM).
2. Strict temporal separation: Data Cutoff <= Creation Timestamp <= Execution <= Resolution Deadline.
3. Multiclass probability distribution (BUY, SELL, NO-TRADE) summing to 1.0.
4. Calibrated confidence and uncertainty tracking.
5. Explicit transaction friction and break-even probability hurdles (including Quotex binary payout modeling).
6. Cross-market incompatibility detection (preventing model from consuming wrong market/timeframe).
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime, timedelta
from enum import Enum
from typing import Any

from core.instrument import MarketType

PREDICTION_CONTRACT_SCHEMA_VERSION = "PRED-CONTRACT-v1.0.0"


class DecisionAction(str, Enum):
    BUY = "BUY"
    SELL = "SELL"
    NO_TRADE = "NO-TRADE"


class TargetDefinition(str, Enum):
    NEXT_BAR_DIRECTION = "next_bar_direction"
    TRIPLE_BARRIER = "triple_barrier"
    BINARY_OPTIONS_EXPIRY = "binary_options_expiry"


@dataclass(frozen=True)
class PredictionContractV1:
    """Immutable, versioned prediction contract adhering to PRED-CONTRACT-v1.0.0."""

    schema_version: str
    prediction_id: str
    instrument: str
    canonical_symbol: str
    market: str
    creation_timestamp_utc: str
    data_cutoff_timestamp_utc: str
    source_candle_interval: str
    prediction_horizon: str
    target_definition: str
    action: str
    buy_probability: float
    sell_probability: float
    no_trade_probability: float
    confidence: float
    uncertainty: float
    model_version: str
    feature_pipeline_version: str
    data_quality_status: str
    entry_price: float
    stop_loss: float
    take_profit: float
    risk_reward_ratio: float
    expected_friction_bps: float
    break_even_probability: float
    resolution_deadline_utc: str
    binary_payout_pct: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> PredictionContractV1:
        return cls(
            schema_version=str(d.get("schema_version", PREDICTION_CONTRACT_SCHEMA_VERSION)),
            prediction_id=str(d["prediction_id"]),
            instrument=str(d["instrument"]),
            canonical_symbol=str(d["canonical_symbol"]),
            market=str(d["market"]).upper(),
            creation_timestamp_utc=str(d["creation_timestamp_utc"]),
            data_cutoff_timestamp_utc=str(d["data_cutoff_timestamp_utc"]),
            source_candle_interval=str(d["source_candle_interval"]),
            prediction_horizon=str(d["prediction_horizon"]),
            target_definition=str(d.get("target_definition", TargetDefinition.TRIPLE_BARRIER.value)),
            action=str(d["action"]).upper(),
            buy_probability=float(d["buy_probability"]),
            sell_probability=float(d["sell_probability"]),
            no_trade_probability=float(d["no_trade_probability"]),
            confidence=float(d["confidence"]),
            uncertainty=float(d.get("uncertainty", 0.0)),
            model_version=str(d["model_version"]),
            feature_pipeline_version=str(d["feature_pipeline_version"]),
            data_quality_status=str(d.get("data_quality_status", "PASSED")),
            entry_price=float(d["entry_price"]),
            stop_loss=float(d["stop_loss"]),
            take_profit=float(d["take_profit"]),
            risk_reward_ratio=float(d.get("risk_reward_ratio", 0.0)),
            expected_friction_bps=float(d.get("expected_friction_bps", 6.0)),
            break_even_probability=float(d.get("break_even_probability", 0.50)),
            resolution_deadline_utc=str(d["resolution_deadline_utc"]),
            binary_payout_pct=float(d["binary_payout_pct"]) if d.get("binary_payout_pct") is not None else None,
            metadata=dict(d.get("metadata", {})),
        )


def validate_prediction_contract(
    contract: PredictionContractV1 | dict[str, Any],
) -> tuple[bool, str]:
    """Strictly validates invariant rules for PRED-CONTRACT-v1.0.0.
    
    Returns (is_valid, reason).
    """
    if isinstance(contract, dict):
        try:
            c = PredictionContractV1.from_dict(contract)
        except Exception as exc:
            return False, f"Contract deserialization error: {exc}"
    else:
        c = contract

    # 1. Schema version
    if c.schema_version != PREDICTION_CONTRACT_SCHEMA_VERSION:
        return False, f"Schema mismatch: expected {PREDICTION_CONTRACT_SCHEMA_VERSION}, got {c.schema_version}"

    # 2. Supported market check
    if c.market not in ("NSE", "BSE", "QUOTEX"):
        return False, f"Unsupported market '{c.market}': must be one of ('NSE', 'BSE', 'QUOTEX')"

    # 3. Canonical symbol match
    expected_prefix = f"{c.market}:"
    if not c.canonical_symbol.startswith(expected_prefix):
        return False, f"Canonical symbol '{c.canonical_symbol}' does not match market prefix '{expected_prefix}'"

    # 4. Action check
    if c.action not in ("BUY", "SELL", "NO-TRADE"):
        return False, f"Invalid decision action '{c.action}': must be BUY, SELL, or NO-TRADE"

    # 5. Probabilities validity & sum
    for p_name, p_val in [
        ("buy_probability", c.buy_probability),
        ("sell_probability", c.sell_probability),
        ("no_trade_probability", c.no_trade_probability),
    ]:
        if math.isnan(p_val) or math.isinf(p_val) or not (0.0 <= p_val <= 1.0):
            return False, f"Invalid {p_name}: {p_val} is not in [0.0, 1.0]"

    prob_sum = c.buy_probability + c.sell_probability + c.no_trade_probability
    if abs(prob_sum - 1.0) > 0.02:
        return False, f"Probabilities do not sum to 1.0: sum is {prob_sum:.4f}"

    # 6. Pricing validity
    if c.entry_price <= 0 or math.isnan(c.entry_price) or math.isinf(c.entry_price):
        return False, f"Invalid entry_price: {c.entry_price}"

    # 7. Temporal causality and sequence
    try:
        t_creation = datetime.fromisoformat(c.creation_timestamp_utc.replace("Z", "+00:00"))
        t_cutoff = datetime.fromisoformat(c.data_cutoff_timestamp_utc.replace("Z", "+00:00"))
        t_resolution = datetime.fromisoformat(c.resolution_deadline_utc.replace("Z", "+00:00"))
    except Exception as exc:
        return False, f"Timestamp parsing failed: {exc}"

    if t_cutoff > t_creation + timedelta(seconds=1):
        return False, f"Lookahead violation: data_cutoff ({t_cutoff}) > creation ({t_creation})"

    if t_resolution <= t_creation:
        return False, f"Resolution deadline ({t_resolution}) must be after creation ({t_creation})"

    # 8. Quotex binary options constraints
    if c.market == "QUOTEX":
        if c.binary_payout_pct is not None:
            if not (0.20 <= c.binary_payout_pct <= 1.00):
                return False, f"Quotex binary payout {c.binary_payout_pct} out of realistic range [0.20, 1.00]"
            expected_breakeven = 1.0 / (1.0 + c.binary_payout_pct)
            if abs(c.break_even_probability - expected_breakeven) > 0.05:
                return False, f"Quotex break_even_probability ({c.break_even_probability}) does not match payout {c.binary_payout_pct} (expected {expected_breakeven:.4f})"

    return True, "PASSED"
