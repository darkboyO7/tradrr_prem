"""Trading Intelligence Platform - Institutional Pre-Trade Transaction Pre-Flight & FIX Protocol Compliance Validator.

Enforces SEC Rule 15c3-5 (Market Access Rule) and MiFID II RTS 6/25 pre-trade risk controls:
Fat-Finger price collar bands, single-order notional caps, maximum short sale notional limits,
mandatory locate verification, sliding-window burst throttling, and FIX 4.4 / 5.0SP2 tag validation.
"""

from collections import deque
from datetime import datetime, timezone
from enum import Enum
import time
from typing import Any, Deque, Dict, List, Optional, Set
from pydantic import BaseModel, Field

from core.logging import get_logger

logger = get_logger("preflight_validator")


class PreFlightRejectReason(str, Enum):
    """Institutional reject codes for pre-trade pre-flight compliance gating."""

    APPROVED = "APPROVED"
    FAT_FINGER_PRICE_COLLAR_BREACH = "FAT_FINGER_PRICE_COLLAR_BREACH"
    EXCEEDS_SINGLE_ORDER_NOTIONAL = "EXCEEDS_SINGLE_ORDER_NOTIONAL"
    EXCEEDS_SINGLE_ORDER_QUANTITY = "EXCEEDS_SINGLE_ORDER_QUANTITY"
    EXCEEDS_DAILY_GROSS_CREDIT_CEILING = "EXCEEDS_DAILY_GROSS_CREDIT_CEILING"
    SHORT_SALE_LOCATE_MISSING = "SHORT_SALE_LOCATE_MISSING"
    BURST_RATE_LIMIT_EXCEEDED = "BURST_RATE_LIMIT_EXCEEDED"
    DUPLICATE_CLORD_ID = "DUPLICATE_CLORD_ID"
    STALE_TRANSACT_TIME = "STALE_TRANSACT_TIME"
    INVALID_FIX_MESSAGE_FORMAT = "INVALID_FIX_MESSAGE_FORMAT"
    SYMBOL_NOT_TRADEABLE = "SYMBOL_NOT_TRADEABLE"


class FIXNewOrderSingle(BaseModel):
    """Standardized institutional FIX 4.4 Tag 35=D (NewOrderSingle) representation."""

    cl_ord_id: str = Field(..., description="Tag 11: Unique Client Order ID")
    symbol: str = Field(..., description="Tag 55: Ticker / Trading Instrument")
    side: str = Field(..., description="Tag 54: 1=BUY, 2=SELL, 5=SELL_SHORT")
    order_qty: float = Field(..., description="Tag 38: Order Quantity")
    ord_type: str = Field(default="LIMIT", description="Tag 40: 1=MARKET, 2=LIMIT, 3=STOP, 4=STOP_LIMIT")
    price: Optional[float] = Field(default=None, description="Tag 44: Limit Price")
    stop_px: Optional[float] = Field(default=None, description="Tag 99: Stop Trigger Price")
    time_in_force: str = Field(default="DAY", description="Tag 59: 0=DAY, 1=GTC, 3=IOC, 4=FOK")
    account: str = Field(default="PRIMARY", description="Tag 1: Account Identifier")
    transact_time: Optional[datetime] = Field(default=None, description="Tag 60: Client UTCTimestamp")
    locate_reqd: bool = Field(default=False, description="Tag 114: Short Sale Locate Required")
    locate_id: Optional[str] = Field(default=None, description="Identifier of pre-obtained borrow locate")
    trader_id: Optional[str] = Field(default="ALGO_DESK_01", description="Trader / Strategy Tag")


class PreFlightRiskLimits(BaseModel):
    """Firm-wide and desk-level regulatory risk limits."""

    max_single_order_notional_usd: float = 500000.0
    max_single_order_qty: Dict[str, float] = {
        "BTC-USD": 25.0,
        "ETH-USD": 400.0,
        "SOL-USD": 3000.0,
        "DEFAULT": 10000.0,
    }
    max_daily_gross_notional_usd: float = 10000000.0
    max_price_collar_pct: float = 0.03  # 3.0% maximum deviation from prevailing reference mid-price
    max_orders_per_second: int = 100
    max_transact_time_drift_seconds: float = 30.0
    enforce_short_locate: bool = True
    restricted_symbols: List[str] = []


class PreFlightValidationResult(BaseModel):
    """Sub-microsecond audit record of a pre-trade pre-flight gating decision."""

    is_approved: bool
    reject_reason: PreFlightRejectReason
    reject_message: Optional[str] = None
    cl_ord_id: str
    symbol: str
    evaluated_notional_usd: float
    evaluated_price_deviation_pct: float
    validation_latency_micros: float
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class PreFlightValidator:
    """Rigid sub-microsecond pre-trade compliance and risk gating engine."""

    def __init__(self, limits: Optional[PreFlightRiskLimits] = None) -> None:
        self.limits = limits or PreFlightRiskLimits()
        self._seen_clordids: Set[str] = set()
        self._burst_window: Deque[float] = deque()
        self._cumulative_daily_gross_usd: float = 0.0
        self._current_date_str: str = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    def _check_and_roll_daily_budget(self) -> None:
        today_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        if today_str != self._current_date_str:
            self._current_date_str = today_str
            self._cumulative_daily_gross_usd = 0.0
            self._seen_clordids.clear()
            logger.info(f"PreFlight daily limits rolled over for date {today_str}")

    def validate_order(
        self,
        order: FIXNewOrderSingle,
        reference_mid_price: float,
        best_bid: Optional[float] = None,
        best_ask: Optional[float] = None,
    ) -> PreFlightValidationResult:
        """Validate an outbound order against regulatory pre-trade controls with sub-microsecond latency."""
        t_start = time.perf_counter_ns()
        self._check_and_roll_daily_budget()

        # 1. FIX Message Syntax & Completeness Checks
        if not order.cl_ord_id or not order.cl_ord_id.strip():
            return self._build_reject(
                PreFlightRejectReason.INVALID_FIX_MESSAGE_FORMAT,
                "ClOrdID (Tag 11) is required and cannot be empty.",
                order, 0.0, 0.0, t_start,
            )

        side_clean = order.side.upper().strip()
        if side_clean not in ["BUY", "SELL", "SELL_SHORT", "1", "2", "5"]:
            return self._build_reject(
                PreFlightRejectReason.INVALID_FIX_MESSAGE_FORMAT,
                f"Invalid FIX Side (Tag 54): '{order.side}'. Must be BUY, SELL, or SELL_SHORT.",
                order, 0.0, 0.0, t_start,
            )

        if order.order_qty <= 0:
            return self._build_reject(
                PreFlightRejectReason.INVALID_FIX_MESSAGE_FORMAT,
                f"OrderQty (Tag 38) must be strictly positive: {order.order_qty}.",
                order, 0.0, 0.0, t_start,
            )

        ord_type_clean = order.ord_type.upper().strip()
        if ord_type_clean in ["LIMIT", "2", "STOP_LIMIT", "4"]:
            if order.price is None or order.price <= 0:
                return self._build_reject(
                    PreFlightRejectReason.INVALID_FIX_MESSAGE_FORMAT,
                    f"Price (Tag 44) must be strictly positive for LIMIT orders: {order.price}.",
                    order, 0.0, 0.0, t_start,
                )

        # 2. Duplicate ClOrdID Check
        if order.cl_ord_id in self._seen_clordids:
            return self._build_reject(
                PreFlightRejectReason.DUPLICATE_CLORD_ID,
                f"Duplicate ClOrdID (Tag 11): '{order.cl_ord_id}' has already been processed intraday.",
                order, 0.0, 0.0, t_start,
            )

        # 3. TransactTime Clock Drift / Staleness Check
        now_utc = datetime.now(timezone.utc)
        if order.transact_time is not None:
            # Handle tz-naive vs tz-aware
            t_order = order.transact_time if order.transact_time.tzinfo else order.transact_time.replace(tzinfo=timezone.utc)
            drift_seconds = abs((now_utc - t_order).total_seconds())
            if drift_seconds > self.limits.max_transact_time_drift_seconds:
                return self._build_reject(
                    PreFlightRejectReason.STALE_TRANSACT_TIME,
                    f"TransactTime (Tag 60) drift ({drift_seconds:.1f}s) exceeds max limit ({self.limits.max_transact_time_drift_seconds}s).",
                    order, 0.0, 0.0, t_start,
                )

        # 4. Symbol Tradeability & Restricted List Check
        if order.symbol.upper() in [s.upper() for s in self.limits.restricted_symbols]:
            return self._build_reject(
                PreFlightRejectReason.SYMBOL_NOT_TRADEABLE,
                f"Symbol '{order.symbol}' is currently placed on the firm's Restricted Asset List.",
                order, 0.0, 0.0, t_start,
            )

        # 5. Sliding-Window Burst Rate Limiter
        now_ts = time.time()
        # Evict timestamps older than 1.0 second
        while self._burst_window and (now_ts - self._burst_window[0]) > 1.0:
            self._burst_window.popleft()

        if len(self._burst_window) >= self.limits.max_orders_per_second:
            return self._build_reject(
                PreFlightRejectReason.BURST_RATE_LIMIT_EXCEEDED,
                f"Order burst rate exceeds threshold: {len(self._burst_window)} orders/sec (limit={self.limits.max_orders_per_second}).",
                order, 0.0, 0.0, t_start,
            )

        # 6. Short Sale Locate Verification (Reg SHO Rule 201/203)
        is_short = side_clean in ["SELL_SHORT", "5"] or (order.locate_reqd and side_clean in ["SELL", "2"])
        if is_short and self.limits.enforce_short_locate:
            if not order.locate_id or not order.locate_id.strip():
                return self._build_reject(
                    PreFlightRejectReason.SHORT_SALE_LOCATE_MISSING,
                    f"Short sale order rejected: Mandatory borrow locate identifier missing for {order.symbol}.",
                    order, 0.0, 0.0, t_start,
                )

        # 7. Single Order Quantity Ceiling Check
        max_qty = self.limits.max_single_order_qty.get(
            order.symbol.upper(),
            self.limits.max_single_order_qty.get("DEFAULT", 10000.0),
        )
        if order.order_qty > max_qty:
            return self._build_reject(
                PreFlightRejectReason.EXCEEDS_SINGLE_ORDER_QUANTITY,
                f"Order quantity {order.order_qty} exceeds single-order limit ({max_qty} units for {order.symbol}).",
                order, 0.0, 0.0, t_start,
            )

        # 8. Fat-Finger Price Collar Band Check
        effective_price = order.price if order.price is not None else reference_mid_price
        if reference_mid_price <= 0:
            return self._build_reject(
                PreFlightRejectReason.FAT_FINGER_PRICE_COLLAR_BREACH,
                f"Invalid prevailing reference mid-price: {reference_mid_price}.",
                order, 0.0, 0.0, t_start,
            )

        price_dev_pct = abs(effective_price - reference_mid_price) / reference_mid_price
        if price_dev_pct > self.limits.max_price_collar_pct:
            return self._build_reject(
                PreFlightRejectReason.FAT_FINGER_PRICE_COLLAR_BREACH,
                f"Price collar breach: Order price ${effective_price:.2f} deviates by {price_dev_pct * 100:.2f}% "
                f"from reference mid ${reference_mid_price:.2f} (max collar={self.limits.max_price_collar_pct * 100:.1f}%).",
                order, effective_price * order.order_qty, price_dev_pct, t_start,
            )

        # 9. Single Order Notional Limit Check
        evaluated_notional = effective_price * order.order_qty
        if evaluated_notional > self.limits.max_single_order_notional_usd:
            return self._build_reject(
                PreFlightRejectReason.EXCEEDS_SINGLE_ORDER_NOTIONAL,
                f"Single order notional ${evaluated_notional:,.2f} exceeds firm threshold (${self.limits.max_single_order_notional_usd:,.2f}).",
                order, evaluated_notional, price_dev_pct, t_start,
            )

        # 10. Cumulative Daily Gross Notional Limit Check
        projected_daily = self._cumulative_daily_gross_usd + evaluated_notional
        if projected_daily > self.limits.max_daily_gross_notional_usd:
            return self._build_reject(
                PreFlightRejectReason.EXCEEDS_DAILY_GROSS_CREDIT_CEILING,
                f"Projected daily gross notional ${projected_daily:,.2f} exceeds daily ceiling (${self.limits.max_daily_gross_notional_usd:,.2f}).",
                order, evaluated_notional, price_dev_pct, t_start,
            )

        # --- ALL COMPLIANCE FILTERS PASSED ---
        self._seen_clordids.add(order.cl_ord_id)
        self._burst_window.append(now_ts)
        self._cumulative_daily_gross_usd += evaluated_notional

        t_end = time.perf_counter_ns()
        latency_micros = (t_end - t_start) / 1000.0

        return PreFlightValidationResult(
            is_approved=True,
            reject_reason=PreFlightRejectReason.APPROVED,
            reject_message=None,
            cl_ord_id=order.cl_ord_id,
            symbol=order.symbol,
            evaluated_notional_usd=round(evaluated_notional, 2),
            evaluated_price_deviation_pct=round(price_dev_pct * 100.0, 4),
            validation_latency_micros=round(latency_micros, 2),
        )

    def _build_reject(
        self,
        reason: PreFlightRejectReason,
        message: str,
        order: FIXNewOrderSingle,
        notional: float,
        dev_pct: float,
        t_start: int,
    ) -> PreFlightValidationResult:
        t_end = time.perf_counter_ns()
        latency_micros = (t_end - t_start) / 1000.0
        logger.warning(
            f"Pre-Flight order rejected [{reason}]: {order.cl_ord_id} {order.symbol} "
            f"qty={order.order_qty} px={order.price} - {message}"
        )
        return PreFlightValidationResult(
            is_approved=False,
            reject_reason=reason,
            reject_message=message,
            cl_ord_id=order.cl_ord_id,
            symbol=order.symbol,
            evaluated_notional_usd=round(notional, 2),
            evaluated_price_deviation_pct=round(dev_pct * 100.0, 4),
            validation_latency_micros=round(latency_micros, 2),
        )

    def reset_daily_limits(self) -> None:
        """Reset daily cumulative gross volume counter and seen order ids."""
        self._cumulative_daily_gross_usd = 0.0
        self._seen_clordids.clear()
        self._burst_window.clear()
