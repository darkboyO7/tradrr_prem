"""Trading Intelligence Platform - Pre-Trade Risk Engine & Portfolio Gatekeeper.

Enforces firm-wide exposure caps, drawdown circuit breakers, position limits,
and mathematical sizing before any order can be routed to execution.
"""

from datetime import UTC, datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

from core.database import get_db_session
from core.logging import get_logger
from core.symbol_resolver import SymbolResolver
from data.storage.models import RiskEventModel
from models.base import SignalAction
from risk.limits import RiskLimits
from risk.sizing import PositionSizer
from signals.fusion import FinalSignalDecision

logger = get_logger(__name__)


class RiskDecisionStatus(str, Enum):
    APPROVED = "APPROVED"
    RESIZED = "RESIZED"
    REJECTED = "REJECTED"


class RiskDecision(BaseModel):
    """Immutable audit outcome of the pre-trade risk evaluation."""

    status: RiskDecisionStatus
    symbol: str
    action: SignalAction
    requested_quantity: float
    approved_quantity: float
    entry_price: float
    stop_loss: float
    take_profit: float
    risk_reward_ratio: float
    rejection_reason: str | None = None
    details: dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))


class RiskEngine:
    """Stateful portfolio risk coordinator and pre-trade gatekeeper."""

    def __init__(
        self,
        initial_capital: float = 100000.0,
        limits: RiskLimits | None = None,
        session_factory: Any | None = None,
    ) -> None:
        self.limits = limits or RiskLimits()
        self._session_factory = session_factory or get_db_session

        self.equity: float = initial_capital
        self.peak_equity: float = initial_capital
        self.session_start_equity: float = initial_capital
        self.open_positions: dict[str, float] = {}  # symbol -> notional exposure

    def update_portfolio_state(self, current_equity: float, open_positions: dict[str, float]) -> None:
        """Update live equity and position tracking."""
        self.equity = current_equity
        if current_equity > self.peak_equity:
            self.peak_equity = current_equity
        self.open_positions = open_positions

    def reset_daily_session(self) -> None:
        """Reset starting equity at the start of a new trading day/session."""
        self.session_start_equity = self.equity

    def evaluate_trade(
        self,
        signal: FinalSignalDecision,
        requested_quantity: float | None = None,
    ) -> RiskDecision:
        """Evaluate trade setup against all risk constraints and determine safe quantity."""
        symbol = signal.symbol
        price = signal.entry_price
        action = signal.action

        # 1. Reject NO_TRADE signals immediately
        if action == SignalAction.NO_TRADE:
            return RiskDecision(
                status=RiskDecisionStatus.REJECTED,
                symbol=symbol,
                action=action,
                requested_quantity=0.0,
                approved_quantity=0.0,
                entry_price=price,
                stop_loss=signal.stop_loss,
                take_profit=signal.take_profit,
                risk_reward_ratio=signal.risk_reward_ratio,
                rejection_reason="Signal action is NO_TRADE",
            )

        # 2. Peak-to-Trough Drawdown Circuit Breaker
        current_dd = (self.peak_equity - self.equity) / self.peak_equity if self.peak_equity > 0 else 0.0
        if current_dd >= self.limits.max_drawdown_limit_pct:
            reason = f"Max drawdown circuit breaker triggered! Drawdown {current_dd*100:.1f}% >= limit {self.limits.max_drawdown_limit_pct*100:.1f}%"
            logger.critical("Trade rejected by Risk Engine: Max Drawdown Breaker", reason=reason)
            return self._create_rejection(signal, requested_quantity or 0.0, reason)

        # 3. Daily Drawdown Circuit Breaker
        daily_dd = (self.session_start_equity - self.equity) / self.session_start_equity if self.session_start_equity > 0 else 0.0
        if daily_dd >= self.limits.daily_drawdown_limit_pct:
            reason = f"Daily loss limit breached! Daily drop {daily_dd*100:.1f}% >= limit {self.limits.daily_drawdown_limit_pct*100:.1f}%"
            logger.warning("Trade rejected by Risk Engine: Daily Loss Limit", reason=reason)
            return self._create_rejection(signal, requested_quantity or 0.0, reason)

        # 4. Maximum Open Positions Limit
        if symbol not in self.open_positions and len(self.open_positions) >= self.limits.max_open_positions:
            reason = f"Max concurrent positions reached ({len(self.open_positions)}/{self.limits.max_open_positions})"
            return self._create_rejection(signal, requested_quantity or 0.0, reason)

        # 5. Risk / Reward Minimum Verification
        if signal.risk_reward_ratio < self.limits.min_risk_reward_ratio:
            reason = f"Risk/Reward {signal.risk_reward_ratio:.2f} below minimum requirement {self.limits.min_risk_reward_ratio:.2f}"
            return self._create_rejection(signal, requested_quantity or 0.0, reason)

        # 6. Mathematical Position Sizing
        safe_quantity = PositionSizer.calculate_fixed_fractional_size(
            equity=self.equity,
            entry_price=price,
            stop_loss=signal.stop_loss,
            risk_per_trade_pct=self.limits.max_risk_per_trade_pct,
            max_position_pct=self.limits.max_single_position_pct,
        )

        if safe_quantity <= 0.0:
            reason = "Calculated position size is zero or invalid price/stop-loss distance"
            return self._create_rejection(signal, requested_quantity or 0.0, reason)

        # 7. Check if requested quantity exceeds safe allowance
        inst = SymbolResolver.resolve(symbol)
        req_q = requested_quantity if (requested_quantity and requested_quantity > 0) else safe_quantity
        status = RiskDecisionStatus.APPROVED
        final_quantity = req_q

        if requested_quantity is not None and requested_quantity > safe_quantity:
            status = RiskDecisionStatus.RESIZED
            final_quantity = safe_quantity
            logger.info("Resized trade quantity to fit risk limits", original=req_q, approved=safe_quantity)
        elif requested_quantity is None and inst is not None:
            final_quantity = inst.round_quantity(safe_quantity)
            req_q = final_quantity

        # Apply instrument tick & lot size constraints if instrument is registered
        if inst:
            final_quantity = inst.round_quantity(final_quantity)
            if final_quantity <= 0:
                reason = f"Quantity {final_quantity} below minimum lot size ({inst.lot_size})"
                return self._create_rejection(signal, req_q, reason)

        # 8. Check Total Portfolio Leverage
        current_exposure = sum(self.open_positions.values())
        new_trade_notional = final_quantity * price
        total_projected_exposure = current_exposure + new_trade_notional
        projected_leverage = total_projected_exposure / self.equity if self.equity > 0 else 99.0

        if projected_leverage > self.limits.max_portfolio_leverage:
            reason = f"Portfolio leverage {projected_leverage:.2f}x exceeds max limit {self.limits.max_portfolio_leverage:.2f}x"
            return self._create_rejection(signal, req_q, reason)

        decision = RiskDecision(
            status=status,
            symbol=symbol,
            action=action,
            requested_quantity=req_q,
            approved_quantity=final_quantity,
            entry_price=price,
            stop_loss=signal.stop_loss,
            take_profit=signal.take_profit,
            risk_reward_ratio=signal.risk_reward_ratio,
            details={
                "equity": self.equity,
                "drawdown_pct": round(current_dd, 4),
                "daily_drawdown_pct": round(daily_dd, 4),
                "leverage": round(projected_leverage, 2),
            },
        )
        return decision

    def _create_rejection(self, signal: FinalSignalDecision, requested_quantity: float, reason: str) -> RiskDecision:
        return RiskDecision(
            status=RiskDecisionStatus.REJECTED,
            symbol=signal.symbol,
            action=signal.action,
            requested_quantity=requested_quantity,
            approved_quantity=0.0,
            entry_price=signal.entry_price,
            stop_loss=signal.stop_loss,
            take_profit=signal.take_profit,
            risk_reward_ratio=signal.risk_reward_ratio,
            rejection_reason=reason,
        )

    async def log_risk_event(self, decision: RiskDecision) -> None:
        """Persist risk decision in SQL audit ledger."""
        try:
            async with self._session_factory() as session:
                event = RiskEventModel(
                    event_id=f"risk_{int(datetime.now(UTC).timestamp()*1000)}",
                    symbol=decision.symbol,
                    risk_check=decision.status.value,
                    decision=decision.status.value,
                    requested_exposure=decision.requested_quantity * decision.entry_price,
                    limit_value=self.equity * self.limits.max_single_position_pct,
                    details={
                        "rejection_reason": decision.rejection_reason,
                        "approved_quantity": decision.approved_quantity,
                        "stop_loss": decision.stop_loss,
                        "take_profit": decision.take_profit,
                        **decision.details,
                    },
                )
                session.add(event)
                await session.commit()
        except Exception as exc:
            logger.error("Failed to log risk event to database", error=str(exc))
