"""Trading Intelligence Platform - Realized Prediction Outcome Tracker.

Tracks in-flight model predictions until their forward horizon expires,
measures actual return vs prediction, calculates MFE / MAE, Brier score, and logs audit outcomes.
"""

from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, Field

from core.database import get_db_session
from core.logging import get_logger
from data.storage.models import PredictionOutcomeModel
from models.base import SignalAction

logger = get_logger(__name__)


class ActivePrediction(BaseModel):
    """In-flight prediction awaiting market realization."""

    prediction_id: str
    model_id: str
    symbol: str
    action: SignalAction
    probability: float
    expected_return: float
    entry_price: float
    horizon_bars: int = 5
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    bars_elapsed: int = 0
    highest_price: float = 0.0
    lowest_price: float = 0.0


class ResolvedOutcome(BaseModel):
    """Finalized prediction vs reality evaluation record."""

    prediction_id: str
    model_id: str
    symbol: str
    action: SignalAction
    entry_price: float
    exit_price: float
    actual_return: float
    predicted_return: float
    outcome: str  # "WIN", "LOSS", "NEUTRAL"
    mfe: float  # Maximum Favorable Excursion
    mae: float  # Maximum Adverse Excursion
    brier_score: float
    error: float
    horizon_bars: int
    evaluated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class OutcomeTracker:
    """Manages active predictions and calculates realization metrics."""

    def __init__(self, session_factory: Any | None = None) -> None:
        self._session_factory = session_factory or get_db_session
        self.active_predictions: dict[str, ActivePrediction] = {}
        self.resolved_outcomes: list[ResolvedOutcome] = []

    def record_prediction(
        self,
        prediction_id: str,
        model_id: str,
        symbol: str,
        action: SignalAction,
        probability: float,
        expected_return: float,
        entry_price: float,
        horizon_bars: int = 5,
    ) -> ActivePrediction:
        """Register a new prediction to be monitored until its horizon elapses."""
        active = ActivePrediction(
            prediction_id=prediction_id,
            model_id=model_id,
            symbol=symbol,
            action=action,
            probability=probability,
            expected_return=expected_return,
            entry_price=entry_price,
            horizon_bars=horizon_bars,
            highest_price=entry_price,
            lowest_price=entry_price,
        )
        self.active_predictions[prediction_id] = active
        return active

    def on_bar(
        self,
        symbol: str,
        close: float,
        high: float,
        low: float,
    ) -> list[ResolvedOutcome]:
        """Update market trajectory for active predictions and resolve those reaching horizon."""
        resolved: list[ResolvedOutcome] = []
        to_remove: list[str] = []

        for pid, pred in self.active_predictions.items():
            if pred.symbol != symbol:
                continue

            pred.bars_elapsed += 1
            if high > pred.highest_price:
                pred.highest_price = high
            if low < pred.lowest_price:
                pred.lowest_price = low

            # Check if prediction reached its forward horizon
            if pred.bars_elapsed >= pred.horizon_bars:
                actual_ret = (close - pred.entry_price) / pred.entry_price
                if pred.action == SignalAction.SELL:
                    actual_ret = -actual_ret

                # Determine WIN / LOSS / NEUTRAL
                if actual_ret > 0.001:
                    outcome = "WIN"
                    y_true = 1.0
                elif actual_ret < -0.001:
                    outcome = "LOSS"
                    y_true = 0.0
                else:
                    outcome = "NEUTRAL"
                    y_true = 0.5

                # MFE & MAE
                if pred.action == SignalAction.BUY:
                    mfe = (pred.highest_price - pred.entry_price) / pred.entry_price
                    mae = (pred.lowest_price - pred.entry_price) / pred.entry_price
                else:
                    mfe = (pred.entry_price - pred.lowest_price) / pred.entry_price
                    mae = (pred.entry_price - pred.highest_price) / pred.entry_price

                # Brier score: (probability - y_true)^2
                brier = (pred.probability - y_true) ** 2
                err = abs(actual_ret - pred.expected_return)

                res = ResolvedOutcome(
                    prediction_id=pred.prediction_id,
                    model_id=pred.model_id,
                    symbol=pred.symbol,
                    action=pred.action,
                    entry_price=pred.entry_price,
                    exit_price=close,
                    actual_return=round(float(actual_ret), 5),
                    predicted_return=round(float(pred.expected_return), 5),
                    outcome=outcome,
                    mfe=round(float(mfe), 5),
                    mae=round(float(mae), 5),
                    brier_score=round(float(brier), 5),
                    error=round(float(err), 5),
                    horizon_bars=pred.horizon_bars,
                )
                resolved.append(res)
                self.resolved_outcomes.append(res)
                to_remove.append(pid)

        for pid in to_remove:
            self.active_predictions.pop(pid, None)

        return resolved

    async def persist_outcome(self, outcome: ResolvedOutcome) -> int | None:
        """Save resolved prediction outcome to SQL database."""
        try:
            async with self._session_factory() as session:
                record = PredictionOutcomeModel(
                    prediction_id=outcome.prediction_id,
                    evaluated_at=outcome.evaluated_at,
                    actual_return=outcome.actual_return,
                    mfe=outcome.mfe,
                    mae=outcome.mae,
                    outcome=outcome.outcome,
                    error=outcome.error,
                    time_to_outcome_secs=outcome.horizon_bars * 60,
                    evaluation_horizon=f"{outcome.horizon_bars}m",
                )
                session.add(record)
                await session.commit()
                return record.id
        except Exception as exc:
            logger.error("Failed to persist prediction outcome to database", error=str(exc))
            return None
