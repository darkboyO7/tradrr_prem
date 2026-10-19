"""Trading Intelligence Platform - Model Governance & Promotion Engine.

Enforces strict, objective promotion gates for controlled self-learning.
Models are NEVER deployed directly into production without untouched out-of-sample validation.
"""

from datetime import UTC, datetime
from enum import Enum
from typing import Any

import pandas as pd
from pydantic import BaseModel, Field

from backtesting.engine import BacktestEngine
from core.database import get_db_session
from core.logging import get_logger
from data.storage.models import ModelPromotionModel
from models.base import BasePredictorModel

logger = get_logger(__name__)


class PromotionStatus(str, Enum):
    PROMOTED = "PROMOTED"
    REJECTED = "REJECTED"


class PromotionVerdict(BaseModel):
    """Auditable promotion decision ledger entry."""

    candidate_model_id: str
    production_model_id: str | None = None
    status: PromotionStatus
    approved_by: str = "AutomatedGovernanceEngine"
    reason: str
    candidate_metrics: dict[str, float]
    production_metrics: dict[str, float] = Field(default_factory=dict)
    gate_checks: dict[str, bool] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))


class ModelGovernor:
    """Enforces rigorous, objective qualification criteria before promoting any model to production."""

    def __init__(
        self,
        min_oos_sharpe: float = 1.0,
        min_oos_win_rate: float = 50.0,
        max_oos_drawdown: float = 15.0,
        sharpe_improvement_margin: float = 0.10,
        session_factory: Any | None = None,
    ) -> None:
        self.min_oos_sharpe = min_oos_sharpe
        self.min_oos_win_rate = min_oos_win_rate
        self.max_oos_drawdown = max_oos_drawdown
        self.sharpe_improvement_margin = sharpe_improvement_margin
        self._session_factory = session_factory or get_db_session

    def evaluate_candidate(
        self,
        candidate_model: BasePredictorModel,
        oos_df: pd.DataFrame,
        production_model: BasePredictorModel | None = None,
    ) -> PromotionVerdict:
        """Run untouched out-of-sample evaluation and determine if candidate meets promotion criteria."""
        engine = BacktestEngine(initial_capital=100000.0)

        # Run candidate on untouched out-of-sample dataset
        candidate_backtest = engine.run(oos_df, ml_model=candidate_model)
        cand_m = candidate_backtest["metrics"]

        candidate_metrics = {
            "sharpe": cand_m.sharpe_ratio,
            "win_rate": cand_m.win_rate,
            "max_drawdown": cand_m.max_drawdown_pct,
            "profit_factor": cand_m.profit_factor,
            "expectancy": cand_m.expectancy_dollars,
            "total_return_pct": cand_m.total_return_pct,
        }

        # Gate 1: Candidate absolute quality checks
        gate_sharpe = cand_m.sharpe_ratio >= self.min_oos_sharpe
        gate_winrate = cand_m.win_rate >= self.min_oos_win_rate
        gate_drawdown = cand_m.max_drawdown_pct <= self.max_oos_drawdown

        gate_checks = {
            "min_sharpe": gate_sharpe,
            "min_win_rate": gate_winrate,
            "max_drawdown": gate_drawdown,
        }

        prod_metrics: dict[str, float] = {}

        # Gate 2: Relative checks against active production model (if exists)
        if production_model is not None:
            prod_backtest = engine.run(oos_df, ml_model=production_model)
            prod_m = prod_backtest["metrics"]
            prod_metrics = {
                "sharpe": prod_m.sharpe_ratio,
                "win_rate": prod_m.win_rate,
                "max_drawdown": prod_m.max_drawdown_pct,
                "profit_factor": prod_m.profit_factor,
            }

            # Must beat production model Sharpe by at least the margin
            gate_beat_prod = cand_m.sharpe_ratio >= (prod_m.sharpe_ratio + self.sharpe_improvement_margin)
            gate_checks["beat_production_sharpe"] = gate_beat_prod
        else:
            gate_checks["beat_production_sharpe"] = True

        all_passed = all(gate_checks.values())

        if all_passed:
            status = PromotionStatus.PROMOTED
            reason = (
                f"Candidate {candidate_model.metadata.model_id} satisfied all promotion gates "
                f"(Sharpe: {cand_m.sharpe_ratio:.2f} >= {self.min_oos_sharpe:.2f}, WR: {cand_m.win_rate:.1f}%, DD: {cand_m.max_drawdown_pct:.1f}%)."
            )
            candidate_model.metadata.is_production_ready = True
            logger.info("MODEL PROMOTION APPROVED", model_id=candidate_model.metadata.model_id, reason=reason)
        else:
            status = PromotionStatus.REJECTED
            failed_gates = [k for k, v in gate_checks.items() if not v]
            reason = f"Candidate failed promotion gates: {failed_gates}"
            logger.warning("MODEL PROMOTION REJECTED", model_id=candidate_model.metadata.model_id, reason=reason)

        return PromotionVerdict(
            candidate_model_id=candidate_model.metadata.model_id,
            production_model_id=production_model.metadata.model_id if production_model else None,
            status=status,
            reason=reason,
            candidate_metrics=candidate_metrics,
            production_metrics=prod_metrics,
            gate_checks=gate_checks,
        )

    async def record_verdict_in_db(self, verdict: PromotionVerdict) -> int | None:
        """Record model promotion verdict in SQL audit ledger."""
        try:
            async with self._session_factory() as session:
                record = ModelPromotionModel(
                    model_id=verdict.candidate_model_id,
                    from_status="CANDIDATE",
                    to_status="PRODUCTION" if verdict.status == PromotionStatus.PROMOTED else "REJECTED",
                    approved_by=verdict.approved_by,
                    reason=verdict.reason,
                    metrics_snapshot={
                        "candidate": verdict.candidate_metrics,
                        "production": verdict.production_metrics,
                        "gate_checks": verdict.gate_checks,
                    },
                )
                session.add(record)
                await session.commit()
                return record.id
        except Exception as exc:
            logger.error("Failed to record model promotion in database", error=str(exc))
            return None
