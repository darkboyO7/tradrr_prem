"""Trading Intelligence Platform - Model Auditor & Drift Diagnostics.

Continuously monitors live and paper model accuracy, detects performance degradation,
measures probability calibration (Brier score), and issues retirement recommendations.
"""

from enum import Enum

import numpy as np
from pydantic import BaseModel, Field

from learning.tracker import ResolvedOutcome


class ModelHealthStatus(str, Enum):
    HEALTHY = "HEALTHY"
    DEGRADED = "DEGRADED"
    CRITICAL_RETIRE = "CRITICAL_RETIRE"


class ModelAuditReport(BaseModel):
    """Diagnostic audit findings for an active model."""

    model_id: str
    status: ModelHealthStatus
    sample_size: int
    win_rate: float
    brier_score: float
    mean_return: float
    mean_error: float
    buy_win_rate: float
    sell_win_rate: float
    diagnostic_warnings: list[str] = Field(default_factory=list)
    retirement_recommended: bool = False


class ModelAuditor:
    """Statistical auditor monitoring model integrity and post-deployment performance."""

    def __init__(
        self,
        min_samples: int = 15,
        min_acceptable_win_rate: float = 0.48,
        retire_win_rate_threshold: float = 0.42,
        max_acceptable_brier_score: float = 0.28,
    ) -> None:
        self.min_samples = min_samples
        self.min_acceptable_win_rate = min_acceptable_win_rate
        self.retire_win_rate_threshold = retire_win_rate_threshold
        self.max_acceptable_brier_score = max_acceptable_brier_score

    def audit_outcomes(self, model_id: str, outcomes: list[ResolvedOutcome]) -> ModelAuditReport:
        """Analyze historical resolved outcomes for a given model."""
        model_outcomes = [o for o in outcomes if o.model_id == model_id]
        n = len(model_outcomes)

        if n < self.min_samples:
            return ModelAuditReport(
                model_id=model_id,
                status=ModelHealthStatus.HEALTHY,
                sample_size=n,
                win_rate=0.5,
                brier_score=0.25,
                mean_return=0.0,
                mean_error=0.0,
                buy_win_rate=0.5,
                sell_win_rate=0.5,
                diagnostic_warnings=[f"Insufficient sample size ({n}/{self.min_samples}) for conclusive audit."],
            )

        wins = [o for o in model_outcomes if o.outcome == "WIN"]
        win_rate = len(wins) / n
        brier = float(np.mean([o.brier_score for o in model_outcomes]))
        mean_ret = float(np.mean([o.actual_return for o in model_outcomes]))
        mean_err = float(np.mean([o.error for o in model_outcomes]))

        buy_outcomes = [o for o in model_outcomes if o.action.value == "BUY"]
        sell_outcomes = [o for o in model_outcomes if o.action.value == "SELL"]

        buy_wr = (len([o for o in buy_outcomes if o.outcome == "WIN"]) / len(buy_outcomes)) if buy_outcomes else 0.5
        sell_wr = (len([o for o in sell_outcomes if o.outcome == "WIN"]) / len(sell_outcomes)) if sell_outcomes else 0.5

        warnings: list[str] = []
        retire = False

        if win_rate < self.retire_win_rate_threshold:
            status = ModelHealthStatus.CRITICAL_RETIRE
            retire = True
            warnings.append(
                f"Severe win rate degradation: {win_rate*100:.1f}% below retirement threshold {self.retire_win_rate_threshold*100:.1f}%"
            )
        elif win_rate < self.min_acceptable_win_rate:
            status = ModelHealthStatus.DEGRADED
            warnings.append(f"Performance decay: Win rate {win_rate*100:.1f}% is below acceptable minimum {self.min_acceptable_win_rate*100:.1f}%")
        else:
            status = ModelHealthStatus.HEALTHY

        if brier > self.max_acceptable_brier_score:
            warnings.append(f"Probability calibration degraded: Brier score ({brier:.3f}) exceeds threshold ({self.max_acceptable_brier_score:.3f})")
            if status == ModelHealthStatus.HEALTHY:
                status = ModelHealthStatus.DEGRADED

        if abs(buy_wr - sell_wr) > 0.25 and len(buy_outcomes) >= 5 and len(sell_outcomes) >= 5:
            warnings.append(f"Directional asymmetry: Buy WR ({buy_wr*100:.1f}%) diverged from Sell WR ({sell_wr*100:.1f}%)")

        return ModelAuditReport(
            model_id=model_id,
            status=status,
            sample_size=n,
            win_rate=round(win_rate, 4),
            brier_score=round(brier, 4),
            mean_return=round(mean_ret, 5),
            mean_error=round(mean_err, 5),
            buy_win_rate=round(buy_wr, 4),
            sell_win_rate=round(sell_wr, 4),
            diagnostic_warnings=warnings,
            retirement_recommended=retire,
        )
