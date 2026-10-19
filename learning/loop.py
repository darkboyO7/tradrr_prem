"""Trading Intelligence Platform - Autonomous Continuous Learning Loop.

Coordinates the automated model evaluation, chronological retraining,
untouched out-of-sample governance, and controlled model promotion.
Guarantees strict zero-lookahead bias and zero unvalidated self-modification.
"""

from datetime import UTC, datetime
from typing import Any

import pandas as pd
from pydantic import BaseModel, Field

from core.logging import get_logger
from features.pipeline import FeaturePipeline
from learning.auditor import ModelAuditor, ModelAuditReport, ModelHealthStatus
from learning.governance import ModelGovernor, PromotionStatus
from learning.tracker import ResolvedOutcome
from models.base import BasePredictorModel
from models.ml_classifier import LightGBMPredictor
from models.registry import ModelRegistry

logger = get_logger("learning_loop")


class LearningCycleResult(BaseModel):
    """Immutable audit outcome of an autonomous continuous learning cycle."""

    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    triggered: bool
    trigger_reason: str
    active_model_id: str | None = None
    audit_report: dict[str, Any] | None = None
    candidate_model_id: str | None = None
    promotion_verdict: dict[str, Any] | None = None
    promoted_to_production: bool = False


class ContinuousLearningCoordinator:
    """Orchestrates controlled self-learning and model governance."""

    def __init__(
        self,
        registry: ModelRegistry | None = None,
        governor: ModelGovernor | None = None,
        auditor: ModelAuditor | None = None,
        feature_pipeline: FeaturePipeline | None = None,
    ) -> None:
        self.registry = registry or ModelRegistry()
        self.governor = governor or ModelGovernor()
        self.auditor = auditor or ModelAuditor()
        self.feature_pipeline = feature_pipeline or FeaturePipeline()
        self.active_production_model: BasePredictorModel | None = None

    def set_production_model(self, model: BasePredictorModel) -> None:
        """Explicitly set current active production model."""
        self.active_production_model = model

    def check_retraining_trigger(
        self,
        audit_report: ModelAuditReport | None = None,
        force: bool = False,
    ) -> tuple[bool, str]:
        """Determine if a retraining cycle should be initiated."""
        if force:
            return True, "Manual/scheduled trigger forced retraining"

        if audit_report is None:
            return False, "No audit report provided and force is False"

        if audit_report.status == ModelHealthStatus.CRITICAL_RETIRE:
            return True, f"Critical model degradation detected (Win rate: {audit_report.win_rate:.1f}%)"

        if audit_report.status == ModelHealthStatus.DEGRADED:
            return True, "Model performance degraded below operational thresholds"

        if audit_report.retirement_recommended:
            return True, "Model auditor explicitly recommended retirement"

        return False, "Active model remains healthy; retraining not required"

    def prepare_chronological_splits(
        self,
        df: pd.DataFrame,
        train_ratio: float = 0.70,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Split historical dataset chronologically into Train and untouched Out-Of-Sample (OOS) splits.

        Strictly avoids random shuffling or lookahead leakage.
        """
        if df.empty:
            raise ValueError("Cannot split empty historical DataFrame")

        split_idx = int(len(df) * train_ratio)
        train_df = df.iloc[:split_idx].copy()
        oos_df = df.iloc[split_idx:].copy()

        logger.info(
            "Prepared chronological dataset splits",
            total_bars=len(df),
            train_bars=len(train_df),
            oos_bars=len(oos_df),
            train_ratio=train_ratio,
        )
        return train_df, oos_df

    def train_candidate_model(
        self,
        train_df: pd.DataFrame,
        model_id_prefix: str = "lgbm_candidate",
        version: str | None = None,
    ) -> LightGBMPredictor:
        """Engineer features and train a fresh LightGBM candidate model."""
        features_df = self.feature_pipeline.compute_all_features(train_df)

        # Generate target: forward 5-bar return direction (0=Sell, 1=Neutral, 2=Buy)
        horizon = 5
        fwd_returns = features_df["close"].pct_change(horizon).shift(-horizon)
        threshold = 0.003  # 0.3% price move threshold

        target = pd.Series(1, index=features_df.index)
        target[fwd_returns > threshold] = 2   # Buy
        target[fwd_returns < -threshold] = 0  # Sell

        # Drop the last horizon rows which have forward NaNs
        valid_idx = features_df.index[:-horizon]
        clean_X = features_df.loc[valid_idx]
        clean_y = target.loc[valid_idx]

        v = version or datetime.now(UTC).strftime("v%Y%m%d_%H%M%S")
        model_id = f"{model_id_prefix}_{v}"

        candidate = LightGBMPredictor(
            model_id=model_id,
            version=v,
            n_estimators=50,
            learning_rate=0.05,
            max_depth=4,
        )

        candidate.fit(clean_X, clean_y)
        logger.info("Trained candidate model", model_id=model_id, version=v, samples=len(clean_X))
        return candidate

    def execute_governance_cycle(
        self,
        historical_candles_df: pd.DataFrame,
        outcomes: list[ResolvedOutcome] | None = None,
        force_retrain: bool = False,
    ) -> LearningCycleResult:
        """Run the complete self-learning and governance workflow."""
        audit_report = None
        if self.active_production_model and outcomes:
            audit_report = self.auditor.audit_outcomes(
                model_id=self.active_production_model.metadata.model_id,
                outcomes=outcomes,
            )

        should_retrain, trigger_reason = self.check_retraining_trigger(audit_report, force=force_retrain)

        if not should_retrain:
            return LearningCycleResult(
                triggered=False,
                trigger_reason=trigger_reason,
                active_model_id=self.active_production_model.metadata.model_id if self.active_production_model else None,
                audit_report=audit_report.model_dump() if audit_report else None,
                promoted_to_production=False,
            )

        logger.info("Initiating Controlled Model Retraining Cycle", reason=trigger_reason)

        # 1. Chronological splitting
        train_df, oos_df = self.prepare_chronological_splits(historical_candles_df, train_ratio=0.70)

        # 2. Train candidate model
        candidate = self.train_candidate_model(train_df)

        # 3. Save candidate to registry
        self.registry.save_model(candidate)

        # 4. Strict Out-of-Sample Governance Evaluation
        verdict = self.governor.evaluate_candidate(
            candidate_model=candidate,
            oos_df=oos_df,
            production_model=self.active_production_model,
        )

        promoted = verdict.status == PromotionStatus.PROMOTED
        if promoted:
            logger.info("Candidate model promoted to Production", model_id=candidate.metadata.model_id)
            self.active_production_model = candidate
        else:
            logger.warning(
                "Candidate model rejected by governance. Production model preserved.",
                model_id=candidate.metadata.model_id,
                reason=verdict.reason,
            )

        return LearningCycleResult(
            triggered=True,
            trigger_reason=trigger_reason,
            active_model_id=self.active_production_model.metadata.model_id if self.active_production_model else None,
            audit_report=audit_report.model_dump() if audit_report else None,
            candidate_model_id=candidate.metadata.model_id,
            promotion_verdict=verdict.model_dump(),
            promoted_to_production=promoted,
        )
