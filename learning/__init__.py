"""Trading Intelligence Platform - Learning & Continuous Adaptation Package."""

from learning.auditor import ModelAuditor, ModelAuditReport, ModelHealthStatus
from learning.controller import (
    ContinuousLearningController,
    LearningMetricsReport,
    LearningState,
    Target99Status,
)
from learning.governance import ModelGovernor, PromotionStatus, PromotionVerdict
from learning.loop import ContinuousLearningCoordinator, LearningCycleResult
from learning.tracker import ActivePrediction, OutcomeTracker, ResolvedOutcome

__all__ = [
    "ActivePrediction",
    "ResolvedOutcome",
    "OutcomeTracker",
    "ModelHealthStatus",
    "ModelAuditReport",
    "ModelAuditor",
    "PromotionStatus",
    "PromotionVerdict",
    "ModelGovernor",
    "ContinuousLearningCoordinator",
    "ContinuousLearningController",
    "LearningCycleResult",
    "LearningState",
    "Target99Status",
    "LearningMetricsReport",
]

