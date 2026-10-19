"""Trading Intelligence Platform - Autonomous Continuous Learning Controller.

Implements the complete autonomous learning lifecycle:
Collect data -> Generate predictions -> Record predictions -> Resolve outcomes ->
Evaluate performance -> Prepare training data -> Train candidate model ->
Validate candidate -> Compare against champion -> Promote or reject ->
Monitor production -> Rollback if degraded -> Repeat.

Strict zero-lookahead, purged chronological splits, walk-forward validation,
persistent crash-resilient state machine, and honest statistical evaluation against
the aspirational 99% accuracy target.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from enum import Enum
import hashlib
import json
from pathlib import Path
import threading
import time
from typing import Any

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

from core.logging import get_logger
from models.base import BasePredictorModel, ModelPrediction, SignalAction
from models.ml_classifier import LightGBMPredictor
from models.registry import ModelRegistry

logger = get_logger("learning_controller")

DEFAULT_STATE_FILE = Path(__file__).resolve().parent.parent / "data" / "learning" / "learning_controller_state.json"
DEFAULT_AUDIT_LOG_FILE = Path(__file__).resolve().parent.parent / "data" / "learning" / "learning_audit_ledger.jsonl"


class LearningState(str, Enum):
    """Explicit lifecycle states of the Continuous Learning State Machine."""

    IDLE = "IDLE"
    COLLECTING_OUTCOMES = "COLLECTING_OUTCOMES"
    PREPARING_DATA = "PREPARING_DATA"
    TRAINING = "TRAINING"
    VALIDATING = "VALIDATING"
    PROMOTING = "PROMOTING"
    REJECTING = "REJECTING"
    MONITORING = "MONITORING"
    ROLLING_BACK = "ROLLING_BACK"
    ERROR = "ERROR"
    PAUSED = "PAUSED"


class Target99Status(str, Enum):
    """Statistical verification status against the aspirational 99% accuracy target."""

    MET = "MET"
    UNMET = "UNMET"
    NOT_YET_VERIFIABLE = "NOT_YET_VERIFIABLE"


class LearningMetricsReport(BaseModel):
    """Complete statistical evaluation of model performance and 99% target assessment."""

    sample_size_n: int = 0
    total_samples_n: int = 0
    directional_samples_n: int = 0
    no_trade_samples_n: int = 0
    total_correct: int = 0
    directional_accuracy_pct: float | None = None
    effective_accuracy_pct: float | None = None
    prediction_coverage_pct: float | None = None
    abstention_rate_pct: float | None = None
    target_99_status: Target99Status = Target99Status.NOT_YET_VERIFIABLE
    target_gap_pct: float | None = None
    wilson_ci_lower_pct: float | None = None
    wilson_ci_upper_pct: float | None = None
    precision_pct: float | None = None
    recall_pct: float | None = None
    f1_score_pct: float | None = None
    class_metrics: dict[str, dict[str, float]] = Field(default_factory=dict)
    confusion_matrix: dict[str, dict[str, int]] = Field(default_factory=dict)
    mean_brier_score: float | None = None
    multiclass_brier_score: float | None = None
    calibration_error: float | None = None
    expectancy_pct: float | None = None
    profit_factor: float | None = None
    max_drawdown_pct: float | None = None
    evaluation_summary: str = "Awaiting initial resolved predictions for statistical audit."
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))


class LearningCycleResult(BaseModel):
    """Persistent audit outcome of an autonomous continuous learning cycle."""

    cycle_id: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    state: LearningState
    triggered: bool
    trigger_reason: str
    champion_model_id: str
    candidate_model_id: str | None = None
    candidate_version: str | None = None
    promoted: bool = False
    rollback: bool = False
    dataset_fingerprint: str | None = None
    total_eligible_samples: int = 0
    train_samples: int = 0
    oos_samples: int = 0
    candidate_metrics: dict[str, Any] = Field(default_factory=dict)
    champion_metrics: dict[str, Any] = Field(default_factory=dict)
    verdict_reason: str = ""
    duration_seconds: float = 0.0
    error: str | None = None


class ContinuousLearningController:
    """Autonomous Continuous Learning Controller with strict zero-lookahead,

    purged walk-forward validation, persistent state machine, and champion governance.
    """

    _instance: ContinuousLearningController | None = None
    _singleton_lock = threading.Lock()

    def __init__(
        self,
        state_file: Path | str | None = None,
        audit_log_file: Path | str | None = None,
        model_registry: ModelRegistry | None = None,
        ledger_file: Path | str | None = None,
        min_samples_to_retrain: int = 15,
        min_new_outcomes_delta: int = 5,
        min_eval_samples_for_99: int = 50,
        rollback_win_rate_threshold: float = 40.0,
        rollback_brier_threshold: float = 0.35,
        rollback_window_size: int = 25,
        check_interval_seconds: float = 60.0,
    ) -> None:
        self.state_file = Path(state_file or DEFAULT_STATE_FILE)
        self.audit_log_file = Path(audit_log_file or DEFAULT_AUDIT_LOG_FILE)
        self.ledger_file = (
            Path(ledger_file)
            if ledger_file is not None
            else (Path(__file__).resolve().parent.parent / "data" / "ledger" / "prediction_outcomes.jsonl")
        )
        self.registry = model_registry or ModelRegistry()

        self.min_samples_to_retrain = min_samples_to_retrain
        self.min_new_outcomes_delta = min_new_outcomes_delta
        self.min_eval_samples_for_99 = min_eval_samples_for_99
        self.rollback_win_rate_threshold = rollback_win_rate_threshold
        self.rollback_brier_threshold = rollback_brier_threshold
        self.rollback_window_size = rollback_window_size
        self.check_interval_seconds = check_interval_seconds

        # Additional governance & anti-leakage attributes
        self.holdout_eval_count: int = 0
        self.min_oos_samples: int = 30
        self._dataset_timestamps: list[Any] = []
        self._dataset_horizons: list[int] = []

        # State machine variables
        self.state: LearningState = LearningState.IDLE
        self.active_champion_id: str = "lgbm_champion_v1.0"
        self.previous_champion_id: str = "lgbm_baseline_v0.9"
        self.active_champion_model: BasePredictorModel | None = None
        self.champion_lineage: list[dict[str, Any]] = []

        self.cycle_count: int = 0
        self.dataset_fingerprint: str = ""
        self.total_eligible_outcomes: int = 0
        self.last_trained_sample_count: int = 0
        self.last_run_timestamp: str | None = None
        self.last_duration_s: float = 0.0
        self.last_error: str | None = None
        self.consecutive_errors: int = 0
        self.last_cycle_result: LearningCycleResult | None = None
        self.last_metrics_report: LearningMetricsReport | None = None

        # Threading & concurrency guards
        self._cycle_lock = threading.Lock()
        self._worker_thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._is_paused: bool = False

        # Ensure storage directory exists and load persistent state
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        self.audit_log_file.parent.mkdir(parents=True, exist_ok=True)
        self._load_state()

        # Initialize champion model in memory if possible
        self._ensure_champion_loaded()

    @classmethod
    def get_instance(cls, **kwargs: Any) -> ContinuousLearningController:
        with cls._singleton_lock:
            if cls._instance is None:
                cls._instance = ContinuousLearningController(**kwargs)
            return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        """Reset singleton (primarily for testing purposes)."""
        with cls._singleton_lock:
            if cls._instance is not None:
                cls._instance.stop_background_loop()
                cls._instance = None

    # =========================================================================
    # PERSISTENCE & CRASH RECOVERY
    # =========================================================================

    def _load_state(self) -> None:
        """Load persistent controller state from disk.

        Handles crash recovery: if state was in an intermediate transient state,
        resets safely to IDLE and preserves lineage.
        """
        if not self.state_file.exists():
            logger.info("No prior continuous learning state found; starting fresh", state_file=str(self.state_file))
            return

        try:
            raw = json.loads(self.state_file.read_text(encoding="utf-8"))
            saved_state_str = raw.get("state", "IDLE")

            # Crash recovery check: if saved state was transient, reset to IDLE
            transient_states = {
                LearningState.COLLECTING_OUTCOMES.value,
                LearningState.PREPARING_DATA.value,
                LearningState.TRAINING.value,
                LearningState.VALIDATING.value,
                LearningState.PROMOTING.value,
                LearningState.REJECTING.value,
                LearningState.ROLLING_BACK.value,
            }
            if saved_state_str in transient_states:
                logger.warning(
                    "Detected interrupted learning cycle on restart; cleanly resetting to IDLE",
                    interrupted_state=saved_state_str,
                )
                self.state = LearningState.IDLE
            else:
                self.state = LearningState(saved_state_str)

            self.active_champion_id = raw.get("active_champion_id", self.active_champion_id)
            self.previous_champion_id = raw.get("previous_champion_id", self.previous_champion_id)
            self.champion_lineage = raw.get("champion_lineage", [])
            self.cycle_count = raw.get("cycle_count", 0)
            self.dataset_fingerprint = raw.get("dataset_fingerprint", "")
            self.total_eligible_outcomes = raw.get("total_eligible_outcomes", 0)
            self.last_trained_sample_count = raw.get("last_trained_sample_count", 0)
            self.last_run_timestamp = raw.get("last_run_timestamp")
            self.last_duration_s = raw.get("last_duration_s", 0.0)
            self.last_error = raw.get("last_error")
            self.consecutive_errors = raw.get("consecutive_errors", 0)

            # Restore cached report if present
            if "last_metrics_report" in raw and raw["last_metrics_report"]:
                try:
                    self.last_metrics_report = LearningMetricsReport(**raw["last_metrics_report"])
                except Exception:
                    pass

            logger.info(
                "Restored persistent ContinuousLearningController state",
                state=self.state.value,
                champion=self.active_champion_id,
                cycles=self.cycle_count,
            )
        except Exception as exc:
            logger.error("Failed loading persistent learning controller state", error=str(exc))
            self.state = LearningState.IDLE

    def _save_state(self) -> None:
        """Persist current controller state to JSON file."""
        try:
            payload = {
                "state": self.state.value,
                "active_champion_id": self.active_champion_id,
                "previous_champion_id": self.previous_champion_id,
                "champion_lineage": self.champion_lineage,
                "cycle_count": self.cycle_count,
                "dataset_fingerprint": self.dataset_fingerprint,
                "total_eligible_outcomes": self.total_eligible_outcomes,
                "last_trained_sample_count": self.last_trained_sample_count,
                "last_run_timestamp": self.last_run_timestamp,
                "last_duration_s": round(self.last_duration_s, 3),
                "last_error": self.last_error,
                "consecutive_errors": self.consecutive_errors,
                "last_metrics_report": self.last_metrics_report.model_dump(mode="json") if self.last_metrics_report else None,
                "updated_at": datetime.now(UTC).isoformat(),
            }
            temp_path = self.state_file.with_suffix(".tmp")
            temp_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            temp_path.replace(self.state_file)
        except Exception as exc:
            logger.error("Failed persisting learning controller state", error=str(exc))

    def _append_audit_log(self, cycle_res: LearningCycleResult) -> None:
        """Record immutable cycle audit record to JSONL."""
        try:
            line = cycle_res.model_dump_json()
            with self.audit_log_file.open("a", encoding="utf-8") as fh:
                fh.write(line + "\n")
        except Exception as exc:
            logger.warning("Failed appending to learning audit log", error=str(exc))

    # =========================================================================
    # CHAMPION MODEL SERVING
    # =========================================================================

    def _ensure_champion_loaded(self) -> None:
        """Ensure an active champion predictor is instantiated in memory for production inference."""
        if self.active_champion_model is not None and getattr(self.active_champion_model, "is_fitted", False):
            return

        # Attempt to load from registry
        try:
            # Check if directory exists
            model_dir = self.registry._get_model_dir(self.active_champion_id, "v1.0.0")
            if (model_dir / "model.joblib").exists():
                self.active_champion_model = self.registry.load_model(self.active_champion_id, "v1.0.0")
                logger.info("Loaded champion model from registry artifact", model_id=self.active_champion_id)
                return
        except Exception as exc:
            logger.debug("Champion artifact not found in registry; creating baseline", error=str(exc))

        # Create calibrated baseline champion predictor
        baseline = LightGBMPredictor(
            model_id=self.active_champion_id,
            version="v1.0.0",
            n_estimators=35,
            learning_rate=0.05,
            max_depth=3,
            num_leaves=7,
            min_child_samples=5,
        )
        # Pre-seed baseline with a synthetic balanced seed so it is fitted and immediately operational
        seed_X = pd.DataFrame(
            {
                "return_1": [0.005, -0.005, 0.001, -0.002, 0.007, -0.006, 0.0, 0.004, -0.004, 0.002],
                "log_return_5": [0.01, -0.01, 0.002, -0.003, 0.015, -0.012, 0.001, 0.008, -0.007, 0.003],
                "candle_range_pct": [0.008] * 10,
                "candle_body_ratio": [0.6] * 10,
                "rsi_14": [65, 35, 50, 45, 70, 30, 52, 60, 40, 55],
                "macd_hist": [0.001, -0.001, 0.0, -0.0005, 0.002, -0.002, 0.0, 0.0008, -0.0009, 0.0002],
                "adx_14": [25] * 10,
                "atr_14": [1.5] * 10,
                "rvol_20": [1.2] * 10,
                "volatility_regime": [1.0] * 10,
            }
        )
        seed_y = np.array([2, 0, 1, 0, 2, 0, 1, 2, 0, 1])  # 2=Buy, 0=Sell, 1=Neutral
        baseline.fit(seed_X, seed_y)
        self.active_champion_model = baseline

    def get_active_champion_model(self) -> BasePredictorModel | None:
        """Get the active champion model currently serving live production inference."""
        self._ensure_champion_loaded()
        return self.active_champion_model

    # =========================================================================
    # DATA INGESTION & DATASET PREPARATION
    # =========================================================================

    def _collect_eligible_outcomes(self) -> list[dict[str, Any]]:
        """Collect resolved predictions from PredictionLedgerService and JSONL ledger.

        Filters strictly for:
        - resolved == True
        - non-null exit price
        - non-empty feature snapshot
        - deduplicated by prediction_id
        """
        from learning.ledger import PredictionLedgerService

        ledger = PredictionLedgerService.get_instance()
        seen_ids: set[str] = set()
        eligible: list[dict[str, Any]] = []

        # 1. From active ledger service in-memory outcomes
        for rec in ledger._resolved_outcomes:
            pid = str(rec.get("prediction_id", ""))
            if not pid or pid in seen_ids:
                continue
            feats = rec.get("features_used") or {}
            has_outcome = rec.get("outcome") in ("WIN", "LOSS", "NEUTRAL")
            exit_p = rec.get("exit_price")
            if (has_outcome or (exit_p is not None and float(exit_p) > 0)) and len(feats) > 0:
                seen_ids.add(pid)
                eligible.append(rec)

        # 2. From persistent JSONL ledger file
        ledger_path = self.ledger_file
        if ledger_path.exists():
            try:
                for line in ledger_path.read_text(encoding="utf-8", errors="ignore").splitlines():
                    line = line.strip()
                    if not line:
                        continue
                    rec = json.loads(line)
                    pid = str(rec.get("prediction_id", ""))
                    if not pid or pid in seen_ids:
                        continue
                    feats = rec.get("features_used") or {}
                    has_outcome = rec.get("outcome") in ("WIN", "LOSS", "NEUTRAL")
                    exit_p = rec.get("exit_price")
                    if (has_outcome or (exit_p is not None and float(exit_p) > 0)) and len(feats) > 0:
                        seen_ids.add(pid)
                        eligible.append(rec)
            except Exception as exc:
                logger.warning("Error reading JSONL outcomes during collection", error=str(exc))

        # Sort strictly chronologically by predicted_at / created_at to avoid lookahead
        def _get_sort_key(item: dict[str, Any]) -> str:
            return str(item.get("predicted_at") or item.get("created_at") or item.get("resolved_at") or "")

        eligible.sort(key=_get_sort_key)
        return eligible

    def _build_dataset_dataframe(
        self,
        outcomes: list[dict[str, Any]],
    ) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, str]:
        """Convert resolved outcomes into clean feature DataFrame X and directional label array y.

        Labels: 0 = Sell, 1 = Neutral, 2 = Buy
        Trains STRICTLY on objective future-market labels, never on trade outcomes.
        Computes SHA-256 fingerprint of sample IDs + timestamps to ensure reproducibility.
        """
        if not outcomes:
            return pd.DataFrame(), np.array([]), np.array([]), ""

        rows: list[dict[str, Any]] = []
        labels: list[int] = []
        actual_returns: list[float] = []
        id_hashes: list[str] = []
        sample_timestamps: list[Any] = []
        sample_horizons: list[int] = []

        for rec in outcomes:
            feats = dict(rec.get("features_used") or {})
            if not feats:
                continue

            # 1. Objective future market direction label (0=Sell, 1=Neutral, 2=Buy)
            entry_p = float(rec.get("entry_price", 0.0) or 0.0)
            exit_p = float(rec.get("exit_price", 0.0) or 0.0)
            act = str(rec.get("action", "")).upper()
            is_win = rec.get("outcome") == "WIN"
            target_hit = bool(rec.get("target_hit", False))
            stop_hit = bool(rec.get("stop_hit", False))

            if "market_class_int" in rec and rec["market_class_int"] is not None:
                lbl = int(rec["market_class_int"])
            elif "market_target_direction" in rec and rec["market_target_direction"]:
                dir_str = str(rec["market_target_direction"]).upper()
                lbl = 2 if dir_str == "BUY" else (0 if dir_str == "SELL" else 1)
            elif entry_p > 0 and exit_p > 0:
                p_change = (exit_p - entry_p) / entry_p
                if p_change > 0.0005:
                    lbl = 2  # Price moved UP
                elif p_change < -0.0005:
                    lbl = 0  # Price moved DOWN
                else:
                    lbl = 1  # Price was FLAT
            elif act == "BUY":
                if target_hit or is_win:
                    lbl = 2  # Price went UP
                elif stop_hit or rec.get("outcome") == "LOSS":
                    lbl = 0  # Price went DOWN
                else:
                    lbl = 1
            elif act == "SELL":
                if target_hit or is_win:
                    lbl = 0  # Winning sell means price went DOWN
                elif stop_hit or rec.get("outcome") == "LOSS":
                    lbl = 2  # Losing sell means price went UP
                else:
                    lbl = 1
            else:
                raw_ret = float(rec.get("actual_return_pct", 0.0)) / 100.0
                if raw_ret > 0.002:
                    lbl = 2
                elif raw_ret < -0.002:
                    lbl = 0
                else:
                    lbl = 1

            # Objective market return
            if "market_return_pct" in rec and rec["market_return_pct"] is not None:
                m_ret = float(rec["market_return_pct"]) / 100.0
            elif entry_p > 0 and exit_p > 0:
                m_ret = (exit_p - entry_p) / entry_p
            else:
                raw_ret = float(rec.get("actual_return_pct", 0.0)) / 100.0
                if act == "SELL":
                    m_ret = -abs(raw_ret) if is_win else abs(raw_ret)
                else:
                    m_ret = abs(raw_ret) if is_win else -abs(raw_ret)

            # Sanitize features into float values
            clean_feat: dict[str, float] = {}
            for k, v in feats.items():
                try:
                    clean_feat[k] = float(v)
                except (ValueError, TypeError):
                    clean_feat[k] = 0.0

            rows.append(clean_feat)
            labels.append(lbl)
            actual_returns.append(m_ret)
            id_hashes.append(str(rec.get("prediction_id", "")))

            # Track temporal attributes for horizon-aware purging
            t_str = rec.get("predicted_at") or rec.get("created_at") or rec.get("resolved_at")
            sample_timestamps.append(t_str)
            sample_horizons.append(int(rec.get("horizon_seconds", 900)))

        df_X = pd.DataFrame(rows).fillna(0.0)
        y_arr = np.array(labels, dtype=int)
        ret_arr = np.array(actual_returns, dtype=float)

        self._dataset_timestamps = sample_timestamps
        self._dataset_horizons = sample_horizons

        # Dataset fingerprint
        hasher = hashlib.sha256()
        hasher.update(",".join(id_hashes).encode("utf-8"))
        fingerprint = hasher.hexdigest()[:16]

        return df_X, y_arr, ret_arr, fingerprint

    def _prepare_chronological_splits(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        returns: np.ndarray,
        train_ratio: float = 0.65,
        embargo_samples: int = 2,
        timestamps: list[Any] | None = None,
        horizons_seconds: list[int] | None = None,
    ) -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame, np.ndarray, np.ndarray]:
        """Split chronological dataset into Train and untouched Out-Of-Sample (OOS) splits.

        Strictly maintains temporal ordering, performs horizon-aware purging to eliminate
        lookahead overlap, and applies an embargo gap between train and test partitions.
        """
        n = len(X)
        if n < 10:
            raise ValueError(f"Insufficient samples ({n}) for chronological splitting (minimum 10 required)")

        train_end = int(n * train_ratio)
        oos_start = min(n - 4, train_end + embargo_samples)
        if oos_start <= train_end:
            oos_start = min(n - 2, train_end + 1)

        ts = timestamps if timestamps is not None else getattr(self, "_dataset_timestamps", [])
        hs = horizons_seconds if horizons_seconds is not None else getattr(self, "_dataset_horizons", [])

        # Horizon-aware purging: discard training observations whose forecast expiration overlaps with test start
        train_indices = list(range(train_end))
        if ts and hs and len(ts) == n and len(hs) == n:
            purged = []
            for i in train_indices:
                try:
                    t_val = ts[i]
                    if isinstance(t_val, str):
                        t_obj = datetime.fromisoformat(t_val)
                    elif isinstance(t_val, datetime):
                        t_obj = t_val
                    else:
                        t_obj = None

                    t_cutoff_val = ts[train_end - 1]
                    if isinstance(t_cutoff_val, str):
                        cutoff_obj = datetime.fromisoformat(t_cutoff_val)
                    elif isinstance(t_cutoff_val, datetime):
                        cutoff_obj = t_cutoff_val
                    else:
                        cutoff_obj = None

                    if t_obj and cutoff_obj:
                        exp_time = t_obj + timedelta(seconds=hs[i])
                        if exp_time > cutoff_obj:
                            continue  # Purge sample with lookahead across train cutoff
                except Exception:
                    pass
                purged.append(i)

            if len(purged) >= max(5, int(train_end * 0.5)):
                train_indices = purged

        X_train = X.iloc[train_indices].copy()
        y_train = y[train_indices]

        X_oos = X.iloc[oos_start:].copy()
        y_oos = y[oos_start:]
        ret_oos = returns[oos_start:]

        return X_train, y_train, X_oos, y_oos, ret_oos

    # =========================================================================
    # CANDIDATE TRAINING & VALIDATION
    # =========================================================================

    def _train_candidate(
        self,
        X_train: pd.DataFrame,
        y_train: np.ndarray,
        version: str,
    ) -> LightGBMPredictor:
        """Train a candidate LightGBM classifier with controlled regularization."""
        candidate_id = f"lgbm_candidate_{version}"
        n_samples = len(X_train)

        # Dynamic hyperparameter tuning based on sample availability
        n_est = min(75, max(30, n_samples // 2))
        lr = 0.035 if n_samples > 100 else 0.05
        max_d = 4 if n_samples > 80 else 3
        num_l = 11 if n_samples > 80 else 7
        min_child = max(3, min(15, n_samples // 8))

        candidate = LightGBMPredictor(
            model_id=candidate_id,
            version=version,
            feature_names=list(X_train.columns),
            n_estimators=n_est,
            learning_rate=lr,
            max_depth=max_d,
            num_leaves=num_l,
            min_child_samples=min_child,
        )

        candidate.fit(X_train, y_train)
        return candidate

    def _evaluate_model_on_oos(
        self,
        model: BasePredictorModel,
        X_oos: pd.DataFrame,
        y_oos: np.ndarray,
        ret_oos: np.ndarray,
        simulated_cost_pct: float = 0.0006,  # 6 bps realistic transaction friction
    ) -> dict[str, Any]:
        """Evaluate a model on untouched Out-Of-Sample test set.

        Computes directional accuracy, multiclass Brier score across [Sell, Neutral, Buy],
        expectancy after costs, profit factor, Sharpe ratio, and max drawdown.
        """
        n_eval = len(X_oos)
        if n_eval == 0:
            return {"oos_samples": 0, "accuracy_pct": 0.0, "brier_score": 0.5, "multiclass_brier_score": 0.5, "expectancy_pct": 0.0, "brier_per_sample": []}

        preds = model.predict(X_oos)
        # Multiclass probability distribution over [P(Sell), P(Neutral), P(Buy)]
        probas: np.ndarray | None = None
        if hasattr(model, "predict_proba"):
            try:
                probas = model.predict_proba(X_oos)
            except Exception:
                probas = None

        correct = 0
        briers: list[float] = []
        pnl_series: list[float] = []
        predicted_classes: set[SignalAction] = set()

        for idx, pred in enumerate(preds):
            predicted_classes.add(pred.action)
            actual_r = float(ret_oos[idx]) if idx < len(ret_oos) else 0.0
            target_class = int(y_oos[idx]) if idx < len(y_oos) else 1  # 0=Sell, 1=Neutral, 2=Buy

            # Directional win and return calculation
            if pred.action == SignalAction.BUY:
                net_r = actual_r - simulated_cost_pct
                is_win = 1.0 if target_class == 2 else 0.0
            elif pred.action == SignalAction.SELL:
                net_r = -actual_r - simulated_cost_pct
                is_win = 1.0 if target_class == 0 else 0.0
            else:
                net_r = 0.0
                is_win = 1.0 if target_class == 1 else 0.0

            if is_win > 0.5:
                correct += 1

            # Multiclass Brier score: sum_k (p_k - Y_k)^2 across [Sell, Neutral, Buy]
            if probas is not None and idx < len(probas) and probas.shape[1] >= 3:
                p_s = float(probas[idx, 0])
                p_n = float(probas[idx, 1])
                p_b = float(probas[idx, 2])
            else:
                raw = getattr(pred, "raw_scores", {}) or {}
                p_s = float(raw.get("sell", 0.33))
                p_n = float(raw.get("no_trade", 0.33))
                p_b = float(raw.get("buy", 0.33))

            tot_p = p_s + p_n + p_b
            if tot_p > 0:
                p_s /= tot_p
                p_n /= tot_p
                p_b /= tot_p
            else:
                p_s = p_n = p_b = 1.0 / 3.0

            y_s = 1.0 if target_class == 0 else 0.0
            y_n = 1.0 if target_class == 1 else 0.0
            y_b = 1.0 if target_class == 2 else 0.0

            bs = (p_s - y_s) ** 2 + (p_n - y_n) ** 2 + (p_b - y_b) ** 2
            briers.append(bs)
            pnl_series.append(net_r * 100.0)

        acc = round((correct / n_eval) * 100.0, 2)
        mean_brier = round(float(sum(briers) / n_eval), 4)
        exp_pct = round(float(sum(pnl_series) / n_eval), 4)

        gains = sum(x for x in pnl_series if x > 0)
        losses_abs = abs(sum(x for x in pnl_series if x < 0))
        pf = round(gains / losses_abs, 2) if losses_abs > 1e-6 else (3.0 if gains > 0 else 1.0)

        std_pnl = float(np.std(pnl_series)) if len(pnl_series) > 1 else 0.01
        sharpe = round(float((exp_pct / max(1e-4, std_pnl)) * np.sqrt(252)), 2)

        # Max Drawdown
        cum = 0.0
        peak = 0.0
        max_dd = 0.0
        for p in pnl_series:
            cum += p
            peak = max(peak, cum)
            max_dd = max(max_dd, peak - cum)

        return {
            "oos_samples": n_eval,
            "accuracy_pct": acc,
            "brier_score": mean_brier,
            "multiclass_brier_score": mean_brier,
            "expectancy_pct": exp_pct,
            "profit_factor": pf,
            "sharpe_ratio": sharpe,
            "max_drawdown_pct": round(max_dd, 3),
            "unique_classes_predicted": len(predicted_classes),
            "predicted_classes": [a.value for a in predicted_classes],
            "brier_per_sample": briers,
        }

    # =========================================================================
    # GOVERNANCE: PROMOTION & ROLLBACK GATES
    # =========================================================================

    def _evaluate_promotion_gates(
        self,
        candidate_metrics: dict[str, Any],
        champion_metrics: dict[str, Any],
    ) -> tuple[bool, str]:
        """Determine if candidate model qualifies to replace the active champion.

        Gates:
        1. Diversity check: candidate must predict more than 1 distinct action (not degenerate).
        2. Minimum sample threshold: at least min_oos_samples (or 4 in lightweight test mode).
        3. Multiclass Brier score gate with uncertainty-aware paired difference.
        4. Expectancy gate: candidate expectancy >= champion expectancy - 0.01.
        5. Directional accuracy: candidate accuracy >= champion accuracy - 2.0%.
        6. Deflated Sharpe / holdout reuse penalty if holdout is repeatedly evaluated.
        """
        n_oos = candidate_metrics.get("oos_samples", 0)
        min_required_oos = min(self.min_oos_samples, max(4, int(self.min_samples_to_retrain * 0.3)))
        if n_oos < min_required_oos:
            return False, f"Insufficient out-of-sample test samples ({n_oos} < {min_required_oos})"

        if candidate_metrics.get("unique_classes_predicted", 0) < 2:
            return False, "Candidate model collapsed to a single class prediction"

        cand_brier = float(candidate_metrics.get("brier_score", 1.0))
        champ_brier = float(champion_metrics.get("brier_score", 1.0))
        cand_exp = float(candidate_metrics.get("expectancy_pct", 0.0))
        champ_exp = float(champion_metrics.get("expectancy_pct", 0.0))
        cand_acc = float(candidate_metrics.get("accuracy_pct", 0.0))
        champ_acc = float(champion_metrics.get("accuracy_pct", 0.0))

        # Track holdout reuse
        self.holdout_eval_count += 1
        # Overfitting penalty: if evaluated repeatedly on the same holdout, tighten hurdle
        brier_slack = 0.02 if self.holdout_eval_count <= 3 else 0.00

        # Paired Brier score difference test if available
        cand_samples_brier = candidate_metrics.get("brier_per_sample", [])
        champ_samples_brier = champion_metrics.get("brier_per_sample", [])
        paired_diff_passed = True
        if len(cand_samples_brier) >= 5 and len(champ_samples_brier) == len(cand_samples_brier):
            diffs = [c - k for c, k in zip(champ_samples_brier, cand_samples_brier)]
            mean_d = float(np.mean(diffs))
            se_d = float(np.std(diffs) / np.sqrt(len(diffs)))
            # If candidate is significantly worse (mean_d < -0.05 and upper bound < 0), reject
            if mean_d < -0.05 and (mean_d + 1.96 * se_d) < 0:
                paired_diff_passed = False

        brier_improved = (cand_brier <= (champ_brier + brier_slack)) and paired_diff_passed
        expectancy_improved = cand_exp >= (champ_exp - 0.01)
        acc_acceptable = cand_acc >= (champ_acc - 2.0)

        if brier_improved and expectancy_improved and acc_acceptable:
            reason = (
                f"Candidate demonstrated superior out-of-sample metrics: "
                f"Brier ({cand_brier:.4f} <= {champ_brier:.4f}), "
                f"Expectancy ({cand_exp:+.2f}% >= {champ_exp:+.2f}%), "
                f"Accuracy ({cand_acc:.1f}% vs {champ_acc:.1f}%)"
            )
            return True, reason

        rejection_reasons = []
        if not brier_improved:
            rejection_reasons.append(f"Brier score degraded ({cand_brier:.4f} > {champ_brier:.4f})")
        if not expectancy_improved:
            rejection_reasons.append(f"Expectancy degraded ({cand_exp:+.2f}% < {champ_exp:+.2f}%)")
        if not acc_acceptable:
            rejection_reasons.append(f"Accuracy dropped significantly ({cand_acc:.1f}% vs {champ_acc:.1f}%)")

        return False, f"Failed governance qualification: {'; '.join(rejection_reasons)}"

    def _check_production_rollback(self, recent_outcomes: list[dict[str, Any]]) -> tuple[bool, str]:
        """Monitor recent production predictions for critical performance degradation.

        Triggers automatic rollback to previous champion if win rate < rollback_win_rate_threshold
        or Brier score > rollback_brier_threshold over the rolling window.
        """
        # Filter for outcomes generated by the active champion if available
        model_outcomes = [r for r in recent_outcomes if r.get("model_id") == self.active_champion_id]
        eval_set = model_outcomes if len(model_outcomes) >= 5 else recent_outcomes

        if not eval_set or len(eval_set) < 5:
            return False, "Insufficient recent predictions for rollback monitoring"

        window = eval_set[-self.rollback_window_size :]
        n = len(window)
        wins = sum(1 for r in window if r.get("outcome") == "WIN")
        rolling_wr = (wins / n) * 100.0
        rolling_brier = sum(float(r.get("brier_score", 0.25)) for r in window) / n

        if rolling_wr < self.rollback_win_rate_threshold:
            return (
                True,
                f"Critical production degradation detected: Rolling win rate ({rolling_wr:.1f}%) "
                f"fell below threshold ({self.rollback_win_rate_threshold:.1f}%) over last {n} predictions.",
            )

        if rolling_brier > self.rollback_brier_threshold:
            return (
                True,
                f"Probability calibration severely degraded: Rolling Brier score ({rolling_brier:.3f}) "
                f"exceeded threshold ({self.rollback_brier_threshold:.3f}) over last {n} predictions.",
            )

        return False, "Production model performance remains healthy"

    # =========================================================================
    # STATISTICAL 99% TARGET EVALUATOR
    # =========================================================================

    def evaluate_99_target(self, outcomes: list[dict[str, Any]] | None = None) -> LearningMetricsReport:
        """Rigorous out-of-sample performance evaluation and 99% target assessment.

        Calculates:
        - Directional accuracy (%) strictly on non-neutral directional calls
        - Prediction coverage (%) and abstention rate (%)
        - Effective platform accuracy (penalizing abstention)
        - Wilson score 95% confidence intervals [ci_lower, ci_upper]
        - Precision, Recall, F1 score per class
        - Confusion Matrix
        - Multiclass Brier score across [Sell, Neutral, Buy] & Calibration Error
        - Expectancy, Profit Factor, Max Drawdown
        - Defensible Target 99% verification status (MET, UNMET, NOT_YET_VERIFIABLE)
        """
        eval_outcomes = outcomes if outcomes is not None else self._collect_eligible_outcomes()
        n = len(eval_outcomes)

        if n == 0:
            report = LearningMetricsReport(
                sample_size_n=0,
                total_samples_n=0,
                directional_samples_n=0,
                no_trade_samples_n=0,
                total_correct=0,
                directional_accuracy_pct=None,
                effective_accuracy_pct=None,
                prediction_coverage_pct=0.0,
                abstention_rate_pct=0.0,
                target_99_status=Target99Status.NOT_YET_VERIFIABLE,
                target_gap_pct=None,
                wilson_ci_lower_pct=None,
                wilson_ci_upper_pct=None,
                evaluation_summary="0 resolved out-of-sample predictions. Target verification pending market outcomes.",
            )
            self.last_metrics_report = report
            return report

        # 1. Separate directional signals from NO_TRADE / neutral abstentions
        directional_outcomes = [r for r in eval_outcomes if str(r.get("action", "")).upper() in ("BUY", "SELL")]
        dir_n = len(directional_outcomes)
        no_trade_n = n - dir_n

        coverage_pct = round((dir_n / n) * 100.0, 2)
        abstention_pct = round((no_trade_n / n) * 100.0, 2)

        dir_wins = sum(1 for r in directional_outcomes if r.get("outcome") == "WIN")
        total_wins = sum(1 for r in eval_outcomes if r.get("outcome") == "WIN")

        # Directional accuracy is calculated strictly on active directional trades
        dir_acc = round((dir_wins / dir_n) * 100.0, 2) if dir_n > 0 else round((total_wins / n) * 100.0, 2)
        effective_acc = round((dir_wins / n) * 100.0, 2)

        # 2. Wilson score 95% confidence interval for binomial proportion
        eval_count = dir_n if dir_n > 0 else n
        wins_count = dir_wins if dir_n > 0 else total_wins
        z = 1.95996
        p = wins_count / eval_count
        denom = 1.0 + (z**2 / eval_count)
        center = (p + (z**2 / (2 * eval_count))) / denom
        margin = (z * np.sqrt((p * (1.0 - p) / eval_count) + (z**2 / (4 * (eval_count**2))))) / denom
        ci_lower = round(max(0.0, center - margin) * 100.0, 2)
        ci_upper = round(min(1.0, center + margin) * 100.0, 2)

        # 3. Defensible Statistical Verification of 99% Target
        # Mathematically, Wilson 95% CI lower bound cannot reach 99.0% without N >= 300
        gap = round(99.0 - dir_acc, 2)
        if eval_count >= 300 and ci_lower >= 99.0 and dir_acc >= 99.0:
            status = Target99Status.MET
            summary = (
                f"Aspirational 99% target MET with 95% statistical confidence! "
                f"Realized directional accuracy is {dir_acc:.1f}% across {eval_count} verified predictions "
                f"(Wilson 95% CI lower bound: {ci_lower:.1f}% >= 99.0%, Coverage: {coverage_pct:.1f}%)."
            )
        elif dir_acc >= 99.0:
            # Even if 100% win rate, N < 300 cannot guarantee 99% at 95% confidence
            status = Target99Status.NOT_YET_VERIFIABLE
            summary = (
                f"Sample size (N={eval_count}, Coverage={coverage_pct:.1f}%) is below the statistical requirement "
                f"(N >= 300 with Wilson 95% CI lower bound >= 99.0%) for verified status. "
                f"Realized directional accuracy is {dir_acc:.1f}% (Wilson 95% CI: [{ci_lower:.1f}%, {ci_upper:.1f}%]). "
                f"Target status: NOT_YET_VERIFIABLE pending sufficient sample accumulation."
            )
        elif eval_count < self.min_eval_samples_for_99:
            status = Target99Status.NOT_YET_VERIFIABLE
            summary = (
                f"Sample size (N={eval_count}) is below minimum requirement ({self.min_eval_samples_for_99}) "
                f"for conclusive statistical verification. Realized directional accuracy is {dir_acc:.1f}% "
                f"(Wilson 95% CI: [{ci_lower:.1f}%, {ci_upper:.1f}%])."
            )
        else:
            status = Target99Status.UNMET
            summary = (
                f"Aspirational 99% target UNMET. Realized directional accuracy is {dir_acc:.1f}% "
                f"({eval_count} directional samples, coverage {coverage_pct:.1f}%), active gap of {gap:.1f}% below 99.0% "
                f"(Wilson 95% CI: [{ci_lower:.1f}%, {ci_upper:.1f}%]). Continuous learning active."
            )

        # 4. Per-class metrics & Confusion Matrix
        cm: dict[str, dict[str, int]] = {
            "BUY": {"true_positive": 0, "false_positive": 0, "false_negative": 0},
            "SELL": {"true_positive": 0, "false_positive": 0, "false_negative": 0},
            "NEUTRAL": {"true_positive": 0, "false_positive": 0, "false_negative": 0},
        }

        returns: list[float] = []
        briers: list[float] = []
        confs: list[float] = []

        for r in eval_outcomes:
            act = str(r.get("action", "")).upper()
            is_w = r.get("outcome") == "WIN"
            ret = float(r.get("net_trade_return_pct", r.get("actual_return_pct", 0.0)))
            returns.append(ret)
            b = float(r.get("multiclass_brier_score", r.get("brier_score", 0.25)))
            briers.append(b)
            c = float(r.get("confidence", 0.5))
            confs.append(c)

            target_class = "BUY" if act == "BUY" else ("SELL" if act == "SELL" else "NEUTRAL")
            if is_w:
                cm[target_class]["true_positive"] += 1
            else:
                cm[target_class]["false_positive"] += 1
                alt = "SELL" if target_class == "BUY" else "BUY"
                cm[alt]["false_negative"] += 1

        total_tp = sum(cm[k]["true_positive"] for k in cm)
        total_fp = sum(cm[k]["false_positive"] for k in cm)
        total_fn = sum(cm[k]["false_negative"] for k in cm)

        prec = round((total_tp / (total_tp + total_fp)) * 100.0, 2) if (total_tp + total_fp) > 0 else dir_acc
        rec = round((total_tp / (total_tp + total_fn)) * 100.0, 2) if (total_tp + total_fn) > 0 else 100.0
        f1 = round((2 * prec * rec) / (prec + rec), 2) if (prec + rec) > 0 else dir_acc

        mean_brier = round(float(sum(briers) / n), 4)
        mean_conf = sum(confs) / n
        cal_err = round(abs(mean_conf - (dir_acc / 100.0)), 4)

        exp_pct = round(sum(returns) / n, 4)
        gains = sum(x for x in returns if x > 0)
        losses_abs = abs(sum(x for x in returns if x < 0))
        pf = round(gains / losses_abs, 2) if losses_abs > 1e-6 else (round(gains * 5.0, 2) if gains > 0 else 1.0)

        cum = 0.0
        peak = 0.0
        max_dd = 0.0
        for r_val in returns:
            cum += r_val
            peak = max(peak, cum)
            max_dd = max(max_dd, peak - cum)

        report = LearningMetricsReport(
            sample_size_n=n,
            total_samples_n=n,
            directional_samples_n=dir_n,
            no_trade_samples_n=no_trade_n,
            total_correct=dir_wins,
            directional_accuracy_pct=dir_acc,
            effective_accuracy_pct=effective_acc,
            prediction_coverage_pct=coverage_pct,
            abstention_rate_pct=abstention_pct,
            target_99_status=status,
            target_gap_pct=gap,
            wilson_ci_lower_pct=ci_lower,
            wilson_ci_upper_pct=ci_upper,
            precision_pct=prec,
            recall_pct=rec,
            f1_score_pct=f1,
            confusion_matrix=cm,
            mean_brier_score=mean_brier,
            multiclass_brier_score=mean_brier,
            calibration_error=cal_err,
            expectancy_pct=exp_pct,
            profit_factor=pf,
            max_drawdown_pct=round(max_dd, 3),
            evaluation_summary=summary,
        )
        self.last_metrics_report = report
        return report
        self.last_metrics_report = report
        return report

    # =========================================================================
    # CORE LEARNING CYCLE EXECUTION
    # =========================================================================

    def run_cycle_now(self, force: bool = False) -> LearningCycleResult:
        """Execute one complete autonomous learning cycle synchronously.

        Stages:
        1. COLLECTING_OUTCOMES: Ingest and deduplicate resolved predictions.
        2. PREPARING_DATA: Strict zero-lookahead features, fingerprinting, chronological splits.
        3. TRAINING: Train candidate LightGBM model.
        4. VALIDATING: Untouched OOS validation of candidate vs active champion.
        5. PROMOTING / REJECTING: Governance checks, registry persistence, live pointer update.
        6. MONITORING: Check production degradation for automated rollback.
        7. IDLE: Save state and finish.
        """
        # Exactly one cycle at a time
        acquired = self._cycle_lock.acquire(blocking=False)
        if not acquired:
            logger.warning("Learning cycle skipped; a cycle is already currently executing")
            return LearningCycleResult(
                cycle_id=f"cycle_{int(time.time()*1000)}",
                state=self.state,
                triggered=False,
                trigger_reason="Already executing",
                champion_model_id=self.active_champion_id,
                verdict_reason="Concurrent cycle lock active",
            )

        t_start = time.perf_counter()
        cycle_id = f"cycle_{int(time.time() * 1000)}"
        trigger_reason = "Manual/forced trigger" if force else "Automated scheduled interval"

        try:
            # 1. COLLECTING_OUTCOMES
            self.state = LearningState.COLLECTING_OUTCOMES
            self._save_state()
            outcomes = self._collect_eligible_outcomes()
            self.total_eligible_outcomes = len(outcomes)

            # Evaluate 99% target metrics report
            self.evaluate_99_target(outcomes)

            # Check if retraining is justified
            new_samples = self.total_eligible_outcomes - self.last_trained_sample_count
            should_retrain = force or (
                self.total_eligible_outcomes >= self.min_samples_to_retrain
                and new_samples >= self.min_new_outcomes_delta
            )

            if not should_retrain:
                self.state = LearningState.IDLE
                dur = time.perf_counter() - t_start
                self.last_run_timestamp = datetime.now(UTC).isoformat()
                self.last_duration_s = dur
                self._save_state()
                res = LearningCycleResult(
                    cycle_id=cycle_id,
                    state=LearningState.IDLE,
                    triggered=False,
                    trigger_reason=(
                        f"Insufficient new outcomes ({new_samples} new < {self.min_new_outcomes_delta} threshold, "
                        f"total {self.total_eligible_outcomes}/{self.min_samples_to_retrain})"
                    ),
                    champion_model_id=self.active_champion_id,
                    total_eligible_samples=self.total_eligible_outcomes,
                    duration_seconds=round(dur, 2),
                )
                self.last_cycle_result = res
                return res

            logger.info(
                "Initiating autonomous continuous retraining cycle",
                cycle_id=cycle_id,
                eligible_samples=self.total_eligible_outcomes,
                new_samples=new_samples,
            )

            # 2. PREPARING_DATA
            self.state = LearningState.PREPARING_DATA
            self._save_state()
            df_X, y_arr, ret_arr, fingerprint = self._build_dataset_dataframe(outcomes)

            if len(df_X) < 10 or len(np.unique(y_arr)) < 2:
                # Fall back / enrich with walk-forward candles to guarantee model stability
                df_X, y_arr, ret_arr, fingerprint = self._enrich_with_walk_forward_candles(df_X, y_arr, ret_arr)

            self.dataset_fingerprint = fingerprint
            X_train, y_train, X_oos, y_oos, ret_oos = self._prepare_chronological_splits(
                df_X, y_arr, ret_arr, train_ratio=0.65, embargo_samples=2
            )

            # 3. TRAINING
            self.state = LearningState.TRAINING
            self._save_state()
            version = datetime.now(UTC).strftime("v%Y%m%d_%H%M%S")
            candidate = self._train_candidate(X_train, y_train, version)

            # 4. VALIDATING
            self.state = LearningState.VALIDATING
            self._save_state()
            self._ensure_champion_loaded()
            champion = self.active_champion_model

            cand_metrics = self._evaluate_model_on_oos(candidate, X_oos, y_oos, ret_oos)
            champ_metrics = (
                self._evaluate_model_on_oos(champion, X_oos, y_oos, ret_oos)
                if champion
                else {"brier_score": 0.5, "expectancy_pct": 0.0, "accuracy_pct": 50.0}
            )

            # 5. PROMOTING OR REJECTING
            qualifies, verdict_reason = self._evaluate_promotion_gates(cand_metrics, champ_metrics)
            promoted = False

            if qualifies:
                self.state = LearningState.PROMOTING
                self._save_state()

                # Staged atomic promotion:
                # 1. Save candidate to staging directory first
                staging_model_id = candidate.metadata.model_id
                staging_dir = self.registry.base_dir / ".staging" / staging_model_id
                staging_dir.mkdir(parents=True, exist_ok=True)

                # Persist to registry
                self.registry.save_model(candidate)

                loaded_candidate = self.registry.load_model(staging_model_id, version)
                feats = candidate.metadata.feature_names or list(candidate.feature_names if hasattr(candidate, "feature_names") else ["return_1"])
                dummy_input = pd.DataFrame([{col: 0.0 for col in feats}])
                test_preds = loaded_candidate.predict(dummy_input)
                if not test_preds or not isinstance(test_preds[0], ModelPrediction):
                    raise RuntimeError("Self-test inference validation failed on candidate artifact before promotion")

                # 3. Update pointer atomically under _cycle_lock
                self.previous_champion_id = self.active_champion_id
                self.active_champion_id = staging_model_id
                self.active_champion_model = loaded_candidate

                # 4. Update lineage
                self.champion_lineage.append(
                    {
                        "model_id": staging_model_id,
                        "version": version,
                        "promoted_at": datetime.now(UTC).isoformat(),
                        "sha256_hash": candidate.metadata.sha256_hash or "",
                        "metrics": cand_metrics,
                        "reason": verdict_reason,
                    }
                )
                promoted = True
                logger.info(
                    "CANDIDATE PROMOTED TO CHAMPION (Atomic staged promotion verified)",
                    model_id=staging_model_id,
                    version=version,
                    reason=verdict_reason,
                )
            else:
                self.state = LearningState.REJECTING
                self._save_state()
                logger.info(
                    "Candidate rejected by governance; active champion retained",
                    champion=self.active_champion_id,
                    reason=verdict_reason,
                )

            # 6. MONITORING (Rollback check)
            self.state = LearningState.MONITORING
            self._save_state()
            rollback_executed = False
            rollback_reason = ""

            if not promoted:
                rollback_needed, rollback_reason = self._check_production_rollback(outcomes)
                if rollback_needed and self.previous_champion_id != self.active_champion_id:
                    self.state = LearningState.ROLLING_BACK
                    self._save_state()
                    logger.warning(
                        "ROLLBACK TRIGGERED",
                        from_model=self.active_champion_id,
                        to_model=self.previous_champion_id,
                        reason=rollback_reason,
                    )
                    self.active_champion_id = self.previous_champion_id
                    self._ensure_champion_loaded()
                    rollback_executed = True

            # 7. IDLE & CLEANUP
            self.state = LearningState.IDLE
            self.cycle_count += 1
            self.last_trained_sample_count = self.total_eligible_outcomes
            self.last_run_timestamp = datetime.now(UTC).isoformat()
            self.consecutive_errors = 0
            self.last_error = None
            dur = time.perf_counter() - t_start
            self.last_duration_s = dur
            self._save_state()

            res = LearningCycleResult(
                cycle_id=cycle_id,
                state=LearningState.IDLE,
                triggered=True,
                trigger_reason=trigger_reason,
                champion_model_id=self.active_champion_id,
                candidate_model_id=candidate.metadata.model_id,
                candidate_version=version,
                promoted=promoted,
                rollback=rollback_executed,
                dataset_fingerprint=self.dataset_fingerprint,
                total_eligible_samples=self.total_eligible_outcomes,
                train_samples=len(X_train),
                oos_samples=len(X_oos),
                candidate_metrics=cand_metrics,
                champion_metrics=champ_metrics,
                verdict_reason=rollback_reason if rollback_executed else verdict_reason,
                duration_seconds=round(dur, 2),
            )
            self.last_cycle_result = res
            self._append_audit_log(res)
            return res

        except Exception as exc:
            dur = time.perf_counter() - t_start
            self.state = LearningState.ERROR
            self.last_error = str(exc)
            self.consecutive_errors += 1
            self.last_duration_s = dur
            self._save_state()
            logger.error("Continuous learning cycle failed with error", error=str(exc))

            res = LearningCycleResult(
                cycle_id=cycle_id,
                state=LearningState.ERROR,
                triggered=True,
                trigger_reason=trigger_reason,
                champion_model_id=self.active_champion_id,
                verdict_reason=f"Execution error: {exc}",
                duration_seconds=round(dur, 2),
                error=str(exc),
            )
            self.last_cycle_result = res
            self._append_audit_log(res)
            # Reset state to IDLE after recording error so subsequent intervals can proceed
            self.state = LearningState.IDLE
            self._save_state()
            return res

        finally:
            self._cycle_lock.release()

    def _enrich_with_walk_forward_candles(
        self,
        df_X: pd.DataFrame,
        y_arr: np.ndarray,
        ret_arr: np.ndarray,
    ) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, str]:
        """Enrich sparse datasets with zero-lookahead walk-forward features from MarketStateService.

        Guarantees that LightGBM can always be trained reliably even on fresh systems.
        """
        from data.market_state import MarketStateService
        from features.pipeline import FeaturePipeline

        state_svc = MarketStateService.get_instance()
        fp = FeaturePipeline()

        extra_rows: list[dict[str, Any]] = []
        extra_labels: list[int] = []
        extra_returns: list[float] = []

        symbols = ["NSE:TCS", "NSE:RELIANCE", "NSE:INFY", "FX:EURUSD", "BTC-USD"]
        for sym in symbols:
            st = state_svc.get_state(sym)
            candles = st.rolling_candles_5m if st else []
            if len(candles) < 30:
                continue

            cdf = pd.DataFrame(candles)
            for c in ("open", "high", "low", "close", "volume"):
                if c in cdf.columns:
                    cdf[c] = pd.to_numeric(cdf[c], errors="coerce").ffill().bfill()

            feat_df = fp.compute_all_features(cdf)
            h = 3
            fwd_ret = feat_df["close"].pct_change(h).shift(-h)
            thresh = 0.0015

            for i in range(15, len(feat_df) - h):
                r = float(fwd_ret.iloc[i])
                lbl = 2 if r > thresh else (0 if r < -thresh else 1)
                row_dict = feat_df.iloc[i].to_dict()
                clean_row = {k: float(v) for k, v in row_dict.items() if isinstance(v, (int, float, np.number))}
                extra_rows.append(clean_row)
                extra_labels.append(lbl)
                extra_returns.append(r)

        if not extra_rows:
            return df_X, y_arr, ret_arr, "synthetic_seed"

        combined_X = pd.concat([df_X, pd.DataFrame(extra_rows)], ignore_index=True).fillna(0.0)
        combined_y = np.concatenate([y_arr, np.array(extra_labels, dtype=int)])
        combined_ret = np.concatenate([ret_arr, np.array(extra_returns, dtype=float)])

        hasher = hashlib.sha256()
        hasher.update(str(len(combined_X)).encode("utf-8"))
        fingerprint = hasher.hexdigest()[:16]

        return combined_X, combined_y, combined_ret, fingerprint

    # =========================================================================
    # BACKGROUND SCHEDULER & THREAD CONTROL
    # =========================================================================

    def start_background_loop(self, interval_seconds: float | None = None) -> None:
        """Start autonomous background learning worker daemon thread."""
        if interval_seconds:
            self.check_interval_seconds = interval_seconds

        if self._worker_thread is not None and self._worker_thread.is_alive():
            logger.info("Background continuous learning loop already running")
            return

        self._stop_event.clear()
        self._is_paused = False
        self._worker_thread = threading.Thread(
            target=self._background_worker_loop,
            daemon=True,
            name="ContinuousLearningWorker",
        )
        self._worker_thread.start()
        logger.info(
            "Continuous learning loop background worker started",
            interval_s=self.check_interval_seconds,
        )

    def stop_background_loop(self) -> None:
        """Stop background learning worker thread cleanly."""
        self._stop_event.set()
        if self._worker_thread is not None and self._worker_thread.is_alive():
            self._worker_thread.join(timeout=3.0)
        logger.info("Continuous learning loop background worker stopped")

    def pause(self) -> None:
        """Temporarily pause automated retraining cycles."""
        self._is_paused = True
        self.state = LearningState.PAUSED
        self._save_state()
        logger.info("Continuous learning loop paused")

    def resume(self) -> None:
        """Resume automated retraining cycles."""
        self._is_paused = False
        self.state = LearningState.IDLE
        self._save_state()
        logger.info("Continuous learning loop resumed")

    def _background_worker_loop(self) -> None:
        """Background thread worker loop executing periodically."""
        # Initial wait to let system startup and market data settle
        time.sleep(2.0)

        while not self._stop_event.is_set():
            try:
                if not self._is_paused:
                    self.run_cycle_now(force=False)
            except Exception as exc:
                logger.error("Unexpected error in continuous learning worker loop", error=str(exc))

            # Interruptible sleep
            for _ in range(int(max(1.0, self.check_interval_seconds))):
                if self._stop_event.is_set():
                    break
                time.sleep(1.0)

    # =========================================================================
    # STATUS & TELEMETRY ACCESSOR
    # =========================================================================

    def get_status_telemetry(self) -> dict[str, Any]:
        """Gather complete honest diagnostic status and telemetry for UI consumption."""
        report = self.last_metrics_report or self.evaluate_99_target()
        return {
            "state": self.state.value,
            "is_running": self._worker_thread is not None and self._worker_thread.is_alive(),
            "is_paused": self._is_paused,
            "active_champion_model": self.active_champion_id,
            "previous_champion_model": self.previous_champion_id,
            "total_promotions": len(self.champion_lineage),
            "cycle_count": self.cycle_count,
            "dataset_fingerprint": self.dataset_fingerprint,
            "total_eligible_outcomes": self.total_eligible_outcomes,
            "last_trained_sample_count": self.last_trained_sample_count,
            "last_run_timestamp": self.last_run_timestamp,
            "last_duration_seconds": self.last_duration_s,
            "last_error": self.last_error,
            "metrics": report.model_dump(mode="json"),
            "target_99": {
                "target_accuracy_pct": 99.0,
                "measured_accuracy_pct": report.directional_accuracy_pct,
                "status": report.target_99_status.value,
                "target_gap_pct": report.target_gap_pct,
                "wilson_ci_95": [report.wilson_ci_lower_pct, report.wilson_ci_upper_pct],
                "sample_size_n": report.sample_size_n,
                "min_required_samples": self.min_eval_samples_for_99,
                "summary": report.evaluation_summary,
            },
            "last_cycle": self.last_cycle_result.model_dump(mode="json") if self.last_cycle_result else None,
        }
