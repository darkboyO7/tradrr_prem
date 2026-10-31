"""Trading Intelligence Platform - LightGBM & Gradient Boosted ML Classifier.

Trainable machine learning predictor with calibrated multiclass probability output,
strict class order contract (SELL=0, NEUTRAL=1, BUY=2), feature importance extraction,
and artifact serialization.
"""

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd

from models.base import BasePredictorModel, ModelMetadata, ModelPrediction, SignalAction

# =============================================================================
# DOCUMENTED CLASS CONTRACT CONSTANTS
# =============================================================================
CLASS_SELL: int = 0
CLASS_NEUTRAL: int = 1
CLASS_BUY: int = 2
CLASS_NAMES: tuple[str, ...] = ("SELL", "NEUTRAL", "BUY")
SUPPORTED_CLASSES: tuple[int, ...] = (CLASS_SELL, CLASS_NEUTRAL, CLASS_BUY)
REQUIRED_NUM_CLASSES: int = 3

DEFAULT_MIN_CONFIDENCE: float = 0.42
DEFAULT_MIN_MARGIN: float = 0.05
UNIFORM_PRIOR: float = 1.0 / 3.0
BAYESIAN_PRIOR_ALPHA: float = 0.15


def validate_and_normalize_probabilities(
    probas: np.ndarray,
    n_expected_rows: int,
) -> np.ndarray:
    """Validate probability matrix contract: shape (N, 3), finite, non-negative, normalized rows."""
    if not isinstance(probas, np.ndarray):
        probas = np.asarray(probas, dtype=float)

    if probas.ndim != 2 or probas.shape != (n_expected_rows, REQUIRED_NUM_CLASSES):
        raise ValueError(
            f"Probability shape contract violation: expected ({n_expected_rows}, {REQUIRED_NUM_CLASSES}), "
            f"got {probas.shape}"
        )

    # Sanitize NaN / Inf
    if not np.all(np.isfinite(probas)):
        probas = np.nan_to_num(probas, nan=UNIFORM_PRIOR, posinf=1.0, neginf=0.0)

    # Non-negativity clip
    probas = np.clip(probas, 0.0, None)

    # Normalize row sums
    row_sums = np.sum(probas, axis=1, keepdims=True)
    zero_rows = (row_sums <= 1e-9).squeeze()

    # Safe division
    normalized = np.where(row_sums > 1e-9, probas / np.maximum(row_sums, 1e-9), UNIFORM_PRIOR)

    # Final sanity bounds
    normalized = np.clip(normalized, 0.0, 1.0)
    final_sums = np.sum(normalized, axis=1, keepdims=True)
    return normalized / np.maximum(final_sums, 1e-9)


class LightGBMPredictor(BasePredictorModel):
    """Gradient boosted decision tree classifier predicting market direction probabilities.
    
    Guarantees strict class ordering:
      Index 0: SELL (Class 0)
      Index 1: NEUTRAL (Class 1)
      Index 2: BUY (Class 2)
    """

    def __init__(
        self,
        model_id: str = "lgbm_classifier_v1",
        version: str = "v1.0.0",
        feature_names: list[str] | None = None,
        n_estimators: int = 100,
        learning_rate: float = 0.05,
        max_depth: int = 4,
        num_leaves: int = 15,
        min_child_samples: int = 20,
        random_state: int = 42,
        min_confidence: float = DEFAULT_MIN_CONFIDENCE,
        min_margin: float = DEFAULT_MIN_MARGIN,
    ) -> None:
        feats = feature_names or [
            "return_1", "log_return_5", "candle_range_pct", "candle_body_ratio",
            "rsi_14", "macd_hist", "adx_14", "atr_14", "rvol_20", "volatility_regime"
        ]
        params = {
            "n_estimators": n_estimators,
            "learning_rate": learning_rate,
            "max_depth": max_depth,
            "num_leaves": num_leaves,
            "min_child_samples": min_child_samples,
            "random_state": random_state,
            "objective": "multiclass",
            "num_class": REQUIRED_NUM_CLASSES,
            "verbosity": -1,
            "min_confidence": min_confidence,
            "min_margin": min_margin,
            "class_contract": {
                "0": "SELL",
                "1": "NEUTRAL",
                "2": "BUY",
            },
        }
        metadata = ModelMetadata(
            model_id=model_id,
            model_name="LightGBMPredictor",
            version=version,
            model_type="LightGBMPredictor",
            feature_names=feats,
            target_name="target_direction",
            parameters=params,
        )
        super().__init__(metadata)

        self.min_confidence = min_confidence
        self.min_margin = min_margin
        self.classifier = lgb.LGBMClassifier(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            num_leaves=num_leaves,
            min_child_samples=min_child_samples,
            random_state=random_state,
            objective="multiclass",
            num_class=REQUIRED_NUM_CLASSES,
            verbosity=-1,
        )
        self.is_fitted = False
        self.is_fully_supported = False
        self.fitted_classes_: list[int] = []

    def _prepare_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Filter and align DataFrame with model's expected feature contract."""
        data_dict: dict[str, Any] = {}
        for f in self.metadata.feature_names:
            if f in X.columns:
                data_dict[f] = pd.to_numeric(X[f], errors="coerce").fillna(0.0)
            else:
                data_dict[f] = 0.0
        return pd.DataFrame(data_dict, index=X.index)

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series | np.ndarray,
        strict_multiclass: bool = False,
    ) -> "LightGBMPredictor":
        """Train LightGBM classifier on features X and categorical labels y (0=Sell, 1=Neutral, 2=Buy).
        
        Args:
            X: Training feature matrix.
            y: Categorical target array (0=Sell, 1=Neutral, 2=Buy).
            strict_multiclass: If True, raises ValueError when any of the 3 classes is missing.
        """
        clean_X = self._prepare_features(X)
        y_arr = np.asarray(y, dtype=int)

        # Standardize labels from [-1, 0, 1] to [0, 1, 2] if legacy encoding supplied
        if -1 in y_arr:
            y_arr = np.where(y_arr == -1, CLASS_SELL, np.where(y_arr == 0, CLASS_NEUTRAL, CLASS_BUY))

        unique_classes = sorted([int(c) for c in np.unique(y_arr)])
        self.fitted_classes_ = unique_classes

        # Validation: 1 class or empty
        if len(unique_classes) < 2:
            self.is_fitted = False
            self.is_fully_supported = False
            if strict_multiclass:
                raise ValueError(
                    f"Insufficient class diversity: found only {len(unique_classes)} class in training set ({unique_classes}). "
                    f"Multiclass models require all 3 classes [0, 1, 2]."
                )
            return self

        # Incomplete classes (e.g. 2 classes present)
        if len(unique_classes) < REQUIRED_NUM_CLASSES:
            self.is_fully_supported = False
            missing_classes = [c for c in SUPPORTED_CLASSES if c not in unique_classes]
            if strict_multiclass:
                raise ValueError(
                    f"Incomplete class distribution: found classes {unique_classes}, missing {missing_classes}. "
                    f"Strict multiclass contract requires all 3 classes [0, 1, 2]."
                )
            # Safe non-strict fitting: fit multiclass on present classes
            self.classifier.set_params(objective="multiclass", num_class=len(unique_classes))
            self.classifier.fit(clean_X, y_arr)
            self.is_fitted = True
            self.metadata.parameters["unsupported_classes"] = missing_classes
            self.metadata.parameters["is_fully_supported"] = False
            self.metadata.trained_at = datetime.now(UTC)
            return self

        # Standard complete 3-class fitting
        self.is_fully_supported = True
        self.classifier.set_params(objective="multiclass", num_class=REQUIRED_NUM_CLASSES)
        self.classifier.fit(clean_X, y_arr)
        self.is_fitted = True
        self.metadata.parameters["unsupported_classes"] = []
        self.metadata.parameters["is_fully_supported"] = True
        self.metadata.trained_at = datetime.now(UTC)
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict probability distribution over [P(Sell=0), P(Neutral=1), P(Buy=2)].
        
        Guarantees:
        1. Always returns exactly 3 columns corresponding to [SELL, NEUTRAL, BUY].
        2. Correct mapping regardless of classifier internal class ordering.
        3. Never silently returns zero probability for an unobserved class (uses Bayesian prior smoothing).
        4. Strict shape (N, 3), finite, non-negative, rows sum to 1.0.
        """
        n_rows = len(X)
        if not self.is_fitted or not hasattr(self.classifier, "classes_") or len(self.classifier.classes_) < 2:
            # Safe uniform fallback
            return np.full((n_rows, REQUIRED_NUM_CLASSES), UNIFORM_PRIOR)

        clean_X = self._prepare_features(X)
        try:
            raw_probas = self.classifier.predict_proba(clean_X)
        except Exception:
            return np.full((n_rows, REQUIRED_NUM_CLASSES), UNIFORM_PRIOR)

        raw_classes = [int(c) for c in self.classifier.classes_]

        # Case 1: All 3 classes observed during fitting
        if set(raw_classes) == set(SUPPORTED_CLASSES) and raw_probas.shape[1] == REQUIRED_NUM_CLASSES:
            full_probas = np.zeros((n_rows, REQUIRED_NUM_CLASSES))
            for col_idx, cls_val in enumerate(raw_classes):
                full_probas[:, cls_val] = raw_probas[:, col_idx]
            return validate_and_normalize_probabilities(full_probas, n_rows)

        # Case 2: Incomplete class model (e.g. only 2 classes were fitted)
        # Never silently assign 0.0 to missing class! Blend with Bayesian Dirichlet/Laplace prior.
        full_probas = np.zeros((n_rows, REQUIRED_NUM_CLASSES))
        missing_classes = [c for c in SUPPORTED_CLASSES if c not in raw_classes]
        alpha = BAYESIAN_PRIOR_ALPHA

        # Allocate model probability scaled by (1 - alpha)
        for col_idx, cls_val in enumerate(raw_classes):
            if cls_val in SUPPORTED_CLASSES:
                full_probas[:, cls_val] = (1.0 - alpha) * raw_probas[:, col_idx]

        # Allocate prior uncertainty alpha across missing classes
        missing_share = alpha / max(1, len(missing_classes))
        for m_cls in missing_classes:
            full_probas[:, m_cls] = missing_share

        return validate_and_normalize_probabilities(full_probas, n_rows)

    def predict(self, X: pd.DataFrame) -> list[ModelPrediction]:
        """Convert input feature vectors into discrete ModelPrediction instances.
        
        Emits:
        - SignalAction.BUY: when P(Buy) > P(Sell), P(Buy) > P(Neutral), P(Buy) >= min_confidence,
          and margin over the runner-up >= min_margin.
        - SignalAction.SELL: when P(Sell) > P(Buy), P(Sell) > P(Neutral), P(Sell) >= min_confidence,
          and margin over the runner-up >= min_margin.
        - SignalAction.NO_TRADE: when evidence is insufficient, conflicting, or market is neutral.
        """
        probas = self.predict_proba(X)
        predictions = []

        for p_sell, p_neutral, p_buy in probas:
            # Runner-up calculations for margin guard
            margin_buy = p_buy - max(p_sell, p_neutral)
            margin_sell = p_sell - max(p_buy, p_neutral)

            is_buy = (
                (p_buy > p_sell)
                and (p_buy > p_neutral)
                and (p_buy >= self.min_confidence)
                and (margin_buy >= self.min_margin)
            )
            is_sell = (
                (p_sell > p_buy)
                and (p_sell > p_neutral)
                and (p_sell >= self.min_confidence)
                and (margin_sell >= self.min_margin)
            )

            if is_buy:
                action = SignalAction.BUY
                prob = p_buy
            elif is_sell:
                action = SignalAction.SELL
                prob = p_sell
            else:
                action = SignalAction.NO_TRADE
                prob = p_neutral

            # Expected return estimation from directional probability asymmetry
            exp_ret = (p_buy - p_sell) * 0.02 if action != SignalAction.NO_TRADE else 0.0

            predictions.append(
                ModelPrediction(
                    action=action,
                    probability=round(float(prob), 4),
                    expected_return=round(float(exp_ret), 5),
                    horizon_bars=5,
                    raw_scores={
                        "sell": round(float(p_sell), 4),
                        "no_trade": round(float(p_neutral), 4),
                        "buy": round(float(p_buy), 4),
                    },
                )
            )
        return predictions

    def get_feature_importance(self) -> dict[str, float]:
        """Extract normalized feature importance (split or gain based)."""
        if not self.is_fitted:
            return dict.fromkeys(self.metadata.feature_names, 0.0)

        importances = self.classifier.feature_importances_
        total = float(np.sum(importances)) or 1.0
        return {
            f: round(float(imp / total), 4)
            for f, imp in zip(self.metadata.feature_names, importances)
        }

    def save(self, directory: str | Path) -> Path:
        """Serialize model weights and metadata to a directory with SHA-256 integrity hash."""
        target_dir = Path(directory)
        target_dir.mkdir(parents=True, exist_ok=True)

        model_file = target_dir / "model.joblib"
        meta_file = target_dir / "metadata.json"

        # Update metadata state
        self.metadata.parameters["min_confidence"] = self.min_confidence
        self.metadata.parameters["min_margin"] = self.min_margin
        self.metadata.parameters["is_fully_supported"] = self.is_fully_supported
        self.metadata.parameters["fitted_classes"] = self.fitted_classes_

        # Save model weights
        joblib.dump(self.classifier, model_file)

        # Compute hash of model weights
        sha256 = hashlib.sha256()
        with open(model_file, "rb") as f:
            while chunk := f.read(65536):
                sha256.update(chunk)
        self.metadata.sha256_hash = sha256.hexdigest()

        # Save metadata
        meta_file.write_text(self.metadata.model_dump_json(indent=2), encoding="utf-8")
        return target_dir

    @classmethod
    def load(cls, directory: str | Path) -> "LightGBMPredictor":
        """Deserialize model weights and metadata with integrity validation."""
        target_dir = Path(directory)
        meta_file = target_dir / "metadata.json"
        model_file = target_dir / "model.joblib"

        if not meta_file.exists() or not model_file.exists():
            raise FileNotFoundError(f"Missing model artifacts in {target_dir}")

        metadata_dict = json.loads(meta_file.read_text(encoding="utf-8"))
        metadata = ModelMetadata(**metadata_dict)

        # Verify hash integrity
        sha256 = hashlib.sha256()
        with open(model_file, "rb") as f:
            while chunk := f.read(65536):
                sha256.update(chunk)
        current_hash = sha256.hexdigest()

        if metadata.sha256_hash and metadata.sha256_hash != current_hash:
            raise ValueError(
                f"Model integrity check failed! Expected {metadata.sha256_hash}, got {current_hash}"
            )

        params = metadata.parameters
        predictor = cls(
            model_id=metadata.model_id,
            version=metadata.version,
            feature_names=metadata.feature_names,
            n_estimators=params.get("n_estimators", 100),
            learning_rate=params.get("learning_rate", 0.05),
            max_depth=params.get("max_depth", 4),
            num_leaves=params.get("num_leaves", 15),
            min_child_samples=params.get("min_child_samples", 20),
            random_state=params.get("random_state", 42),
            min_confidence=params.get("min_confidence", DEFAULT_MIN_CONFIDENCE),
            min_margin=params.get("min_margin", DEFAULT_MIN_MARGIN),
        )
        predictor.classifier = joblib.load(model_file)
        predictor.is_fitted = True
        predictor.is_fully_supported = params.get("is_fully_supported", True)
        predictor.fitted_classes_ = params.get("fitted_classes", [0, 1, 2])
        predictor.metadata = metadata
        return predictor
