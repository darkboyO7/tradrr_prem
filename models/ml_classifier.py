"""Trading Intelligence Platform - LightGBM & Gradient Boosted ML Classifier.

Trainable machine learning predictor with calibrated probability output,
feature importance extraction, and artifact serialization.
"""

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd

from models.base import BasePredictorModel, ModelMetadata, ModelPrediction, SignalAction


class LightGBMPredictor(BasePredictorModel):
    """Gradient boosted decision tree classifier predicting market direction probabilities."""

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
            "num_class": 3,
            "verbosity": -1,
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

        self.classifier = lgb.LGBMClassifier(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            num_leaves=num_leaves,
            min_child_samples=min_child_samples,
            random_state=random_state,
            objective="multiclass",
            verbosity=-1,
        )
        self.is_fitted = False

    def _prepare_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Filter and align DataFrame with model's expected feature contract."""
        clean_X = pd.DataFrame(index=X.index)
        for f in self.metadata.feature_names:
            if f in X.columns:
                clean_X[f] = pd.to_numeric(X[f], errors="coerce").fillna(0.0)
            else:
                clean_X[f] = 0.0
        return clean_X

    def fit(self, X: pd.DataFrame, y: pd.Series | np.ndarray) -> "LightGBMPredictor":
        """Train LightGBM classifier on features X and categorical labels y (0=Sell, 1=Neutral, 2=Buy)."""
        clean_X = self._prepare_features(X)
        y_arr = np.asarray(y, dtype=int)

        # Ensure labels are 0, 1, 2
        # If labels are -1, 0, 1: map to 0, 1, 2
        if -1 in y_arr:
            y_arr = np.where(y_arr == -1, 0, np.where(y_arr == 0, 1, 2))

        unique_classes = np.unique(y_arr)
        if len(unique_classes) < 2:
            self.is_fitted = False
            return self
        elif len(unique_classes) == 2:
            self.classifier.set_params(objective="binary")
            if "num_class" in self.classifier.get_params():
                self.classifier.set_params(num_class=None)
        else:
            self.classifier.set_params(objective="multiclass", num_class=len(unique_classes))

        self.classifier.fit(clean_X, y_arr)
        self.is_fitted = True
        self.metadata.trained_at = datetime.now(UTC)
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict probability distribution over [P(Sell), P(Neutral), P(Buy)]."""
        if not self.is_fitted:
            # Return neutral uniform if not fitted
            return np.full((len(X), 3), 1.0 / 3.0)

        clean_X = self._prepare_features(X)
        probas = self.classifier.predict_proba(clean_X)
        # If fewer than 3 classes were seen during training, pad to 3 columns
        if probas.shape[1] < 3:
            full_probas = np.zeros((len(X), 3))
            for i, cls_idx in enumerate(self.classifier.classes_):
                full_probas[:, int(cls_idx)] = probas[:, i]
            return full_probas
        return probas

    def predict(self, X: pd.DataFrame) -> list[ModelPrediction]:
        """Convert input feature vectors into discrete ModelPrediction instances."""
        probas = self.predict_proba(X)
        predictions = []

        for p_sell, p_neutral, p_buy in probas:
            # Minimum confidence threshold for directional signal
            if p_buy > p_sell and p_buy >= 0.52:
                action = SignalAction.BUY
                prob = p_buy
            elif p_sell > p_buy and p_sell >= 0.52:
                action = SignalAction.SELL
                prob = p_sell
            else:
                action = SignalAction.NO_TRADE
                prob = p_neutral

            # Expected return estimation from directional probability asymmetry
            exp_ret = (p_buy - p_sell) * 0.02

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

        # Save model weights
        joblib.dump(self.classifier, model_file)

        # Compute hash of model weights
        sha256 = hashlib.sha256()
        with open(model_file, "rb") as f:
            while chunk := f.read(65536):
                sha256.update(chunk)
        self.metadata.sha256_hash = sha256.hexdigest()

        # Save metadata
        meta_file.write_text(self.metadata.model_dump_json(indent=2))
        return target_dir

    @classmethod
    def load(cls, directory: str | Path) -> "LightGBMPredictor":
        """Deserialize model weights and metadata with integrity validation."""
        target_dir = Path(directory)
        meta_file = target_dir / "metadata.json"
        model_file = target_dir / "model.joblib"

        if not meta_file.exists() or not model_file.exists():
            raise FileNotFoundError(f"Missing model artifacts in {target_dir}")

        metadata_dict = json.loads(meta_file.read_text())
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
        )
        predictor.classifier = joblib.load(model_file)
        predictor.is_fitted = True
        predictor.metadata = metadata
        return predictor
