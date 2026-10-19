"""Trading Intelligence Platform - Base Model Interface & Prediction Protocols.

Defines the contract for all statistical, machine learning, and deep learning models,
enforcing feature schema contracts, model metadata lineage, and calibrated output predictions.
"""

from abc import ABC, abstractmethod
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field


class SignalAction(str, Enum):
    BUY = "BUY"
    SELL = "SELL"
    NO_TRADE = "NO_TRADE"


class ModelPrediction(BaseModel):
    """Calibrated prediction output from any statistical or ML model."""

    action: SignalAction
    probability: float = Field(ge=0.0, le=1.0, description="Confidence / probability of predicted direction")
    expected_return: float = Field(default=0.0, description="Estimated forward return magnitude")
    horizon_bars: int = Field(default=5, description="Expected forward horizon in bars")
    raw_scores: dict[str, float] = Field(default_factory=dict, description="Raw class probabilities or logits")
    metadata: dict[str, Any] = Field(default_factory=dict)


class ModelMetadata(BaseModel):
    """Complete lineage and serialization metadata for an AI/ML model."""

    model_id: str
    model_name: str
    version: str = "v1.0.0"
    model_type: str  # e.g. "LightGBMPredictor", "StatisticalBaseline"
    feature_names: list[str]
    target_name: str
    parameters: dict[str, Any] = Field(default_factory=dict)
    metrics: dict[str, float] = Field(default_factory=dict)
    trained_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    sha256_hash: str | None = None
    is_production_ready: bool = False


class BasePredictorModel(ABC):
    """Abstract base class that all candidate and production predictive models must inherit."""

    def __init__(self, metadata: ModelMetadata) -> None:
        self.metadata = metadata

    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series | np.ndarray) -> "BasePredictorModel":
        """Train the model on historical features X and target y."""
        pass

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> list[ModelPrediction]:
        """Generate structured ModelPrediction objects for incoming feature rows."""
        pass

    @abstractmethod
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Return raw probability distribution over classes."""
        pass

    @abstractmethod
    def get_feature_importance(self) -> dict[str, float]:
        """Return normalized feature importance dictionary."""
        pass

    @abstractmethod
    def save(self, directory: str | Path) -> Path:
        """Serialize model weights and metadata to a directory."""
        pass

    @classmethod
    @abstractmethod
    def load(cls, directory: str | Path) -> "BasePredictorModel":
        """Deserialize model weights and metadata from a directory."""
        pass
