"""Trading Intelligence Platform - Model Registry & Lineage Governance.

Manages versioned model artifacts, cryptographically validates integrity,
and tracks model lifecycle states (CANDIDATE -> VALIDATING -> PRODUCTION -> RETIRED).
"""

import os
from pathlib import Path
from typing import Any

from core.config import settings
from core.database import get_db_session
from core.logging import get_logger
from data.storage.models import ModelRegistryModel
from models.base import BasePredictorModel
from models.ml_classifier import LightGBMPredictor
from models.statistical import MajorityClassBaselineModel, StatisticalBaselineModel, UniformRandomBaselineModel

logger = get_logger(__name__)

PROD_MODELS_DIR = Path(__file__).resolve().parent.parent / "models_store"


class ModelRegistry:
    """Central repository for model artifacts, version lineage, and database registration."""

    def __init__(
        self,
        base_dir: str | Path | None = None,
        session_factory: Any | None = None,
    ) -> None:
        env_dir = os.environ.get("MODEL_STORAGE_DIR")
        default_dir = env_dir or getattr(settings, "MODEL_STORAGE_DIR", PROD_MODELS_DIR)
        self.base_dir = Path(base_dir) if base_dir else Path(default_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self._session_factory = session_factory or get_db_session

    def _get_model_dir(self, model_id: str, version: str) -> Path:
        """Get canonical directory for a specific model version."""
        model_dir = self.base_dir / model_id / version
        model_dir.mkdir(parents=True, exist_ok=True)
        return model_dir

    def save_model(self, model: BasePredictorModel) -> Path:
        """Persist model artifacts to disk and compute hash lineage."""
        if os.environ.get("PYTEST_CURRENT_TEST") and not os.environ.get("ALLOW_PRODUCTION_FILE_MUTATION_IN_TEST"):
            try:
                if self.base_dir.resolve() == PROD_MODELS_DIR.resolve():
                    raise PermissionError(
                        f"TEST ISOLATION VIOLATION: Refusing to save model into production directory '{self.base_dir}' during test execution. "
                        "Tests must use isolated temporary model stores."
                    )
            except Exception as exc:
                if isinstance(exc, PermissionError):
                    raise

        target_dir = self._get_model_dir(model.metadata.model_id, model.metadata.version)
        model.save(target_dir)
        logger.info(
            "Saved model artifacts to registry",
            model_id=model.metadata.model_id,
            version=model.metadata.version,
            hash=model.metadata.sha256_hash,
            path=str(target_dir),
        )
        return target_dir

    def load_model(self, model_id: str, version: str, model_type: str = "LightGBMPredictor") -> BasePredictorModel:
        """Load a model from the registry."""
        target_dir = self._get_model_dir(model_id, version)
        if model_type == "LightGBMPredictor":
            return LightGBMPredictor.load(target_dir)
        elif model_type == "StatisticalBaselineModel":
            return StatisticalBaselineModel.load(target_dir)
        elif model_type == "MajorityClassBaselineModel":
            return MajorityClassBaselineModel(model_id=model_id, version=version)
        elif model_type == "UniformRandomBaselineModel":
            return UniformRandomBaselineModel(model_id=model_id, version=version)
        else:
            raise ValueError(f"Unknown model type: {model_type}")

    async def register_model_in_db(
        self,
        model: BasePredictorModel,
        status: str = "CANDIDATE",
        description: str = "",
    ) -> int | None:
        """Record model metadata and lineage in SQL database."""
        meta = model.metadata
        model_dir = self._get_model_dir(meta.model_id, meta.version)

        try:
            async with self._session_factory() as session:
                record = ModelRegistryModel(
                    model_id=meta.model_id,
                    model_name=meta.model_name,
                    version=meta.version,
                    algorithm=meta.model_type,
                    status=status,
                    hyperparameters=meta.parameters,
                    training_period={
                        "trained_at": meta.trained_at.isoformat(),
                        "sha256_hash": meta.sha256_hash or "",
                        "description": description,
                        "artifact_uri": str(model_dir),
                    },
                )
                session.add(record)
                await session.commit()
                logger.info("Registered model in database catalog", model_id=meta.model_id, status=status)
                return record.id
        except Exception as exc:
            logger.error("Failed to register model in database", error=str(exc))
            return None
