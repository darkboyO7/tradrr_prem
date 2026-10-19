"""Trading Intelligence Platform - Statistical Baseline Model.

A transparent, deterministic statistical benchmark model combining trend slope,
mean-reversion thresholds, and volatility normalization.
Every ML model must strictly outperform this baseline.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from models.base import BasePredictorModel, ModelMetadata, ModelPrediction, SignalAction


class StatisticalBaselineModel(BasePredictorModel):
    """Deterministic statistical benchmark model for directional trading intelligence."""

    def __init__(
        self,
        model_id: str = "stat_baseline_v1",
        version: str = "v1.0.0",
        rsi_oversold: float = 30.0,
        rsi_overbought: float = 70.0,
        adx_trend_threshold: float = 25.0,
    ) -> None:
        metadata = ModelMetadata(
            model_id=model_id,
            model_name="StatisticalBaseline",
            version=version,
            model_type="StatisticalBaselineModel",
            feature_names=["rsi_14", "adx_14", "ema_20_slope_5", "dist_ema_20_pct"],
            target_name="fwd_direction",
            parameters={
                "rsi_oversold": rsi_oversold,
                "rsi_overbought": rsi_overbought,
                "adx_trend_threshold": adx_trend_threshold,
            },
        )
        super().__init__(metadata)
        self.rsi_oversold = rsi_oversold
        self.rsi_overbought = rsi_overbought
        self.adx_trend_threshold = adx_trend_threshold

    def fit(self, X: pd.DataFrame, y: pd.Series | np.ndarray) -> "StatisticalBaselineModel":
        """Statistical baseline is rule-based and parameter-driven; validates feature columns."""
        for feat in self.metadata.feature_names:
            if feat not in X.columns:
                # Add default if missing
                X[feat] = 0.0
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Estimate pseudo-probabilities for [P(Sell), P(NoTrade), P(Buy)]."""
        probas = []
        for _, row in X.iterrows():
            rsi = float(row.get("rsi_14", 50.0))
            adx = float(row.get("adx_14", 15.0))
            slope = float(row.get("ema_20_slope_5", 0.0))
            dist_ema = float(row.get("dist_ema_20_pct", 0.0))

            # Momentum / Trend score in [-1.0, 1.0]
            score = 0.0
            if adx >= self.adx_trend_threshold:
                # Trend following
                score += 0.5 * np.sign(slope)
                score += 0.3 * np.sign(dist_ema)
            else:
                # Mean reversion
                if rsi <= self.rsi_oversold:
                    score += 0.6
                elif rsi >= self.rsi_overbought:
                    score -= 0.6

            score = np.clip(score, -1.0, 1.0)
            if score > 0.3:
                p_buy = 0.5 + (0.4 * score)
                p_sell = 0.5 * (1.0 - p_buy)
                p_neutral = 1.0 - p_buy - p_sell
            elif score < -0.3:
                p_sell = 0.5 + (0.4 * abs(score))
                p_buy = 0.5 * (1.0 - p_sell)
                p_neutral = 1.0 - p_buy - p_sell
            else:
                p_neutral = 0.6
                p_buy = 0.2
                p_sell = 0.2

            probas.append([p_sell, p_neutral, p_buy])

        return np.array(probas)

    def predict(self, X: pd.DataFrame) -> list[ModelPrediction]:
        """Convert input feature vectors into discrete ModelPrediction instances."""
        probas = self.predict_proba(X)
        predictions = []

        for p_sell, p_neutral, p_buy in probas:
            if p_buy > p_sell and p_buy >= 0.55:
                action = SignalAction.BUY
                prob = p_buy
            elif p_sell > p_buy and p_sell >= 0.55:
                action = SignalAction.SELL
                prob = p_sell
            else:
                action = SignalAction.NO_TRADE
                prob = p_neutral

            predictions.append(
                ModelPrediction(
                    action=action,
                    probability=round(float(prob), 4),
                    expected_return=round(float((p_buy - p_sell) * 0.015), 5),
                    horizon_bars=5,
                    raw_scores={"sell": round(p_sell, 4), "no_trade": round(p_neutral, 4), "buy": round(p_buy, 4)},
                )
            )
        return predictions

    def get_feature_importance(self) -> dict[str, float]:
        """Hard-coded weights reflecting the deterministic logic."""
        return {
            "rsi_14": 0.35,
            "adx_14": 0.25,
            "ema_20_slope_5": 0.25,
            "dist_ema_20_pct": 0.15,
        }

    def save(self, directory: str | Path) -> Path:
        """Save model metadata to disk."""
        target_dir = Path(directory)
        target_dir.mkdir(parents=True, exist_ok=True)
        meta_file = target_dir / "metadata.json"
        meta_file.write_text(self.metadata.model_dump_json(indent=2))
        return target_dir

    @classmethod
    def load(cls, directory: str | Path) -> "StatisticalBaselineModel":
        """Load model configuration from disk."""
        target_dir = Path(directory)
        meta_file = target_dir / "metadata.json"
        if not meta_file.exists():
            raise FileNotFoundError(f"Model metadata not found at {meta_file}")
        meta_data = json.loads(meta_file.read_text())
        params = meta_data.get("parameters", {})
        model = cls(
            model_id=meta_data.get("model_id", "stat_baseline"),
            version=meta_data.get("version", "v1.0.0"),
            rsi_oversold=params.get("rsi_oversold", 30.0),
            rsi_overbought=params.get("rsi_overbought", 70.0),
            adx_trend_threshold=params.get("adx_trend_threshold", 25.0),
        )
        return model
