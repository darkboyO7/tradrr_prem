"""Trading Intelligence Platform - Market Regime Engine.

Detects macro and tactical market environments across three orthogonal dimensions:
1. Trend Regime (Strong Bull, Weak Bull, Ranging / Consolidation, Weak Bear, Strong Bear)
2. Volatility Regime (Compressed / Squeeze, Normal, Elevated, Extreme)
3. Volume / Liquidity Regime (Low Volume, Normal Volume, Volume Expansion, Climax / Exhaustion)
"""

from enum import Enum

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

from features.pipeline import FeaturePipeline


class TrendRegime(str, Enum):
    STRONG_BULL = "STRONG_BULL"
    WEAK_BULL = "WEAK_BULL"
    RANGING = "RANGING"
    WEAK_BEAR = "WEAK_BEAR"
    STRONG_BEAR = "STRONG_BEAR"


class VolatilityRegime(str, Enum):
    COMPRESSED = "COMPRESSED"
    NORMAL = "NORMAL"
    ELEVATED = "ELEVATED"
    EXTREME = "EXTREME"


class VolumeRegime(str, Enum):
    LOW_VOLUME = "LOW_VOLUME"
    NORMAL_VOLUME = "NORMAL_VOLUME"
    VOLUME_EXPANSION = "VOLUME_EXPANSION"
    VOLUME_CLIMAX = "VOLUME_CLIMAX"


class MarketRegimeState(BaseModel):
    """Encapsulates the complete multi-dimensional market regime at a specific point in time."""

    timestamp: str | None = None
    trend: TrendRegime = TrendRegime.RANGING
    volatility: VolatilityRegime = VolatilityRegime.NORMAL
    volume: VolumeRegime = VolumeRegime.NORMAL_VOLUME
    adx: float = Field(default=0.0, description="Trend strength indicator value")
    atr_ratio: float = Field(default=1.0, description="Short-term ATR relative to long-term baseline")
    rvol: float = Field(default=1.0, description="Relative volume compared to 20-bar SMA")
    confidence: float = Field(default=0.8, ge=0.0, le=1.0, description="Regime classification confidence")
    regime_code: str = Field(default="", description="Unique composite code e.g. STRONG_BULL_NORMAL_VOLUME")

    def __init__(self, **data) -> None:
        super().__init__(**data)
        if not self.regime_code:
            self.regime_code = f"{self.trend.value}_{self.volatility.value}_{self.volume.value}"


class MarketRegimeDetector:
    """Classifies market conditions from engineered features or raw OHLCV DataFrames."""

    def __init__(self, feature_pipeline: FeaturePipeline | None = None) -> None:
        self.pipeline = feature_pipeline or FeaturePipeline()

    def classify_bar(self, row: pd.Series) -> MarketRegimeState:
        """Classify a single feature row into a comprehensive MarketRegimeState."""
        close = float(row.get("close", 0.0))
        ema_20 = float(row.get("ema_20", close))
        ema_50 = float(row.get("ema_50", close))
        adx = float(row.get("adx_14", 15.0))
        plus_di = float(row.get("plus_di_14", 20.0))
        minus_di = float(row.get("minus_di_14", 20.0))

        # 1. Trend Classification
        if adx >= 25.0 and close > ema_20 and ema_20 > ema_50 and plus_di > minus_di:
            trend = TrendRegime.STRONG_BULL
            confidence = min(0.95, 0.70 + (adx / 100.0))
        elif close > ema_20:
            trend = TrendRegime.WEAK_BULL
            confidence = 0.75
        elif adx >= 25.0 and close < ema_20 and ema_20 < ema_50 and minus_di > plus_di:
            trend = TrendRegime.STRONG_BEAR
            confidence = min(0.95, 0.70 + (adx / 100.0))
        elif close < ema_20:
            trend = TrendRegime.WEAK_BEAR
            confidence = 0.75
        else:
            trend = TrendRegime.RANGING
            confidence = 0.65

        # 2. Volatility Classification
        vol_ratio = float(row.get("volatility_ratio", 1.0))
        bb_squeeze = int(row.get("bb_squeeze", 0))

        if bb_squeeze == 1 or vol_ratio < 0.75:
            volatility = VolatilityRegime.COMPRESSED
        elif vol_ratio < 1.30:
            volatility = VolatilityRegime.NORMAL
        elif vol_ratio < 2.0:
            volatility = VolatilityRegime.ELEVATED
        else:
            volatility = VolatilityRegime.EXTREME

        # 3. Volume Classification
        rvol = float(row.get("rvol_20", 1.0))
        if rvol < 0.75:
            volume = VolumeRegime.LOW_VOLUME
        elif rvol <= 1.60:
            volume = VolumeRegime.NORMAL_VOLUME
        elif rvol < 3.0:
            volume = VolumeRegime.VOLUME_EXPANSION
        else:
            volume = VolumeRegime.VOLUME_CLIMAX

        ts_str = str(row.get("timestamp", "")) if "timestamp" in row else None

        return MarketRegimeState(
            timestamp=ts_str,
            trend=trend,
            volatility=volatility,
            volume=volume,
            adx=round(adx, 2),
            atr_ratio=round(vol_ratio, 2),
            rvol=round(rvol, 2),
            confidence=round(confidence, 2),
        )

    def classify_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Vectorized classification of entire dataset into regime columns."""
        # Ensure features are present
        if "adx_14" not in df.columns or "ema_20" not in df.columns:
            feat_df = self.pipeline.compute_all_features(df)
        else:
            feat_df = df

        c = feat_df["close"]
        e20 = feat_df.get("ema_20", c)
        e50 = feat_df.get("ema_50", c)
        adx = feat_df.get("adx_14", pd.Series(15.0, index=feat_df.index))
        plus_di = feat_df.get("plus_di_14", pd.Series(20.0, index=feat_df.index))
        minus_di = feat_df.get("minus_di_14", pd.Series(20.0, index=feat_df.index))

        # Vectorized Trend
        is_strong_bull = (adx >= 25.0) & (c > e20) & (e20 > e50) & (plus_di > minus_di)
        is_weak_bull = (c > e20) & (~is_strong_bull)
        is_strong_bear = (adx >= 25.0) & (c < e20) & (e20 < e50) & (minus_di > plus_di)
        is_weak_bear = (c < e20) & (~is_strong_bear)

        trend_series = np.select(
            [is_strong_bull, is_weak_bull, is_strong_bear, is_weak_bear],
            [
                TrendRegime.STRONG_BULL.value,
                TrendRegime.WEAK_BULL.value,
                TrendRegime.STRONG_BEAR.value,
                TrendRegime.WEAK_BEAR.value,
            ],
            default=TrendRegime.RANGING.value,
        )

        # Vectorized Volatility
        vol_ratio = feat_df.get("volatility_ratio", pd.Series(1.0, index=feat_df.index))
        bb_squeeze = feat_df.get("bb_squeeze", pd.Series(0, index=feat_df.index))

        is_compressed = (bb_squeeze == 1) | (vol_ratio < 0.75)
        is_normal = (vol_ratio >= 0.75) & (vol_ratio < 1.30)
        is_elevated = (vol_ratio >= 1.30) & (vol_ratio < 2.0)
        is_extreme = vol_ratio >= 2.0

        vol_series = np.select(
            [is_compressed, is_normal, is_elevated, is_extreme],
            [
                VolatilityRegime.COMPRESSED.value,
                VolatilityRegime.NORMAL.value,
                VolatilityRegime.ELEVATED.value,
                VolatilityRegime.EXTREME.value,
            ],
            default=VolatilityRegime.NORMAL.value,
        )

        # Vectorized Volume
        rvol = feat_df.get("rvol_20", pd.Series(1.0, index=feat_df.index))
        vol_regime_series = np.select(
            [rvol < 0.75, (rvol >= 0.75) & (rvol <= 1.60), (rvol > 1.60) & (rvol < 3.0), rvol >= 3.0],
            [
                VolumeRegime.LOW_VOLUME.value,
                VolumeRegime.NORMAL_VOLUME.value,
                VolumeRegime.VOLUME_EXPANSION.value,
                VolumeRegime.VOLUME_CLIMAX.value,
            ],
            default=VolumeRegime.NORMAL_VOLUME.value,
        )

        res_df = feat_df.copy()
        res_df["trend_regime"] = trend_series
        res_df["volatility_regime"] = vol_series
        res_df["volume_regime"] = vol_regime_series
        res_df["regime_code"] = [
            f"{t}_{v}_{m}" for t, v, m in zip(trend_series, vol_series, vol_regime_series, strict=False)
        ]
        return res_df
