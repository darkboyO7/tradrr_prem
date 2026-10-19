"""Trading Intelligence Platform - Intraday Volume Profiler.

Estimates intra-day volume distribution patterns (e.g. 24-hour bins or session boundaries)
to generate expected volume fractions for volume-weighted execution algorithms (VWAP).
"""

from datetime import datetime, time, timezone
from typing import Dict, List, Optional
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

from core.logging import get_logger

logger = get_logger("volume_profile")


class IntradayVolumeProfile(BaseModel):
    """Normalized intra-day volume curve across hourly bins."""

    symbol: str
    num_bins: int = 24
    bin_fractions: List[float] = Field(description="Normalized volume expected in each bin (sums to 1.0)")
    cumulative_fractions: List[float] = Field(description="Cumulative volume curve F(t) from 0.0 to 1.0")
    peak_bin_hour: int
    trough_bin_hour: int


class VolumeProfiler:
    """Computes empirical and prior intra-day volume distributions."""

    # Canonical 24-hour crypto volume profile prior (UTC hours: active during London/NY overlap, lower in late US)
    CANONICAL_CRYPTO_PRIOR = np.array([
        0.038, 0.035, 0.032, 0.030, 0.031, 0.033,  # 00:00 - 05:00 UTC (Asia mid/late)
        0.037, 0.042, 0.048, 0.052, 0.051, 0.049,  # 06:00 - 11:00 UTC (Europe open)
        0.055, 0.062, 0.065, 0.063, 0.058, 0.052,  # 12:00 - 17:00 UTC (US open & overlap)
        0.046, 0.042, 0.039, 0.038, 0.039, 0.039   # 18:00 - 23:00 UTC (US close / Asia open)
    ])

    def __init__(self) -> None:
        # Normalize prior to exactly 1.0
        self._prior_normalized = self.CANONICAL_CRYPTO_PRIOR / self.CANONICAL_CRYPTO_PRIOR.sum()

    def get_canonical_profile(self, symbol: str) -> IntradayVolumeProfile:
        """Return canonical prior volume profile for an asset."""
        fractions = list(self._prior_normalized)
        cum_fractions = list(np.cumsum(fractions))
        # Ensure last is exactly 1.0
        cum_fractions[-1] = 1.0

        peak_hour = int(np.argmax(fractions))
        trough_hour = int(np.argmin(fractions))

        return IntradayVolumeProfile(
            symbol=symbol.upper(),
            num_bins=24,
            bin_fractions=[round(float(f), 6) for f in fractions],
            cumulative_fractions=[round(float(c), 6) for c in cum_fractions],
            peak_bin_hour=peak_hour,
            trough_bin_hour=trough_hour,
        )

    def estimate_profile_from_history(
        self,
        symbol: str,
        hourly_candles_df: pd.DataFrame,
    ) -> IntradayVolumeProfile:
        """Compute empirical volume distribution from historical hourly candles.
        
        Args:
            symbol: Ticker symbol.
            hourly_candles_df: DataFrame with datetime index or 'timestamp' column, and 'volume' column.
        """
        df = hourly_candles_df.copy()
        if not isinstance(df.index, pd.DatetimeIndex):
            if "timestamp" in df.columns:
                df["timestamp"] = pd.to_datetime(df["timestamp"])
                df = df.set_index("timestamp")
            else:
                return self.get_canonical_profile(symbol)

        if "volume" not in df.columns or len(df) < 48:
            # Fall back to canonical prior if history is insufficient (< 2 full days)
            return self.get_canonical_profile(symbol)

        # Extract hour of day
        df["hour"] = df.index.hour
        hourly_vols = df.groupby("hour")["volume"].mean()

        # Reindex to ensure all 24 hours are present
        hourly_vols = hourly_vols.reindex(range(24), fill_value=0.0)
        tot_vol = hourly_vols.sum()

        if tot_vol <= 0:
            return self.get_canonical_profile(symbol)

        # Blend empirical volume with canonical prior (Bayesian shrinkage)
        empirical_fractions = (hourly_vols / tot_vol).values
        # 70% empirical weight, 30% prior weight
        blended = (0.70 * empirical_fractions) + (0.30 * self._prior_normalized)
        blended /= blended.sum()

        cum_fractions = list(np.cumsum(blended))
        cum_fractions[-1] = 1.0

        return IntradayVolumeProfile(
            symbol=symbol.upper(),
            num_bins=24,
            bin_fractions=[round(float(f), 6) for f in blended],
            cumulative_fractions=[round(float(c), 6) for c in cum_fractions],
            peak_bin_hour=int(np.argmax(blended)),
            trough_bin_hour=int(np.argmin(blended)),
        )

    def get_slice_fractions_for_window(
        self,
        profile: IntradayVolumeProfile,
        start_time: datetime,
        duration_hours: float,
        num_slices: int,
    ) -> List[float]:
        """Compute exact weight fractions for N execution slices over a time window."""
        if num_slices <= 0:
            raise ValueError(f"num_slices must be > 0, got {num_slices}")

        slice_dur_hours = duration_hours / num_slices
        raw_weights = []

        for i in range(num_slices):
            slice_center_hour = (start_time.hour + (i + 0.5) * slice_dur_hours) % 24.0
            bin_idx = int(slice_center_hour) % 24
            raw_weights.append(profile.bin_fractions[bin_idx])

        total = sum(raw_weights)
        if total <= 0:
            return [1.0 / num_slices] * num_slices

        normalized = [round(w / total, 6) for w in raw_weights]
        # Adjust rounding difference on last slice
        diff = round(1.0 - sum(normalized), 6)
        normalized[-1] = round(normalized[-1] + diff, 6)
        return normalized
