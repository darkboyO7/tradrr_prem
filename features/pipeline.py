"""Trading Intelligence Platform - Modular Feature Pipeline & Multi-Timeframe Alignment.

Coordinates feature generation across price, trend, momentum, volatility, volume,
and market structure categories. Enforces strict zero-lookahead point-in-time
merging for multi-timeframe feature sets.
"""

from datetime import timedelta

import pandas as pd

from core.logging import get_logger
from data.candle.resampler import parse_timeframe_to_minutes
from features.microstructure import compute_microstructure_features
from features.momentum import compute_momentum_features
from features.price import compute_price_features
from features.structure import compute_market_structure
from features.trend import compute_trend_features
from features.volatility import compute_volatility_features
from features.volume import compute_volume_features

logger = get_logger(__name__)

FEATURE_ENGINE_VERSION = "v1.0.0"


class FeaturePipeline:
    """Master feature calculation and multi-timeframe alignment engine."""

    def __init__(self, version: str = FEATURE_ENGINE_VERSION) -> None:
        self.version = version

    def compute_all_features(
        self,
        df: pd.DataFrame,
        include_price: bool = True,
        include_trend: bool = True,
        include_momentum: bool = True,
        include_volatility: bool = True,
        include_volume: bool = True,
        include_structure: bool = True,
        include_microstructure: bool = False,
    ) -> pd.DataFrame:
        """Compute all modular feature groups for a single-timeframe candle DataFrame.

        Expects columns: open, high, low, close, volume, and optional quote_volume, vwap.
        Returns a DataFrame with the original OHLCV plus all generated feature columns.
        """
        if df.empty:
            return pd.DataFrame()

        feature_blocks: list[pd.DataFrame] = [df]

        if include_price:
            feature_blocks.append(compute_price_features(df))
        if include_trend:
            feature_blocks.append(compute_trend_features(df))
        if include_momentum:
            feature_blocks.append(compute_momentum_features(df))
        if include_volatility:
            feature_blocks.append(compute_volatility_features(df))
        if include_volume:
            feature_blocks.append(compute_volume_features(df))
        if include_structure:
            feature_blocks.append(compute_market_structure(df, lookback=20))
        if include_microstructure or any(c in df.columns for c in ("order_book_imbalance", "bid_depth_top5", "best_bid")):
            feature_blocks.append(compute_microstructure_features(df))

        result = pd.concat(feature_blocks, axis=1)
        # Drop any duplicated column names if they occur
        result = result.loc[:, ~result.columns.duplicated()]
        return result

    def merge_multi_timeframe_features(
        self,
        ltf_df: pd.DataFrame,
        htf_feature_dict: dict[str, pd.DataFrame],
    ) -> pd.DataFrame:
        """Merge higher-timeframe (HTF) features onto a lower-timeframe (LTF) index with ZERO lookahead.

        CRITICAL LOOKAHEAD GUARD:
        A higher-timeframe bar with start timestamp `T` (e.g. 1H bar at 12:00) is only finalized
        and fully known at `T + timeframe_duration` (e.g. 13:00).
        To prevent lookahead leakage, HTF features are shifted by their timeframe duration
        so that an LTF bar at 12:30 only ever sees the HTF bar completed at 12:00 (which started at 11:00).

        Args:
            ltf_df: Lower timeframe DataFrame with 'timestamp' column or DatetimeIndex.
            htf_feature_dict: Mapping of timeframe name (e.g. '15m', '1h', '4h') to its feature DataFrame.

        Returns:
            Merged DataFrame indexed by LTF timestamps containing both LTF and prefixed HTF features.
        """
        if ltf_df.empty:
            return pd.DataFrame()

        # Ensure ltf has timestamp column
        work_ltf = ltf_df.copy()
        if "timestamp" not in work_ltf.columns and isinstance(work_ltf.index, pd.DatetimeIndex):
            work_ltf["timestamp"] = work_ltf.index
        work_ltf["timestamp"] = pd.to_datetime(work_ltf["timestamp"], utc=True)
        work_ltf = work_ltf.sort_values(by="timestamp").reset_index(drop=True)

        merged = work_ltf

        for tf_name, htf_raw in htf_feature_dict.items():
            if htf_raw.empty:
                continue

            htf = htf_raw.copy()
            if "timestamp" not in htf.columns and isinstance(htf.index, pd.DatetimeIndex):
                htf["timestamp"] = htf.index
            htf["timestamp"] = pd.to_datetime(htf["timestamp"], utc=True)

            # Drop raw OHLCV columns from HTF to avoid clutter and conflicts, keeping engineered features
            drop_cols = [c for c in ["open", "high", "low", "close", "volume", "quote_volume", "trade_count", "vwap", "is_closed", "symbol", "timeframe"] if c in htf.columns]
            htf_features_only = htf.drop(columns=drop_cols)

            # Prefix HTF columns: e.g. "htf_1h_rsi_14"
            prefix = f"htf_{tf_name.lower().strip()}_"
            htf_prefixed = htf_features_only.rename(columns={c: f"{prefix}{c}" for c in htf_features_only.columns})

            # Calculate exact completion time of the HTF bar
            tf_minutes = parse_timeframe_to_minutes(tf_name)
            htf_prefixed["available_at"] = htf["timestamp"] + timedelta(minutes=tf_minutes)
            htf_prefixed = htf_prefixed.sort_values(by="available_at").reset_index(drop=True)

            # Point-in-time backward merge (asof) on LTF timestamp >= HTF available_at
            merged = pd.merge_asof(
                merged,
                htf_prefixed,
                left_on="timestamp",
                right_on="available_at",
                direction="backward",
            )
            # Remove temporary available_at column
            if "available_at" in merged.columns:
                merged = merged.drop(columns=["available_at"])

        return merged
