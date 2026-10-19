"""Trading Intelligence Platform - Historical Data Manager.

Orchestrates historical dataset acquisition, Parquet storage, dataset versioning,
data integrity verification, and chronological walk-forward splits without lookahead bias.
"""

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from core.database import get_db_session
from core.events import Candle
from core.logging import get_logger
from data.historical.parquet_store import ParquetCandleStore
from data.storage.models import DatasetModel

logger = get_logger(__name__)


class HistoricalDataManager:
    """Central orchestrator for historical dataset persistence, lineage, and ML dataset preparation."""

    def __init__(
        self,
        store: ParquetCandleStore | None = None,
        session_factory: Any | None = None,
    ) -> None:
        self.store = store or ParquetCandleStore()
        self._session_factory = session_factory or get_db_session

    def save_dataset(
        self,
        symbol: str,
        timeframe: str,
        candles_or_df: list[Candle] | pd.DataFrame,
    ) -> Path:
        """Store historical candles in the partitioned Parquet store."""
        return self.store.write_candles(symbol, timeframe, candles_or_df)

    def load_dataset(
        self,
        symbol: str,
        timeframe: str,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
    ) -> pd.DataFrame:
        """Load historical candles as a pandas DataFrame."""
        return self.store.read_candles(symbol, timeframe, start_time, end_time)

    def load_candles(
        self,
        symbol: str,
        timeframe: str,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
    ) -> list[Candle]:
        """Load historical candles as a list of canonical Candle models."""
        return self.store.read_candles_as_objects(symbol, timeframe, start_time, end_time)

    async def register_dataset_metadata(
        self,
        dataset_name: str,
        symbol: str,
        timeframe: str,
        version: str = "v1.0.0",
        description: str = "",
    ) -> int | None:
        """Record dataset metadata and cryptographic checksum in the SQL audit database."""
        df = self.load_dataset(symbol, timeframe)
        if df.empty:
            logger.warning("Cannot register empty dataset", symbol=symbol, timeframe=timeframe)
            return None

        sha256 = self.store.compute_dataset_hash(symbol, timeframe)
        min_ts = df["timestamp"].min().to_pydatetime()
        max_ts = df["timestamp"].max().to_pydatetime()
        parquet_path = str(self.store._get_partition_path(symbol, timeframe))

        try:
            async with self._session_factory() as session:
                record = DatasetModel(
                    name=dataset_name,
                    version=version,
                    symbol=symbol,
                    timeframe=timeframe,
                    start_time=min_ts,
                    end_time=max_ts,
                    row_count=len(df),
                    storage_path=parquet_path,
                    sha256_hash=sha256 or "",
                    is_immutable=True,
                    metadata_json={
                        "description": description,
                        "columns": list(df.columns),
                        "registered_at": datetime.now(UTC).isoformat(),
                    },
                )
                session.add(record)
                await session.commit()
                logger.info(
                    "Registered historical dataset lineage",
                    name=dataset_name,
                    symbol=symbol,
                    timeframe=timeframe,
                    version=version,
                    rows=len(df),
                    hash=sha256,
                )
                return record.id
        except Exception as exc:
            logger.error("Failed to register dataset in database", error=str(exc))
            return None

    @staticmethod
    def chronological_split(
        df: pd.DataFrame,
        train_ratio: float = 0.70,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        embargo_bars: int = 0,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Split a time-series dataset strictly chronologically with zero lookahead leakage.

        Args:
            df: Chronologically sorted DataFrame.
            train_ratio: Proportion of data allocated to training.
            val_ratio: Proportion of data allocated to validation.
            test_ratio: Proportion of data allocated to testing / out-of-sample.
            embargo_bars: Number of bars dropped between splits to prevent autocorrelation
                          leakage (e.g., from rolling features or target horizons).

        Returns:
            Tuple of (train_df, val_df, test_df)
        """
        if df.empty:
            return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

        total_rows = len(df)
        if total_rows < 10:
            raise ValueError(f"Dataset has too few rows ({total_rows}) for chronological splitting.")

        norm_sum = train_ratio + val_ratio + test_ratio
        t_rat = train_ratio / norm_sum
        v_rat = val_ratio / norm_sum

        train_end_idx = int(total_rows * t_rat)
        val_start_idx = train_end_idx + embargo_bars
        val_end_idx = int(total_rows * (t_rat + v_rat))
        test_start_idx = val_end_idx + embargo_bars

        if val_start_idx >= total_rows or test_start_idx >= total_rows:
            raise ValueError("Embargo bars too large for dataset size.")

        train_df = df.iloc[:train_end_idx].copy().reset_index(drop=True)
        val_df = df.iloc[val_start_idx:val_end_idx].copy().reset_index(drop=True)
        test_df = df.iloc[test_start_idx:].copy().reset_index(drop=True)

        return train_df, val_df, test_df
