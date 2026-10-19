"""Trading Intelligence Platform - Historical Parquet Storage Engine.

High-throughput, columnar storage engine for market candles using Apache Arrow & Parquet.
Enforces deduplication, chronological sorting, and cryptographic hash verification.
"""

import hashlib
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from core.config import settings
from core.events import Candle
from core.logging import get_logger

logger = get_logger(__name__)

CANDLE_PYARROW_SCHEMA = pa.schema([
    pa.field("symbol", pa.string(), nullable=False),
    pa.field("timeframe", pa.string(), nullable=False),
    pa.field("timestamp", pa.timestamp("us", tz="UTC"), nullable=False),
    pa.field("open", pa.float64(), nullable=False),
    pa.field("high", pa.float64(), nullable=False),
    pa.field("low", pa.float64(), nullable=False),
    pa.field("close", pa.float64(), nullable=False),
    pa.field("volume", pa.float64(), nullable=False),
    pa.field("quote_volume", pa.float64(), nullable=False),
    pa.field("trade_count", pa.int64(), nullable=False),
    pa.field("vwap", pa.float64(), nullable=False),
    pa.field("is_closed", pa.bool_(), nullable=False),
])


class ParquetCandleStore:
    """Manages reading and writing immutable historical market candle partitions."""

    def __init__(self, base_dir: str | Path | None = None) -> None:
        self.base_dir = Path(base_dir) if base_dir else Path(settings.HISTORICAL_DATA_DIR)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def _get_partition_path(self, symbol: str, timeframe: str) -> Path:
        """Get canonical directory path for a symbol-timeframe partition."""
        # Sanitize symbol e.g. BTC-USD -> BTC_USD
        clean_symbol = symbol.replace("/", "_").replace("-", "_").upper()
        clean_tf = timeframe.lower().strip()
        partition_dir = self.base_dir / clean_symbol / clean_tf
        partition_dir.mkdir(parents=True, exist_ok=True)
        return partition_dir / "candles.parquet"

    def write_candles(
        self,
        symbol: str,
        timeframe: str,
        data: list[Candle] | pd.DataFrame,
    ) -> Path:
        """Persist candles into Parquet partition with automated deduplication and sorting.

        If partition already exists, merges new data with existing data, deduplicating on timestamp.
        Returns the absolute Path of the written Parquet file.
        """
        clean_symbol = symbol.replace("/", "_").replace("-", "_").upper()
        clean_tf = timeframe.lower().strip()
        target_file = self._get_partition_path(symbol, timeframe)

        if isinstance(data, list):
            if not data:
                return target_file
            records = [
                {
                    "symbol": c.symbol,
                    "timeframe": c.timeframe,
                    "timestamp": pd.to_datetime(c.timestamp, utc=True),
                    "open": float(c.open),
                    "high": float(c.high),
                    "low": float(c.low),
                    "close": float(c.close),
                    "volume": float(c.volume),
                    "quote_volume": float(c.quote_volume),
                    "trade_count": int(c.trade_count),
                    "vwap": float(c.vwap),
                    "is_closed": bool(c.is_closed),
                }
                for c in data
            ]
            new_df = pd.DataFrame(records)
        elif isinstance(data, pd.DataFrame):
            if data.empty:
                return target_file
            new_df = data.copy()
            if "timestamp" not in new_df.columns and isinstance(new_df.index, pd.DatetimeIndex):
                new_df["timestamp"] = new_df.index
            new_df["timestamp"] = pd.to_datetime(new_df["timestamp"], utc=True)
            if "symbol" not in new_df.columns:
                new_df["symbol"] = symbol
            if "timeframe" not in new_df.columns:
                new_df["timeframe"] = timeframe
            if "quote_volume" not in new_df.columns:
                new_df["quote_volume"] = new_df["close"] * new_df["volume"]
            if "trade_count" not in new_df.columns:
                new_df["trade_count"] = 0
            if "vwap" not in new_df.columns:
                new_df["vwap"] = new_df["close"]
            if "is_closed" not in new_df.columns:
                new_df["is_closed"] = True
        else:
            raise TypeError("Data must be either a list of Candle objects or a pandas DataFrame.")

        # Read existing file if present to merge & deduplicate
        if target_file.exists():
            existing_table = pq.read_table(target_file)
            existing_df = existing_table.to_pandas()
            combined_df = pd.concat([existing_df, new_df], ignore_index=True)
        else:
            combined_df = new_df

        # Deduplicate and sort chronologically
        combined_df = combined_df.drop_duplicates(subset=["timestamp"], keep="last")
        combined_df = combined_df.sort_values(by="timestamp").reset_index(drop=True)

        # Write Parquet with snappy compression and PyArrow schema
        table = pa.Table.from_pandas(combined_df, schema=CANDLE_PYARROW_SCHEMA, preserve_index=False)
        pq.write_table(table, target_file, compression="snappy")

        logger.info(
            "Wrote Parquet candle partition",
            symbol=symbol,
            timeframe=timeframe,
            row_count=len(combined_df),
            path=str(target_file),
        )
        return target_file

    def read_candles(
        self,
        symbol: str,
        timeframe: str,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
    ) -> pd.DataFrame:
        """Read candle DataFrame from Parquet partition with optional datetime filtering."""
        target_file = self._get_partition_path(symbol, timeframe)
        if not target_file.exists():
            return pd.DataFrame(columns=[
                "symbol", "timeframe", "timestamp", "open", "high", "low",
                "close", "volume", "quote_volume", "trade_count", "vwap", "is_closed"
            ])

        filters = []
        if start_time:
            utc_start = start_time.astimezone(UTC) if start_time.tzinfo else start_time.replace(tzinfo=UTC)
            filters.append(("timestamp", ">=", pd.Timestamp(utc_start)))
        if end_time:
            utc_end = end_time.astimezone(UTC) if end_time.tzinfo else end_time.replace(tzinfo=UTC)
            filters.append(("timestamp", "<=", pd.Timestamp(utc_end)))

        table = pq.read_table(target_file, filters=filters if filters else None)
        df = table.to_pandas()
        if not df.empty:
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
            df = df.sort_values(by="timestamp").reset_index(drop=True)
        return df

    def read_candles_as_objects(
        self,
        symbol: str,
        timeframe: str,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
    ) -> list[Candle]:
        """Read Parquet partition directly into a list of canonical Candle models."""
        df = self.read_candles(symbol, timeframe, start_time, end_time)
        candles: list[Candle] = []
        for _, row in df.iterrows():
            candles.append(
                Candle(
                    symbol=row["symbol"],
                    timeframe=row["timeframe"],
                    timestamp=row["timestamp"].to_pydatetime(),
                    open=float(row["open"]),
                    high=float(row["high"]),
                    low=float(row["low"]),
                    close=float(row["close"]),
                    volume=float(row["volume"]),
                    quote_volume=float(row["quote_volume"]),
                    trade_count=int(row["trade_count"]),
                    vwap=float(row["vwap"]),
                    is_closed=bool(row["is_closed"]),
                )
            )
        return candles

    def compute_dataset_hash(self, symbol: str, timeframe: str) -> str | None:
        """Compute SHA-256 checksum of Parquet partition for data lineage tracking."""
        target_file = self._get_partition_path(symbol, timeframe)
        if not target_file.exists():
            return None

        sha256 = hashlib.sha256()
        with open(target_file, "rb") as f:
            while chunk := f.read(65536):
                sha256.update(chunk)
        return sha256.hexdigest()
