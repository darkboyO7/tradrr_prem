"""Trading Intelligence Platform - Historical Data Importer.

Parses external historical CSV and JSON data files from various venues and brokers,
normalizing columns into standard canonical Candle models.
"""

from pathlib import Path

import pandas as pd

from core.events import Candle
from core.logging import get_logger

logger = get_logger(__name__)

# Common column alias mappings
COLUMN_ALIASES: dict[str, str] = {
    "date": "timestamp",
    "time": "timestamp",
    "datetime": "timestamp",
    "open_time": "timestamp",
    "timestamp_utc": "timestamp",
    "o": "open",
    "h": "high",
    "l": "low",
    "c": "close",
    "v": "volume",
    "vol": "volume",
    "base_volume": "volume",
    "quote_vol": "quote_volume",
    "quote_volume": "quote_volume",
    "trades": "trade_count",
    "num_trades": "trade_count",
    "count": "trade_count",
}


class HistoricalDataImporter:
    """Imports raw CSV or JSON candle records into canonical data formats."""

    @staticmethod
    def import_csv(
        file_path: str | Path,
        symbol: str,
        timeframe: str = "1m",
        column_mapping: dict[str, str] | None = None,
    ) -> list[Candle]:
        """Read CSV file and return sorted, validated Candle models."""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Historical file not found: {path}")

        df = pd.read_csv(path)
        return HistoricalDataImporter.normalize_dataframe(df, symbol=symbol, timeframe=timeframe, custom_mapping=column_mapping)

    @staticmethod
    def normalize_dataframe(
        df: pd.DataFrame,
        symbol: str,
        timeframe: str = "1m",
        custom_mapping: dict[str, str] | None = None,
    ) -> list[Candle]:
        """Convert arbitrary OHLCV DataFrame into canonical Candle instances."""
        work_df = df.copy()

        # Rename columns using case-insensitive mapping
        rename_map = {}
        for col in work_df.columns:
            col_lower = str(col).lower().strip()
            if custom_mapping and col in custom_mapping:
                rename_map[col] = custom_mapping[col]
            elif custom_mapping and col_lower in custom_mapping:
                rename_map[col] = custom_mapping[col_lower]
            elif col_lower in COLUMN_ALIASES:
                rename_map[col] = COLUMN_ALIASES[col_lower]
            else:
                rename_map[col] = col_lower
        work_df = work_df.rename(columns=rename_map)

        # Validate mandatory columns
        required = ["timestamp", "open", "high", "low", "close", "volume"]
        missing = [c for c in required if c not in work_df.columns]
        if missing:
            raise ValueError(f"Missing mandatory OHLCV columns in dataset: {missing}")

        # Standardize timestamps
        if pd.api.types.is_numeric_dtype(work_df["timestamp"]):
            # Check if unix seconds or milliseconds
            sample = work_df["timestamp"].dropna().iloc[0] if not work_df["timestamp"].dropna().empty else 0
            unit = "ms" if sample > 1e11 else ("s" if sample > 1e8 else "ns")
            work_df["timestamp"] = pd.to_datetime(work_df["timestamp"], unit=unit, utc=True)
        else:
            work_df["timestamp"] = pd.to_datetime(work_df["timestamp"], utc=True)

        work_df = work_df.sort_values(by="timestamp").drop_duplicates(subset=["timestamp"], keep="last")

        candles: list[Candle] = []
        for _, row in work_df.iterrows():
            o = float(row["open"])
            h = float(row["high"])
            l = float(row["low"])
            c = float(row["close"])
            v = float(row["volume"])
            qv = float(row.get("quote_volume", c * v))
            tc = int(row.get("trade_count", 0))
            vw = float(row.get("vwap", (qv / v) if v > 0 else c))

            candles.append(
                Candle(
                    symbol=symbol,
                    timeframe=timeframe,
                    timestamp=row["timestamp"].to_pydatetime(),
                    open=o,
                    high=h,
                    low=l,
                    close=c,
                    volume=v,
                    quote_volume=qv,
                    trade_count=tc,
                    vwap=vw,
                    is_closed=True,
                )
            )

        logger.info("Imported historical dataset", symbol=symbol, timeframe=timeframe, candle_count=len(candles))
        return candles
