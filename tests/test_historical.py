"""Tests for Historical Data Parquet Store, Importer, and Manager."""

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pandas as pd

from core.events import Candle
from data.historical.importer import HistoricalDataImporter
from data.historical.manager import HistoricalDataManager
from data.historical.parquet_store import ParquetCandleStore


def create_test_candles(count: int = 50) -> list[Candle]:
    base_time = datetime(2026, 10, 4, 10, 0, 0, tzinfo=UTC)
    candles = []
    for i in range(count):
        ts = base_time + timedelta(minutes=i)
        candles.append(
            Candle(
                symbol="BTC-USD",
                timeframe="1m",
                timestamp=ts,
                open=60000.0 + i,
                high=60010.0 + i,
                low=59990.0 + i,
                close=60005.0 + i,
                volume=1.5,
                quote_volume=(60005.0 + i) * 1.5,
                trade_count=10,
                vwap=60005.0 + i,
                is_closed=True,
            )
        )
    return candles


def test_parquet_store_read_write(tmp_path: Path):
    store = ParquetCandleStore(base_dir=tmp_path)
    candles = create_test_candles(30)

    # 1. Write candles
    file_path = store.write_candles("BTC-USD", "1m", candles)
    assert file_path.exists()

    # 2. Read as DataFrame
    df = store.read_candles("BTC-USD", "1m")
    assert len(df) == 30
    assert "open" in df.columns
    assert "close" in df.columns
    assert df["open"].iloc[0] == 60000.0

    # 3. Read with time filter
    start_filter = datetime(2026, 10, 4, 10, 10, 0, tzinfo=UTC)
    filtered_df = store.read_candles("BTC-USD", "1m", start_time=start_filter)
    assert len(filtered_df) == 20

    # 4. Read as Candle objects
    candle_objs = store.read_candles_as_objects("BTC-USD", "1m")
    assert len(candle_objs) == 30
    assert isinstance(candle_objs[0], Candle)

    # 5. Check hash
    h1 = store.compute_dataset_hash("BTC-USD", "1m")
    assert h1 is not None and len(h1) == 64


def test_parquet_store_deduplication(tmp_path: Path):
    store = ParquetCandleStore(base_dir=tmp_path)
    candles = create_test_candles(20)

    # Write initial 20
    store.write_candles("ETH-USD", "5m", candles)
    assert len(store.read_candles("ETH-USD", "5m")) == 20

    # Overlapping write: 10 old + 10 new
    base_time = datetime(2026, 10, 4, 10, 10, 0, tzinfo=UTC)
    more_candles = []
    for i in range(20):
        ts = base_time + timedelta(minutes=i)
        more_candles.append(
            Candle(
                symbol="ETH-USD",
                timeframe="5m",
                timestamp=ts,
                open=3000.0 + i,
                high=3010.0 + i,
                low=2990.0 + i,
                close=3005.0 + i,
                volume=2.0,
                quote_volume=6000.0,
                trade_count=5,
                vwap=3000.0,
                is_closed=True,
            )
        )

    store.write_candles("ETH-USD", "5m", more_candles)
    # Total unique timestamps: 0..19 + 10..29 = 30 timestamps
    combined_df = store.read_candles("ETH-USD", "5m")
    assert len(combined_df) == 30


def test_historical_importer_csv(tmp_path: Path):
    csv_file = tmp_path / "raw_btc.csv"
    data = """time,o,h,l,c,vol
2026-10-04 10:00:00,60000.0,60100.0,59900.0,60050.0,10.5
2026-10-04 10:01:00,60050.0,60120.0,60010.0,60080.0,8.2
"""
    csv_file.write_text(data)

    candles = HistoricalDataImporter.import_csv(csv_file, symbol="BTC-USD", timeframe="1m")
    assert len(candles) == 2
    assert candles[0].symbol == "BTC-USD"
    assert candles[0].open == 60000.0
    assert candles[1].close == 60080.0
    assert candles[0].volume == 10.5


def test_chronological_split():
    # 100 rows
    dates = pd.date_range("2026-01-01", periods=100, freq="1h", tz="UTC")
    df = pd.DataFrame({
        "timestamp": dates,
        "close": range(100),
    })

    train_df, val_df, test_df = HistoricalDataManager.chronological_split(
        df, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15, embargo_bars=2
    )

    # Train: 0 to 70
    assert len(train_df) == 70
    assert train_df["close"].iloc[0] == 0
    assert train_df["close"].iloc[-1] == 69

    # Embargo of 2 bars dropped: rows 70 and 71 dropped
    assert val_df["close"].iloc[0] == 72
    # Val ends at row 85: 72 to 84 (13 rows)
    assert len(val_df) == 13

    # Embargo of 2 bars dropped: rows 85 and 86 dropped
    assert test_df["close"].iloc[0] == 87
    assert test_df["close"].iloc[-1] == 99
    assert len(test_df) == 13
