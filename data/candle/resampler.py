"""Trading Intelligence Platform - Multi-Timeframe Candle Resampler.

Converts fine-grained candles (typically 1-minute base bars) into higher-timeframe
candles (5m, 15m, 30m, 1h, 4h, 1d, etc.) with strict boundary alignment, volume aggregation,
and volume-weighted average price (VWAP) calculation.
"""

import re
from datetime import UTC, datetime

import pandas as pd

from core.events import Candle

TIMEFRAME_TO_PANDAS_FREQ = {
    "1m": "1min",
    "3m": "3min",
    "5m": "5min",
    "15m": "15min",
    "30m": "30min",
    "1h": "1h",
    "2h": "2h",
    "4h": "4h",
    "6h": "6h",
    "8h": "8h",
    "12h": "12h",
    "1d": "1D",
    "1w": "1W",
}

TIMEFRAME_TO_MINUTES = {
    "1m": 1,
    "3m": 3,
    "5m": 5,
    "15m": 15,
    "30m": 30,
    "1h": 60,
    "2h": 120,
    "4h": 240,
    "6h": 360,
    "8h": 480,
    "12h": 720,
    "1d": 1440,
    "1w": 10080,
}


def parse_timeframe_to_minutes(timeframe: str) -> int:
    """Parse timeframe string (e.g., '5m', '4h', '1d') into total minutes."""
    tf = timeframe.lower().strip()
    if tf in TIMEFRAME_TO_MINUTES:
        return TIMEFRAME_TO_MINUTES[tf]

    match = re.match(r"^(\d+)([mhdwn])$", tf)
    if not match:
        raise ValueError(f"Unsupported timeframe format: {timeframe}")

    val, unit = int(match.group(1)), match.group(2)
    if unit in ("m", "n"):
        return val
    elif unit == "h":
        return val * 60
    elif unit == "d":
        return val * 1440
    elif unit == "w":
        return val * 10080
    raise ValueError(f"Unknown timeframe unit: {unit}")


def get_pandas_freq(timeframe: str) -> str:
    """Map canonical timeframe format to pandas offset alias."""
    tf = timeframe.lower().strip()
    if tf in TIMEFRAME_TO_PANDAS_FREQ:
        return TIMEFRAME_TO_PANDAS_FREQ[tf]
    minutes = parse_timeframe_to_minutes(tf)
    return f"{minutes}min"


def floor_time_to_timeframe(dt: datetime, timeframe: str) -> datetime:
    """Floor a datetime to the boundary of the given timeframe interval in UTC."""
    minutes = parse_timeframe_to_minutes(timeframe)
    total_minutes = int(dt.timestamp() // 60)
    floored_minutes = (total_minutes // minutes) * minutes
    return datetime.fromtimestamp(floored_minutes * 60, tz=UTC)


def resample_dataframe(
    df: pd.DataFrame,
    target_timeframe: str,
    closed_only: bool = True,
) -> pd.DataFrame:
    """Resample a OHLCV DataFrame indexed by UTC datetime to a target timeframe.

    Expects DataFrame columns: open, high, low, close, volume, and optionally
    quote_volume, trade_count, vwap.
    """
    if df.empty:
        return pd.DataFrame()

    freq = get_pandas_freq(target_timeframe)

    # Ensure DatetimeIndex in UTC
    work_df = df.copy()
    if not isinstance(work_df.index, pd.DatetimeIndex):
        if "timestamp" in work_df.columns:
            work_df["timestamp"] = pd.to_datetime(work_df["timestamp"], utc=True)
            work_df = work_df.set_index("timestamp")
        else:
            raise ValueError("DataFrame must have a DatetimeIndex or 'timestamp' column.")

    work_df = work_df.sort_index()

    # Build aggregation dictionary
    agg_dict = {
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    }
    if "quote_volume" in work_df.columns:
        agg_dict["quote_volume"] = "sum"
    if "trade_count" in work_df.columns:
        agg_dict["trade_count"] = "sum"

    resampled = work_df.resample(freq, label="left", closed="left").agg(agg_dict)
    resampled = resampled.dropna(subset=["open", "high", "low", "close"])

    # Compute VWAP
    if "quote_volume" in resampled.columns and "volume" in resampled.columns:
        resampled["vwap"] = resampled.apply(
            lambda row: round(row["quote_volume"] / row["volume"], 4) if row["volume"] > 0 else row["close"],
            axis=1,
        )
    elif "vwap" not in resampled.columns:
        resampled["vwap"] = resampled["close"]

    if "trade_count" not in resampled.columns:
        resampled["trade_count"] = 0
    if "quote_volume" not in resampled.columns:
        resampled["quote_volume"] = resampled["close"] * resampled["volume"]

    resampled["is_closed"] = True
    return resampled


def resample_candles(
    candles: list[Candle],
    target_timeframe: str,
) -> list[Candle]:
    """Resample a list of chronological Candle models into target timeframe Candle models."""
    if not candles:
        return []

    symbol = candles[0].symbol
    records = []
    for c in candles:
        records.append({
            "timestamp": c.timestamp,
            "open": c.open,
            "high": c.high,
            "low": c.low,
            "close": c.close,
            "volume": c.volume,
            "quote_volume": c.quote_volume,
            "trade_count": c.trade_count,
        })

    df = pd.DataFrame(records)
    resampled_df = resample_dataframe(df, target_timeframe)

    result: list[Candle] = []
    for ts, row in resampled_df.iterrows():
        result.append(
            Candle(
                symbol=symbol,
                timeframe=target_timeframe,
                timestamp=ts.to_pydatetime() if hasattr(ts, "to_pydatetime") else ts,
                open=float(row["open"]),
                high=float(row["high"]),
                low=float(row["low"]),
                close=float(row["close"]),
                volume=round(float(row["volume"]), 8),
                quote_volume=round(float(row.get("quote_volume", 0.0)), 4),
                trade_count=int(row.get("trade_count", 0)),
                vwap=round(float(row.get("vwap", row["close"])), 4),
                is_closed=bool(row.get("is_closed", True)),
            )
        )
    return result
