"""Trading Intelligence Platform - Candle aggregation & multi-timeframe package."""

from data.candle.engine import ActiveCandleState, CandleEngine, floor_to_minute
from data.candle.multi_timeframe import MultiTimeframeAggregator
from data.candle.resampler import (
    floor_time_to_timeframe,
    get_pandas_freq,
    parse_timeframe_to_minutes,
    resample_candles,
    resample_dataframe,
)

__all__ = [
    "floor_to_minute",
    "ActiveCandleState",
    "CandleEngine",
    "floor_time_to_timeframe",
    "get_pandas_freq",
    "parse_timeframe_to_minutes",
    "resample_candles",
    "resample_dataframe",
    "MultiTimeframeAggregator",
]
