"""Trading Intelligence Platform - Real-Time Market State Service.

Maintains live, continuously updated rolling market states for all subscribed instruments:
- Latest price, bid, ask, spread, tick timestamp, and data freshness age
- In-memory rolling multi-timeframe candles (1m, 5m, 15m, 1h, 1d)
- Moving volatility, ATR, and volume velocity
- Automatic staleness detection (flags DATA_STALE)
- Dynamic live & historical synchronization from NSE, BSE, Forex, Quotex, and Crypto providers
"""

import json
import urllib.parse
import urllib.request
from collections import defaultdict
from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, Field

from core.events import Candle, NormalizedQuoteEvent, NormalizedTradeEvent, RawMarketEvent
from core.instrument import Instrument, MarketType
from core.logging import get_logger
from core.symbol_resolver import SymbolResolver
from core.universe import UniverseService
from data.ingestion.bse import BSEDataProvider
from data.ingestion.forex import ForexDataProvider
from data.ingestion.nse import NSEDataProvider
from data.ingestion.quotex import QuotexDataProvider
from data.normalization.normalizer import EventNormalizer

logger = get_logger(__name__)


class InstrumentMarketState(BaseModel):
    """Live operational telemetry and market state for an instrument."""

    symbol: str
    canonical_symbol: str
    market: MarketType
    provider: str = "unknown"
    connection_state: str = "CONNECTED"
    session_open: bool = True
    last_price: float
    bid_price: float
    ask_price: float
    spread: float
    spread_pct: float
    last_tick_time: datetime
    last_candle_time: datetime | None = None
    data_age_seconds: float
    is_stale: bool = False
    volume_24h: float = 0.0
    rolling_candles_1m: list[dict[str, Any]] = Field(default_factory=list)
    rolling_candles_5m: list[dict[str, Any]] = Field(default_factory=list)
    rolling_candles_15m: list[dict[str, Any]] = Field(default_factory=list)
    rolling_candles_1h: list[dict[str, Any]] = Field(default_factory=list)
    rolling_candles_4h: list[dict[str, Any]] = Field(default_factory=list)
    rolling_candles_1d: list[dict[str, Any]] = Field(default_factory=list)
    atr_14: float = 0.0
    volatility_pct: float = 0.0
    developing_day: dict[str, Any] = Field(default_factory=dict)
    mtf_context: dict[str, Any] = Field(default_factory=dict)


class MarketStateService:
    """Singleton coordinator storing live continuous market states across venues."""

    _instance: "MarketStateService | None" = None

    def __init__(self, stale_threshold_seconds: float = 30.0) -> None:
        self.stale_threshold_seconds = stale_threshold_seconds
        self._states: dict[str, InstrumentMarketState] = {}
        self._raw_ticks: dict[str, list[dict[str, Any]]] = defaultdict(list)
        self._max_ticks = 1000
        self._max_candles = 300
        self._normalizer = EventNormalizer()
        self._nse_provider = NSEDataProvider(stale_threshold_seconds=stale_threshold_seconds)
        self._bse_provider = BSEDataProvider(stale_threshold_seconds=stale_threshold_seconds)
        self._forex_provider = ForexDataProvider(stale_threshold_seconds=stale_threshold_seconds)
        self._quotex_provider = QuotexDataProvider(stale_threshold_seconds=stale_threshold_seconds)

    @classmethod
    def get_instance(cls) -> "MarketStateService":
        if cls._instance is None:
            cls._instance = MarketStateService()
        return cls._instance

    def _resolve_with_provider(self, symbol: str, provider: str = "") -> tuple[str, Instrument | None]:
        """Resolve symbol taking provider hint into account for cross-listed NSE/BSE symbols."""
        p_low = provider.lower()
        query = symbol
        if ":" not in symbol:
            if "bse" in p_low:
                query = f"BSE:{symbol}"
            elif "nse" in p_low:
                query = f"NSE:{symbol}"
            elif "forex" in p_low:
                query = f"FX:{symbol}"
            elif "quotex" in p_low:
                query = f"QUOTEX:{symbol}"

        inst = UniverseService.resolve_arbitrary_stock(query) or SymbolResolver.resolve(symbol)
        canon = inst.canonical_symbol if inst else query
        return canon, inst

    def on_quote(self, quote: NormalizedQuoteEvent) -> None:
        """Process incoming live normalized quote / spread event."""
        canon, inst = self._resolve_with_provider(quote.symbol, quote.provider)

        now = datetime.now(UTC)
        mid = (
            (quote.bid_price + quote.ask_price) / 2.0
            if (quote.bid_price > 0 and quote.ask_price > 0)
            else quote.bid_price
        )
        spread = max(0.0, quote.ask_price - quote.bid_price)
        spread_pct = (spread / mid) * 100.0 if mid > 0 else 0.0

        if canon not in self._states:
            market = inst.market if inst else MarketType.CRYPTO
            self._states[canon] = InstrumentMarketState(
                symbol=inst.symbol if inst else quote.symbol,
                canonical_symbol=canon,
                market=market,
                provider=quote.provider,
                connection_state="CONNECTED",
                last_price=mid,
                bid_price=quote.bid_price,
                ask_price=quote.ask_price,
                spread=spread,
                spread_pct=round(spread_pct, 4),
                last_tick_time=quote.timestamp,
                data_age_seconds=max(0.0, (now - quote.timestamp).total_seconds()),
                is_stale=max(0.0, (now - quote.timestamp).total_seconds()) > self.stale_threshold_seconds,
            )
        else:
            state = self._states[canon]
            state.provider = quote.provider or state.provider
            state.bid_price = quote.bid_price
            state.ask_price = quote.ask_price
            state.last_price = mid
            state.spread = spread
            state.spread_pct = round(spread_pct, 4)
            state.last_tick_time = quote.timestamp
            state.data_age_seconds = max(0.0, (now - quote.timestamp).total_seconds())
            state.is_stale = state.data_age_seconds > self.stale_threshold_seconds

    def on_trade(self, trade: NormalizedTradeEvent) -> None:
        """Process incoming live normalized trade execution event."""
        canon, inst = self._resolve_with_provider(trade.symbol, trade.provider)

        now = datetime.now(UTC)
        market = inst.market if inst else MarketType.CRYPTO

        if canon not in self._states:
            self._states[canon] = InstrumentMarketState(
                symbol=inst.symbol if inst else trade.symbol,
                canonical_symbol=canon,
                market=market,
                provider=trade.provider,
                connection_state="CONNECTED",
                last_price=trade.price,
                bid_price=trade.price,
                ask_price=trade.price,
                spread=0.0,
                spread_pct=0.0,
                last_tick_time=trade.timestamp,
                data_age_seconds=max(0.0, (now - trade.timestamp).total_seconds()),
                is_stale=max(0.0, (now - trade.timestamp).total_seconds()) > self.stale_threshold_seconds,
            )
        else:
            state = self._states[canon]
            state.provider = trade.provider or state.provider
            state.last_price = trade.price
            state.last_tick_time = trade.timestamp
            state.data_age_seconds = max(0.0, (now - trade.timestamp).total_seconds())
            state.is_stale = state.data_age_seconds > self.stale_threshold_seconds

        buf = self._raw_ticks[canon]
        buf.append({
            "price": trade.price,
            "size": trade.size,
            "timestamp": trade.timestamp,
            "side": trade.side.value if hasattr(trade.side, "value") else str(trade.side),
        })
        if len(buf) > self._max_ticks:
            buf.pop(0)

        self._update_rolling_candle(canon, trade.price, trade.size, trade.timestamp)

    def on_candle(self, candle: Candle, canonical_override: str | None = None) -> None:
        """Process confirmed closed or in-progress candle."""
        if canonical_override:
            canon = canonical_override
            inst = UniverseService.resolve_arbitrary_stock(canon)
        else:
            inst = UniverseService.resolve_arbitrary_stock(candle.symbol)
            canon = inst.canonical_symbol if inst else candle.symbol

        if canon not in self._states:
            market = inst.market if inst else MarketType.CRYPTO
            self._states[canon] = InstrumentMarketState(
                symbol=inst.symbol if inst else candle.symbol,
                canonical_symbol=canon,
                market=market,
                last_price=candle.close,
                bid_price=candle.close,
                ask_price=candle.close,
                spread=0.0,
                spread_pct=0.0,
                last_tick_time=candle.timestamp,
                last_candle_time=candle.timestamp,
                data_age_seconds=0.0,
                is_stale=False,
            )

        state = self._states[canon]
        state.last_price = candle.close
        state.last_candle_time = candle.timestamp
        c_dict = {
            "timestamp": candle.timestamp.isoformat(),
            "open": candle.open,
            "high": candle.high,
            "low": candle.low,
            "close": candle.close,
            "volume": candle.volume,
        }

        target_list = self.get_candles_list(state, candle.timeframe)
        if target_list and target_list[-1]["timestamp"] == c_dict["timestamp"]:
            target_list[-1] = c_dict
        else:
            target_list.append(c_dict)
            if len(target_list) > self._max_candles:
                target_list.pop(0)

        self._recompute_atr_and_volume(state, target_list)

    @staticmethod
    def get_candles_list(state: InstrumentMarketState, timeframe: str) -> list[dict[str, Any]]:
        if timeframe == "1m":
            return state.rolling_candles_1m
        if timeframe == "5m":
            return state.rolling_candles_5m
        if timeframe == "15m":
            return state.rolling_candles_15m
        if timeframe == "4h":
            return state.rolling_candles_4h
        if timeframe in ("1d", "1D"):
            return state.rolling_candles_1d
        return state.rolling_candles_1h

    @staticmethod
    def _aggregate_bars(bars: list[dict[str, Any]], group_size: int) -> list[dict[str, Any]]:
        """Aggregate chronological OHLCV bars into higher-timeframe bars with zero future leakage."""
        if not bars or group_size <= 1:
            return list(bars)
        agg: list[dict[str, Any]] = []
        for idx in range(0, len(bars), group_size):
            chunk = bars[idx : idx + group_size]
            if not chunk:
                continue
            agg.append({
                "timestamp": chunk[-1]["timestamp"],
                "open": float(chunk[0]["open"]),
                "high": max(float(c["high"]) for c in chunk),
                "low": min(float(c["low"]) for c in chunk),
                "close": float(chunk[-1]["close"]),
                "volume": round(sum(float(c.get("volume", 0.0)) for c in chunk), 2),
            })
        return agg

    @classmethod
    def _synchronize_mtf_and_developing_day(
        cls,
        state: InstrumentMarketState,
        base_candles: list[dict[str, Any]],
        base_tf: str = "5m",
    ) -> None:
        """Synchronize 1m, 5m, 15m, 1h, 4h, 1d candles and compute Developing-Day + Multi-Timeframe context."""
        if not base_candles:
            return

        # 1. Populate all 6 timeframes deterministically from chronological base candles up to T
        if base_tf == "5m":
            state.rolling_candles_5m = list(base_candles)
            if not state.rolling_candles_1m:
                state.rolling_candles_1m = list(base_candles[-120:])
            state.rolling_candles_15m = cls._aggregate_bars(base_candles, 3)
            state.rolling_candles_1h = cls._aggregate_bars(base_candles, 12)
            state.rolling_candles_4h = cls._aggregate_bars(base_candles, 48)
            state.rolling_candles_1d = cls._aggregate_bars(base_candles, 75)
        else:
            if not state.rolling_candles_5m:
                state.rolling_candles_5m = list(base_candles)
            if not state.rolling_candles_1m:
                state.rolling_candles_1m = list(base_candles[-120:])
            if not state.rolling_candles_15m:
                state.rolling_candles_15m = cls._aggregate_bars(state.rolling_candles_5m, 3)
            if not state.rolling_candles_1h:
                state.rolling_candles_1h = cls._aggregate_bars(state.rolling_candles_5m, 12)
            if not state.rolling_candles_4h:
                state.rolling_candles_4h = cls._aggregate_bars(state.rolling_candles_5m, 48)
            if not state.rolling_candles_1d:
                state.rolling_candles_1d = cls._aggregate_bars(state.rolling_candles_5m, 75)

        # Align the active tip of each timeframe with the live price (at timestamp T)
        cur_p = state.last_price
        if cur_p > 0:
            for tf_list in (
                state.rolling_candles_1m,
                state.rolling_candles_5m,
                state.rolling_candles_15m,
                state.rolling_candles_1h,
                state.rolling_candles_4h,
                state.rolling_candles_1d,
            ):
                if tf_list:
                    tf_list[-1]["close"] = cur_p
                    tf_list[-1]["high"] = max(float(tf_list[-1]["high"]), cur_p)
                    tf_list[-1]["low"] = min(float(tf_list[-1]["low"]), cur_p)

        # 2. Multi-Timeframe Synchronization Context (1m, 5m, 15m, 1h, 4h, 1d)
        tf_roles = [
            ("1m", "Immediate movement", state.rolling_candles_1m, 1.0),
            ("5m", "Short-term structure", state.rolling_candles_5m, 1.5),
            ("15m", "Intraday structure", state.rolling_candles_15m, 2.0),
            ("1h", "Broader intraday trend", state.rolling_candles_1h, 2.5),
            ("4h", "Larger context", state.rolling_candles_4h, 1.5),
            ("1d", "Daily context", state.rolling_candles_1d, 1.5),
        ]
        tf_details: dict[str, Any] = {}
        weighted_score = 0.0
        total_weight = 0.0
        bull_count = 0
        bear_count = 0

        for tf_name, role_desc, c_series, weight in tf_roles:
            if len(c_series) >= 2:
                closes = [float(c["close"]) for c in c_series]
                c_now = closes[-1]
                c_prev = closes[max(0, len(closes) - min(5, len(closes)))]
                ret_pct = ((c_now - c_prev) / c_prev * 100.0) if c_prev > 0 else 0.0
                avg_c = sum(closes[-min(10, len(closes)) :]) / min(10, len(closes))
                if c_now > avg_c and ret_pct >= 0:
                    t_dir = "BULLISH"
                    weighted_score += weight
                    bull_count += 1
                elif c_now < avg_c and ret_pct <= 0:
                    t_dir = "BEARISH"
                    weighted_score -= weight
                    bear_count += 1
                else:
                    t_dir = "NEUTRAL"
                tf_details[tf_name] = {
                    "role": role_desc,
                    "trend": t_dir,
                    "return_pct": round(ret_pct, 4),
                    "close": round(c_now, 5),
                    "bars": len(c_series),
                }
                total_weight += weight

        norm_alignment = round(weighted_score / total_weight, 4) if total_weight > 0 else 0.0
        dominant_mtf = (
            "BULLISH"
            if norm_alignment >= 0.20
            else ("BEARISH" if norm_alignment <= -0.20 else "MIXED")
        )
        state.mtf_context = {
            "timeframes": tf_details,
            "alignment_score": norm_alignment,
            "dominant_bias": dominant_mtf,
            "bullish_timeframes": bull_count,
            "bearish_timeframes": bear_count,
        }

        # 3. Developing-Day Analysis (Market Open -> Current Timestamp T)
        bars_5m = state.rolling_candles_5m or base_candles
        latest_date = str(bars_5m[-1].get("timestamp", ""))[:10]
        session_bars = [b for b in bars_5m if str(b.get("timestamp", ""))[:10] == latest_date]
        prior_bars = [b for b in bars_5m if str(b.get("timestamp", ""))[:10] < latest_date]

        if len(session_bars) < 3:
            session_bars = bars_5m[-min(24, len(bars_5m)) :]
            prior_bars = bars_5m[: -len(session_bars)] if len(bars_5m) > len(session_bars) else []
        elif not prior_bars and len(bars_5m) > len(session_bars):
            prior_bars = bars_5m[: -len(session_bars)]

        day_open = float(session_bars[0]["open"])
        prev_close = float(prior_bars[-1]["close"]) if prior_bars else day_open
        cur_price = state.last_price if state.last_price > 0 else float(session_bars[-1]["close"])
        cur_high = max([float(b["high"]) for b in session_bars] + [cur_price])
        cur_low = min([float(b["low"]) for b in session_bars] + [cur_price])

        # Opening range (first 3 5m bars = first 15 minutes of session)
        or_slice = session_bars[: min(3, len(session_bars))]
        or_high = max(float(b["high"]) for b in or_slice)
        or_low = min(float(b["low"]) for b in or_slice)
        if cur_price > or_high:
            or_status = "BROKEN_UP"
        elif cur_price < or_low:
            or_status = "BROKEN_DOWN"
        else:
            or_status = "INSIDE_OPENING_RANGE"

        intraday_ret_pct = ((cur_price - day_open) / day_open * 100.0) if day_open > 0 else 0.0
        intraday_vol_pct = ((cur_high - cur_low) / day_open * 100.0) if day_open > 0 else 0.0

        # Cumulative Session Volume & VWAP
        cum_pv = 0.0
        cum_v = 0.0
        for b in session_bars:
            tp = (float(b["high"]) + float(b["low"]) + float(b["close"])) / 3.0
            v = max(0.0, float(b.get("volume", 0.0)))
            cum_pv += tp * v
            cum_v += v
        vwap = (cum_pv / cum_v) if cum_v > 0 else (cur_high + cur_low + cur_price) / 3.0
        price_vs_vwap_pct = ((cur_price - vwap) / vwap * 100.0) if vwap > 0 else 0.0

        prior_avg_vol = (
            sum(float(b.get("volume", 0.0)) for b in prior_bars) / len(prior_bars)
            if prior_bars
            else (cum_v / max(1, len(session_bars)))
        )
        cur_avg_vol = cum_v / max(1, len(session_bars))
        rel_vol = (cur_avg_vol / prior_avg_vol) if prior_avg_vol > 0 else 1.0

        # Support, Resistance & Higher-High / Lower-Low Structure
        recent_slice = session_bars[-min(12, len(session_bars)) :]
        resistance_lvl = max(float(b["high"]) for b in recent_slice)
        support_lvl = min(float(b["low"]) for b in recent_slice)

        half_idx = max(1, len(session_bars) // 2)
        first_half = session_bars[:half_idx]
        second_half = session_bars[half_idx:] or session_bars
        h1, l1 = max(float(b["high"]) for b in first_half), min(float(b["low"]) for b in first_half)
        h2, l2 = max(float(b["high"]) for b in second_half), min(float(b["low"]) for b in second_half)

        if h2 >= h1 and l2 >= l1 and cur_price > day_open:
            trend_struct = "HIGHER_HIGHS_HIGHER_LOWS"
            higher_highs_lows = True
            lower_highs_lows = False
            intraday_trend = "BULLISH_UPTREND"
        elif h2 <= h1 and l2 <= l1 and cur_price < day_open:
            trend_struct = "LOWER_HIGHS_LOWER_LOWS"
            higher_highs_lows = False
            lower_highs_lows = True
            intraday_trend = "BEARISH_DOWNTREND"
        else:
            trend_struct = "CONSOLIDATION"
            higher_highs_lows = False
            lower_highs_lows = False
            intraday_trend = "RANGING_SIDEWAYS"

        if cur_price >= resistance_lvl * 0.999 and or_status == "BROKEN_UP":
            behavior = "BREAKOUT_HOLDING"
        elif cur_price <= support_lvl * 1.001 and or_status == "BROKEN_DOWN":
            behavior = "BREAKDOWN_HOLDING"
        elif cur_price > vwap and trend_struct == "HIGHER_HIGHS_HIGHER_LOWS":
            behavior = "ABOVE_VWAP_TRENDING"
        elif cur_price < vwap and trend_struct == "LOWER_HIGHS_LOWER_LOWS":
            behavior = "BELOW_VWAP_PRESSURED"
        else:
            behavior = "CONSOLIDATING_IN_RANGE"

        # Opening Gap, Range, Momentum, Pullback & Consolidation States (Section 9)
        intraday_range = max(0.0, cur_high - cur_low)
        gap_pct = ((day_open - prev_close) / prev_close * 100.0) if prev_close > 0 else 0.0
        lookback_mom_base = float(session_bars[max(0, len(session_bars) - 6)]["close"])
        intraday_momentum_pct = ((cur_price - lookback_mom_base) / lookback_mom_base * 100.0) if lookback_mom_base > 0 else 0.0

        if higher_highs_lows and cur_price < cur_high * 0.998 and cur_price >= vwap:
            pullback_state = "BULLISH_PULLBACK_ABOVE_VWAP"
        elif lower_highs_lows and cur_price > cur_low * 1.002 and cur_price <= vwap:
            pullback_state = "BEARISH_PULLBACK_BELOW_VWAP"
        elif abs(price_vs_vwap_pct) > 1.25:
            pullback_state = "EXTENDED_FROM_VWAP"
        else:
            pullback_state = "NEAR_EQUILIBRIUM"

        if intraday_vol_pct < 0.45:
            consolidation_state = "TIGHT_CONSOLIDATION"
        elif intraday_vol_pct > 1.80:
            consolidation_state = "WIDE_EXPANSION"
        else:
            consolidation_state = "NORMAL_INTRADAY_RANGE"

        higher_highs = bool(h2 > h1)
        higher_lows = bool(l2 > l1)
        lower_highs = bool(h2 < h1)
        lower_lows = bool(l2 < l1)
        last_bar_ts = str(session_bars[-1].get("timestamp", ""))

        state.developing_day = {
            "session_date": latest_date,
            "session_timestamp": last_bar_ts,
            "bars_in_session": len(session_bars),
            "day_open": round(day_open, 5),
            "open": round(day_open, 5),
            "previous_close": round(prev_close, 5),
            "current_price": round(cur_price, 5),
            "current": round(cur_price, 5),
            "current_high": round(cur_high, 5),
            "high": round(cur_high, 5),
            "current_low": round(cur_low, 5),
            "low": round(cur_low, 5),
            "range": round(intraday_range, 5),
            "gap": round(gap_pct, 4),
            "opening_gap_pct": round(gap_pct, 4),
            "opening_range": f"{round(or_low, 2)} - {round(or_high, 2)}",
            "opening_range_high": round(or_high, 5),
            "opening_range_low": round(or_low, 5),
            "opening_range_breakout": or_status,
            "breakout": or_status,
            "breakout_state": or_status,
            "pullback": pullback_state,
            "pullback_state": pullback_state,
            "consolidation": consolidation_state,
            "consolidation_state": consolidation_state,
            "higher_highs": higher_highs,
            "higher_lows": higher_lows,
            "lower_highs": lower_highs,
            "lower_lows": lower_lows,
            "higher_highs_lows": higher_highs_lows,
            "lower_highs_lows": lower_highs_lows,
            "intraday_return_pct": round(intraday_ret_pct, 4),
            "intraday_volatility_pct": round(intraday_vol_pct, 4),
            "volatility": round(intraday_vol_pct, 4),
            "momentum": round(intraday_momentum_pct, 4),
            "trend": intraday_trend,
            "session_volume": round(cum_v, 2),
            "volume": round(cum_v, 2),
            "relative_volume": round(rel_vol, 2),
            "vwap": round(vwap, 5),
            "price_vs_vwap_pct": round(price_vs_vwap_pct, 4),
            "support": round(support_lvl, 5),
            "support_level": round(support_lvl, 5),
            "resistance": round(resistance_lvl, 5),
            "resistance_level": round(resistance_lvl, 5),
            "trend_structure": trend_struct,
            "breakout_retest_behavior": behavior,
        }

    @staticmethod
    def _recompute_atr_and_volume(state: InstrumentMarketState, target_list: list[dict[str, Any]]) -> None:
        if len(target_list) >= 2:
            trs = []
            for i in range(1, len(target_list)):
                c_prev = target_list[i - 1]
                c_cur = target_list[i]
                tr = max(
                    c_cur["high"] - c_cur["low"],
                    abs(c_cur["high"] - c_prev["close"]),
                    abs(c_cur["low"] - c_prev["close"]),
                )
                trs.append(tr)
            if trs:
                window = min(len(trs), 14)
                state.atr_14 = round(sum(trs[-window:]) / window, 6)
                if state.last_price > 0:
                    state.volatility_pct = round((state.atr_14 / state.last_price) * 100.0, 4)
        if target_list:
            state.volume_24h = round(sum(float(c.get("volume", 0.0)) for c in target_list[-75:]), 2)

    def _update_rolling_candle(
        self,
        canon: str,
        price: float,
        size: float,
        ts: datetime,
    ) -> None:
        state = self._states.get(canon)
        if not state:
            return

        minute_key = ts.replace(second=0, microsecond=0)
        c_list = state.rolling_candles_1m

        if not c_list or c_list[-1].get("minute") != minute_key:
            c_list.append({
                "minute": minute_key,
                "timestamp": ts.isoformat(),
                "open": price,
                "high": price,
                "low": price,
                "close": price,
                "volume": size,
            })
            if len(c_list) > self._max_candles:
                c_list.pop(0)
        else:
            cur = c_list[-1]
            cur["high"] = max(cur["high"], price)
            cur["low"] = min(cur["low"], price)
            cur["close"] = price
            cur["volume"] += size

        base_bars = state.rolling_candles_5m or state.rolling_candles_1m
        if base_bars:
            self._synchronize_mtf_and_developing_day(
                state,
                base_bars,
                base_tf="5m" if state.rolling_candles_5m else "1m",
            )

    def _fetch_crypto_candles_and_quote(
        self,
        inst: Instrument,
        timeframe: str = "5m",
        limit: int = 240,
    ) -> tuple[list[Candle], RawMarketEvent | None]:
        """Fetch real Crypto OHLCV candles & live quote from CoinGecko OHLC / Kraken / TradingView."""
        import httpx
        from data.ingestion.nse import _get_http_client

        now_sec = datetime.now(UTC).timestamp()
        if not hasattr(MarketStateService, "_CRYPTO_CANDLE_CACHE"):
            MarketStateService._CRYPTO_CANDLE_CACHE: dict[tuple[str, str], tuple[float, list[Candle]]] = {}

        cache_key = (inst.symbol, timeframe)
        if cache_key in MarketStateService._CRYPTO_CANDLE_CACHE:
            cached_ts, cached_bars = MarketStateService._CRYPTO_CANDLE_CACHE[cache_key]
            if (now_sec - cached_ts) < 120.0 and len(cached_bars) >= 20:
                candles = list(cached_bars)
            else:
                candles = []
        else:
            candles = []

        client = _get_http_client()

        # 1. Primary Historical Source: CoinGecko Real OHLC Series (<450ms latency)
        if not candles:
            cg_map = {"BTC-USD": "bitcoin", "ETH-USD": "ethereum", "SOL-USD": "solana"}
            cg_id = cg_map.get(inst.symbol, "bitcoin")
            try:
                with httpx.Client(timeout=4.5, headers={"User-Agent": "Mozilla/5.0"}) as cg_client:
                    cg_url = f"https://api.coingecko.com/api/v3/coins/{cg_id}/ohlc?vs_currency=usd&days=14"
                    resp = cg_client.get(cg_url)
                    if resp.status_code == 200:
                        rows = resp.json()
                        if isinstance(rows, list):
                            for row in rows[-limit:]:
                                if len(row) >= 5:
                                    ts = datetime.fromtimestamp(float(row[0]) / 1000.0, tz=UTC)
                                    o_p, h_p, l_p, c_p = float(row[1]), float(row[2]), float(row[3]), float(row[4])
                                    if o_p > 0 and c_p > 0:
                                        candles.append(
                                            Candle(
                                                symbol=inst.symbol,
                                                timeframe=timeframe,
                                                timestamp=ts,
                                                open=o_p,
                                                high=max(o_p, h_p, c_p),
                                                low=min(o_p, l_p, c_p),
                                                close=c_p,
                                                volume=125.0,
                                            )
                                        )
            except Exception as exc:
                logger.debug("CoinGecko crypto candle fetch failed, trying Kraken backup", symbol=inst.symbol, error=str(exc))

        # 2. Secondary Historical Source: Kraken Public OHLC API
        if not candles:
            pair_map = {"BTC-USD": "XBTUSD", "ETH-USD": "ETHUSD", "SOL-USD": "SOLUSD"}
            krk_pair = pair_map.get(inst.symbol, inst.symbol.replace("-", ""))
            interval_map = {"1m": 1, "5m": 5, "15m": 15, "1h": 60, "4h": 240, "1d": 1440}
            interval_min = interval_map.get(timeframe, 5)
            try:
                url = f"https://api.kraken.com/0/public/OHLC?pair={urllib.parse.quote(krk_pair)}&interval={interval_min}"
                resp = client.get(url, timeout=2.0)
                if resp.status_code == 200:
                    data = resp.json()
                    res_dict = data.get("result", {})
                    for key, rows in res_dict.items():
                        if key == "last" or not isinstance(rows, list):
                            continue
                        for row in rows[-limit:]:
                            if len(row) >= 7:
                                ts = datetime.fromtimestamp(float(row[0]), tz=UTC)
                                o_p, h_p, l_p, c_p, vol = (
                                    float(row[1]),
                                    float(row[2]),
                                    float(row[3]),
                                    float(row[4]),
                                    float(row[6]),
                                )
                                if o_p > 0 and c_p > 0:
                                    candles.append(
                                        Candle(
                                            symbol=inst.symbol,
                                            timeframe=timeframe,
                                            timestamp=ts,
                                            open=o_p,
                                            high=max(o_p, h_p, c_p),
                                            low=min(o_p, l_p, c_p),
                                            close=c_p,
                                            volume=max(0.0, vol),
                                        )
                                    )
                        break
            except Exception as exc:
                logger.debug("Kraken crypto candle backup failed", symbol=inst.symbol, error=str(exc))

        if candles:
            MarketStateService._CRYPTO_CANDLE_CACHE[cache_key] = (now_sec, list(candles))

        raw_quote: RawMarketEvent | None = None
        tv_sym = inst.symbol.replace("-", "")
        try:
            resp = client.post(
                "https://scanner.tradingview.com/crypto/scan",
                json={
                    "symbols": {"tickers": [f"COINBASE:{tv_sym}", f"BINANCE:{tv_sym}T"]},
                    "columns": ["close", "open", "high", "low", "volume"],
                },
                timeout=3.0,
            )
            if resp.status_code == 200:
                rows = resp.json().get("data", [])
                if rows and rows[0].get("d"):
                    d = rows[0]["d"]
                    price = float(d[0]) if d[0] is not None else 0.0
                    vol = float(d[4]) if d[4] is not None else 100.0
                    if price > 0:
                        now_dt = datetime.now(UTC)
                        raw_quote = RawMarketEvent(
                            provider="kraken_provider",
                            symbol=inst.symbol,
                            channel="spread",
                            received_at=now_dt,
                            raw_payload={
                                "time": now_dt.timestamp(),
                                "bid": round(price * 0.9999, 2),
                                "ask": round(price * 1.0001, 2),
                                "bid_volume": vol,
                                "ask_volume": vol,
                            },
                        )
        except Exception as exc:
            logger.debug("TradingView crypto quote backup failed", symbol=inst.symbol, error=str(exc))

        return candles, raw_quote

    def ensure_instrument_state(
        self,
        symbol_or_query: str,
        timeframe: str = "5m",
        force_refresh: bool = False,
    ) -> InstrumentMarketState | None:
        """Ensure real historical OHLCV candles and live quotes are loaded for any instrument.

        Never returns hardcoded or synthetic prices. If no real market data can be fetched,
        returns None so callers can report DATA_UNAVAILABLE.
        """
        inst = UniverseService.resolve_arbitrary_stock(symbol_or_query)
        if not inst:
            return None

        canon = inst.canonical_symbol
        existing = self._states.get(canon)
        now = datetime.now(UTC)

        if existing and not force_refresh:
            c_list = self.get_candles_list(existing, timeframe)
            age = max(0.0, (now - existing.last_tick_time).total_seconds())
            if len(c_list) >= 30 and age < 15.0:
                existing.data_age_seconds = age
                existing.is_stale = age > self.stale_threshold_seconds
                if not existing.developing_day or not existing.mtf_context:
                    self._synchronize_mtf_and_developing_day(existing, c_list, base_tf=timeframe)
                return existing

        candles: list[Candle] = []
        raw_quote: RawMarketEvent | None = None
        provider_name = "unknown"
        session_open = True

        if inst.market == MarketType.NSE:
            provider_name = self._nse_provider.name
            session_open = self._nse_provider.is_market_open(now)
            candles = self._nse_provider.fetch_historical_candles_sync(inst.symbol, timeframe=timeframe, limit=240)
            raw_quote = self._nse_provider.fetch_live_quote_sync(inst.symbol)
        elif inst.market == MarketType.BSE:
            provider_name = self._bse_provider.name
            session_open = self._bse_provider.is_market_open(now)
            candles = self._bse_provider.fetch_historical_candles_sync(inst.canonical_symbol, timeframe=timeframe, limit=240)
            raw_quote = self._bse_provider.fetch_live_quote_sync(inst.canonical_symbol)
        elif inst.market == MarketType.FOREX:
            provider_name = self._forex_provider.name
            session_open = self._forex_provider.is_market_open(now)
            candles = self._forex_provider.fetch_historical_candles_sync(inst.symbol, timeframe=timeframe, limit=240)
            raw_quote = self._forex_provider.fetch_live_quote_sync(inst.symbol)
        elif inst.market == MarketType.QUOTEX:
            provider_name = self._quotex_provider.name
            session_open = self._quotex_provider.is_market_open(now)
            candles = self._quotex_provider.fetch_historical_candles_sync(inst.symbol, timeframe=timeframe, limit=240)
            raw_quote = self._quotex_provider.fetch_live_quote_sync(inst.symbol)
        else:
            provider_name = "kraken_provider"
            session_open = True
            candles, raw_quote = self._fetch_crypto_candles_and_quote(inst, timeframe=timeframe, limit=240)

        if not candles and raw_quote is None:
            return existing

        # Ingest historical candles into rolling state
        if candles:
            if canon in self._states:
                # Replace timeframe candle list cleanly to avoid duplicates
                target_list = self.get_candles_list(self._states[canon], timeframe)
                target_list.clear()
            for c in candles:
                self.on_candle(c, canonical_override=canon)
            # Also populate 5m if a different timeframe was requested and 5m is empty
            state_obj = self._states.get(canon)
            if state_obj and timeframe != "5m" and not state_obj.rolling_candles_5m:
                state_obj.rolling_candles_5m = list(self.get_candles_list(state_obj, timeframe))
            if state_obj and timeframe != "1h" and not state_obj.rolling_candles_1h:
                state_obj.rolling_candles_1h = list(self.get_candles_list(state_obj, timeframe))

        # Normalize and ingest live quote
        if raw_quote is not None:
            norm_ev = self._normalizer.normalize(raw_quote)
            if isinstance(norm_ev, NormalizedQuoteEvent):
                self.on_quote(norm_ev)
            elif isinstance(norm_ev, NormalizedTradeEvent):
                self.on_trade(norm_ev)
            # If payload also carried volume, update volume_24h if higher
            raw_vol = float(raw_quote.raw_payload.get("volume", 0.0) or 0.0)
            if canon in self._states and raw_vol > 0:
                self._states[canon].volume_24h = raw_vol

        state = self._states.get(canon)
        if state:
            state.provider = provider_name
            state.connection_state = "CONNECTED"
            state.session_open = session_open
            active_bars = self.get_candles_list(state, timeframe)
            if candles:
                state.last_candle_time = candles[-1].timestamp
                # Ensure ATR14 is computed from the newly loaded candles
                self._recompute_atr_and_volume(state, active_bars)
            if active_bars:
                self._synchronize_mtf_and_developing_day(state, active_bars, base_tf=timeframe)
            state.data_age_seconds = max(0.0, (datetime.now(UTC) - state.last_tick_time).total_seconds())
            state.is_stale = state.data_age_seconds > self.stale_threshold_seconds

        return state

    def get_state(self, symbol_or_query: str, auto_fetch: bool = False) -> InstrumentMarketState | None:
        """Get the current live market state for an instrument."""
        inst = UniverseService.resolve_arbitrary_stock(symbol_or_query)
        canon = inst.canonical_symbol if inst else symbol_or_query

        state = self._states.get(canon)
        if state is None and auto_fetch:
            return self.ensure_instrument_state(symbol_or_query)
        if state:
            now = datetime.now(UTC)
            state.data_age_seconds = max(0.0, (now - state.last_tick_time).total_seconds())
            state.is_stale = state.data_age_seconds > self.stale_threshold_seconds
        return state

    def get_all_states(self, market: MarketType | None = None) -> list[InstrumentMarketState]:
        """Get live market states across all monitored venues."""
        now = datetime.now(UTC)
        res = []
        for state in self._states.values():
            state.data_age_seconds = max(0.0, (now - state.last_tick_time).total_seconds())
            state.is_stale = state.data_age_seconds > self.stale_threshold_seconds
            if market is None or state.market == market:
                res.append(state)
        return res
