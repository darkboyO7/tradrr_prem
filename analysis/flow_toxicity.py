"""Institutional Real-Time Order Flow Toxicity & Adverse Selection Engine.

Implements:
1. Volume-Synchronized Probability of Toxicity (VPIN) continuous estimation (Easley, López de Prado, O'Hara 2012):
   - Dynamic volume-clock binning with constant volume bucket size V_bar
   - Trade direction classification via Lee-Ready algorithm and tick rule
   - Trade order flow imbalance accumulation: VPIN = (1 / (N * V_bar)) * sum_{tau=1}^{N} |V_B^tau - V_S^tau|
2. Roll Spread Model (Richard Roll 1984):
   - Effective spread estimation from serial autocovariance of price changes:
     s = 2 * sqrt(-cov(Delta P_t, Delta P_{t-1})) if cov < 0 else 0
3. Glosten-Milgrom / Kyle Lambda Adverse Selection Parameter:
   - Price impact lambda = cov(Delta P_t, Q_t) / var(Q_t) where Q_t in {-1, +1} is trade direction
4. Real-time toxic flow classification and gating signals for market makers and SOR.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


class FlowToxicityRegime(str, Enum):
    """Regime classification for informed/toxic order flow."""
    BENIGN = "BENIGN"                     # VPIN < 0.25 (Safe for tight market making)
    MODERATE = "MODERATE"                 # 0.25 <= VPIN < 0.45 (Normal flow)
    ELEVATED_TOXICITY = "ELEVATED_TOXICITY" # 0.45 <= VPIN < 0.65 (Widen MM spreads, reduce size)
    EXTREME_ADVERSE = "EXTREME_ADVERSE"   # VPIN >= 0.65 (High adverse selection, pull quotes / pause)


@dataclass(frozen=True)
class VolumeBucket:
    """Completed volume bucket in VPIN volume clock."""
    bucket_index: int
    start_timestamp_ns: int
    end_timestamp_ns: int
    total_volume: float
    buy_volume: float
    sell_volume: float
    vwap: float
    order_imbalance: float  # |V_B - V_S|

    def to_dict(self) -> Dict[str, Any]:
        return {
            "bucket_index": int(self.bucket_index),
            "start_timestamp_ns": int(self.start_timestamp_ns),
            "end_timestamp_ns": int(self.end_timestamp_ns),
            "total_volume": float(self.total_volume),
            "buy_volume": float(self.buy_volume),
            "sell_volume": float(self.sell_volume),
            "vwap": float(self.vwap),
            "order_imbalance": float(self.order_imbalance),
        }


@dataclass(frozen=True)
class ToxicityMetricsSnapshot:
    """Comprehensive real-time toxicity and adverse selection metrics."""
    symbol: str
    timestamp_utc: datetime
    current_vpin: float
    toxicity_regime: FlowToxicityRegime
    roll_effective_spread: float
    roll_spread_bps: float
    kyle_lambda_impact: float
    informed_trader_ratio: float
    completed_buckets_count: int
    total_volume_processed: float
    recommendation: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "timestamp_utc": self.timestamp_utc.isoformat(),
            "current_vpin": float(self.current_vpin),
            "toxicity_regime": self.toxicity_regime.value,
            "roll_effective_spread": float(self.roll_effective_spread),
            "roll_spread_bps": float(self.roll_spread_bps),
            "kyle_lambda_impact": float(self.kyle_lambda_impact),
            "informed_trader_ratio": float(self.informed_trader_ratio),
            "completed_buckets_count": int(self.completed_buckets_count),
            "total_volume_processed": float(self.total_volume_processed),
            "recommendation": self.recommendation,
        }


class OrderFlowToxicityEngine:
    """Institutional Order Flow Toxicity & Adverse Selection Engine.

    Maintains volume buckets, tracks trade-by-trade signed order flow,
    calibrates VPIN and Roll effective spreads dynamically.
    """

    def __init__(
        self,
        symbol: str,
        bucket_volume: float = 10.0,       # Size of each volume bucket V_bar
        vpin_window_buckets: int = 50,     # Number of buckets N over which to compute VPIN
        price_history_length: int = 200,   # Rolling length for Roll model & Kyle's Lambda
    ) -> None:
        self.symbol = symbol.upper()
        self.bucket_volume = float(bucket_volume)
        self.vpin_window_buckets = int(vpin_window_buckets)
        self.price_history_length = int(price_history_length)

        # Volume-clock state
        self._current_bucket_idx: int = 0
        self._current_bucket_buy_vol: float = 0.0
        self._current_bucket_sell_vol: float = 0.0
        self._current_bucket_total_vol: float = 0.0
        self._current_bucket_notional: float = 0.0
        self._current_bucket_start_ts: int = 0

        # Completed buckets deque
        self._completed_buckets: deque[VolumeBucket] = deque(maxlen=self.vpin_window_buckets)

        # Microstructure trade history for Roll & Kyle Lambda
        self._price_history: deque[float] = deque(maxlen=self.price_history_length)
        self._direction_history: deque[int] = deque(maxlen=self.price_history_length)  # +1 buy, -1 sell
        self._last_trade_price: Optional[float] = None
        self._last_bbo_mid: Optional[float] = None
        self._total_volume_processed: float = 0.0

    def update_bbo(self, bid: float, ask: float) -> None:
        """Update top-of-book mid-price for Lee-Ready trade direction classification."""
        if bid > 0 and ask >= bid:
            self._last_bbo_mid = 0.5 * (bid + ask)

    def process_trade(
        self,
        price: float,
        quantity: float,
        timestamp_ns: int,
        side: Optional[str] = None,  # If provided ("BUY"/"SELL"), overrides classification
    ) -> Optional[VolumeBucket]:
        """Process incoming trade event and advance the volume clock.

        Returns:
            VolumeBucket if a bucket completed on this trade, else None.
        """
        if quantity <= 0 or price <= 0:
            return None

        p = float(price)
        q = float(quantity)
        ts = int(timestamp_ns)

        # 1. Determine trade direction: Lee-Ready algorithm with Tick rule fallback
        direction = self._classify_trade_direction(p, side)

        self._price_history.append(p)
        self._direction_history.append(direction)
        self._last_trade_price = p
        self._total_volume_processed += q

        # 2. Accumulate into current volume bucket
        rem_q = q
        last_completed: Optional[VolumeBucket] = None

        while rem_q > 1e-9:
            space_in_bucket = self.bucket_volume - self._current_bucket_total_vol
            fill_in_bucket = min(rem_q, space_in_bucket)

            if self._current_bucket_total_vol <= 1e-9:
                self._current_bucket_start_ts = ts

            if direction > 0:
                self._current_bucket_buy_vol += fill_in_bucket
            else:
                self._current_bucket_sell_vol += fill_in_bucket

            self._current_bucket_total_vol += fill_in_bucket
            self._current_bucket_notional += p * fill_in_bucket
            rem_q -= fill_in_bucket

            # Check if bucket is full
            if self._current_bucket_total_vol >= self.bucket_volume - 1e-9:
                vwap = (
                    self._current_bucket_notional / self._current_bucket_total_vol
                    if self._current_bucket_total_vol > 0
                    else p
                )
                imbalance = abs(self._current_bucket_buy_vol - self._current_bucket_sell_vol)
                bucket = VolumeBucket(
                    bucket_index=self._current_bucket_idx,
                    start_timestamp_ns=self._current_bucket_start_ts,
                    end_timestamp_ns=ts,
                    total_volume=self._current_bucket_total_vol,
                    buy_volume=self._current_bucket_buy_vol,
                    sell_volume=self._current_bucket_sell_vol,
                    vwap=vwap,
                    order_imbalance=imbalance,
                )
                self._completed_buckets.append(bucket)
                self._current_bucket_idx += 1
                last_completed = bucket

                # Reset bucket
                self._current_bucket_buy_vol = 0.0
                self._current_bucket_sell_vol = 0.0
                self._current_bucket_total_vol = 0.0
                self._current_bucket_notional = 0.0
                self._current_bucket_start_ts = ts

        return last_completed

    def compute_vpin(self) -> float:
        """Compute Volume-Synchronized Probability of Toxicity across window buckets."""
        if not self._completed_buckets:
            return 0.0

        total_imbalance = sum(b.order_imbalance for b in self._completed_buckets)
        total_vol = sum(b.total_volume for b in self._completed_buckets)

        if total_vol <= 1e-9:
            return 0.0

        vpin = total_imbalance / total_vol
        return float(np.clip(vpin, 0.0, 1.0))

    def compute_roll_spread(self) -> Tuple[float, float]:
        """Compute Roll (1984) effective bid-ask spread and spread in bps.

        s = 2 * sqrt(-cov(Delta P_t, Delta P_{t-1})) if cov < 0 else 0
        """
        if len(self._price_history) < 4:
            return (0.0, 0.0)

        prices = np.array(self._price_history, dtype=np.float64)
        delta_p = np.diff(prices)

        if len(delta_p) < 2:
            return (0.0, 0.0)

        dp_t = delta_p[1:]
        dp_prev = delta_p[:-1]

        # Serial autocovariance
        cov = float(np.cov(dp_t, dp_prev)[0, 1])

        if cov < 0:
            roll_spread = 2.0 * math.sqrt(-cov)
        else:
            roll_spread = 0.0

        last_p = float(prices[-1])
        roll_bps = (roll_spread / last_p * 10000.0) if last_p > 0 else 0.0

        return (float(roll_spread), float(roll_bps))

    def compute_kyle_lambda(self) -> float:
        """Compute Kyle's Lambda price impact parameter = cov(Delta P, Q) / var(Q)."""
        if len(self._price_history) < 4 or len(self._direction_history) < 4:
            return 0.0

        prices = np.array(self._price_history, dtype=np.float64)
        directions = np.array(self._direction_history, dtype=np.float64)

        delta_p = np.diff(prices)
        dirs = directions[1:]

        var_q = float(np.var(dirs))
        if var_q <= 1e-9:
            return 0.0

        cov_pq = float(np.cov(delta_p, dirs)[0, 1])
        kyle_lambda = max(0.0, cov_pq / var_q)
        return float(kyle_lambda)

    def get_metrics_snapshot(self) -> ToxicityMetricsSnapshot:
        """Compile complete real-time order flow toxicity snapshot."""
        vpin = self.compute_vpin()
        roll_spread, roll_bps = self.compute_roll_spread()
        kyle_lambda = self.compute_kyle_lambda()

        # Regime classification
        if vpin >= 0.65:
            regime = FlowToxicityRegime.EXTREME_ADVERSE
            rec = "CRITICAL TOXICITY: Halt passive quoting, aggressive flow skew detected."
        elif vpin >= 0.45:
            regime = FlowToxicityRegime.ELEVATED_TOXICITY
            rec = "ELEVATED TOXICITY: Widen quoting spreads by 50%, reduce quote sizes."
        elif vpin >= 0.25:
            regime = FlowToxicityRegime.MODERATE
            rec = "MODERATE: Normal market making parameters with standard inventory limits."
        else:
            regime = FlowToxicityRegime.BENIGN
            rec = "BENIGN FLOW: Flow is predominantly uninformed noise; capture full spread."

        # Informed trader fraction estimation: alpha * delta = VPIN
        informed_ratio = vpin

        return ToxicityMetricsSnapshot(
            symbol=self.symbol,
            timestamp_utc=datetime.now(timezone.utc),
            current_vpin=round(vpin, 4),
            toxicity_regime=regime,
            roll_effective_spread=round(roll_spread, 4),
            roll_spread_bps=round(roll_bps, 2),
            kyle_lambda_impact=round(kyle_lambda, 6),
            informed_trader_ratio=round(informed_ratio, 4),
            completed_buckets_count=len(self._completed_buckets),
            total_volume_processed=round(self._total_volume_processed, 4),
            recommendation=rec,
        )

    # -------------------------------------------------------------------------
    # Internal Classification Routines
    # -------------------------------------------------------------------------

    def _classify_trade_direction(self, price: float, side: Optional[str]) -> int:
        """Classify trade as buyer-initiated (+1) or seller-initiated (-1)."""
        if side:
            s_up = side.upper()
            if s_up in ("BUY", "BID"):
                return 1
            if s_up in ("SELL", "ASK"):
                return -1

        # 1. Lee-Ready Quote Rule: Compare trade price to prior BBO mid
        if self._last_bbo_mid is not None:
            if price > self._last_bbo_mid + 1e-8:
                return 1
            if price < self._last_bbo_mid - 1e-8:
                return -1

        # 2. Tick Rule fallback: Compare trade price to prior trade price
        if self._last_trade_price is not None:
            if price > self._last_trade_price + 1e-8:
                return 1
            if price < self._last_trade_price - 1e-8:
                return -1

        # 3. If price identical to prior mid and trade, inherit previous trade direction
        if self._direction_history:
            return self._direction_history[-1]

        return 1  # Default to buy if first trade with no prior info
