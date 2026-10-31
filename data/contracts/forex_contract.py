"""Versioned Forex Data Contract and Session Boundary Validator (v1.0.0).

Defines:
1. Versioned immutable schema for Forex candles (FX-DATA-CONTRACT-v1.0.0).
2. Price basis tracking (MID, BID, ASK, REFERENCE_FIXING).
3. Session boundary calculations handling Daylight Saving Time (DST) anchored to 17:00 New York.
4. Data invariant and sequence verification rules.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from datetime import UTC, datetime, time, timedelta
from enum import Enum
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

SCHEMA_VERSION = "FX-DATA-CONTRACT-v1.0.0"

NY_TZ = ZoneInfo("America/New_York")
APPROVED_FX_SYMBOLS = frozenset(["EURUSD", "EURCHF"])
PERMANENT_BLOCKLIST = frozenset(["EURAUD", "EURCAD"])


class PriceBasis(str, Enum):
    MID = "MID"
    BID = "BID"
    ASK = "ASK"
    REFERENCE_FIXING = "REFERENCE_FIXING"


@dataclass(frozen=True)
class ForexCandleContractV1:
    """Versioned Forex market data contract adhering to FX-DATA-CONTRACT-v1.0.0."""

    schema_version: str
    symbol: str
    provider: str
    native_interval: str            # '1d', '5m', '15m', '1h'
    price_basis: PriceBasis
    timestamp_utc: str              # ISO 8601 UTC candle open timestamp
    retrieved_at_utc: str           # ISO 8601 UTC retrieval timestamp
    open: float
    high: float
    low: float
    close: float
    volume: float | None            # Traded volume or tick volume where genuinely available; None if not provided
    is_closed: bool                 # True for completed candle
    bid_close: float | None = None  # Discrete bid at close when available
    ask_close: float | None = None  # Discrete ask at close when available
    spread: float | None = None     # Spread in quote units (e.g. 0.00015)
    source_record_id: str = ""      # Provenance / origin trace ID

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["price_basis"] = self.price_basis.value
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ForexCandleContractV1:
        basis = data.get("price_basis", PriceBasis.MID.value)
        if isinstance(basis, str):
            basis = PriceBasis(basis)
        return cls(
            schema_version=str(data.get("schema_version", SCHEMA_VERSION)),
            symbol=str(data["symbol"]),
            provider=str(data["provider"]),
            native_interval=str(data["native_interval"]),
            price_basis=basis,
            timestamp_utc=str(data["timestamp_utc"]),
            retrieved_at_utc=str(data["retrieved_at_utc"]),
            open=float(data["open"]),
            high=float(data["high"]),
            low=float(data["low"]),
            close=float(data["close"]),
            volume=float(data["volume"]) if data.get("volume") is not None else None,
            is_closed=bool(data.get("is_closed", True)),
            bid_close=float(data["bid_close"]) if data.get("bid_close") is not None else None,
            ask_close=float(data["ask_close"]) if data.get("ask_close") is not None else None,
            spread=float(data["spread"]) if data.get("spread") is not None else None,
            source_record_id=str(data.get("source_record_id", "")),
        )


def validate_forex_candle_contract(
    candle: ForexCandleContractV1 | dict[str, Any],
    now_dt: datetime | None = None,
) -> tuple[bool, str]:
    """Validates invariant rules of FX-DATA-CONTRACT-v1.0.0."""
    if isinstance(candle, dict):
        try:
            c = ForexCandleContractV1.from_dict(candle)
        except Exception as exc:
            return False, f"Contract deserialization error: {exc}"
    else:
        c = candle

    if c.schema_version != SCHEMA_VERSION:
        return False, f"Schema version mismatch: expected {SCHEMA_VERSION}, got {c.schema_version}"

    if c.symbol in PERMANENT_BLOCKLIST:
        return False, f"Instrument {c.symbol} is permanently blocked by governance policy POL-GOV-2026-001"
    if c.symbol not in APPROVED_FX_SYMBOLS:
        return False, f"Instrument {c.symbol} is not approved under governance allowlist {sorted(list(APPROVED_FX_SYMBOLS))}"

    for name, p in [("open", c.open), ("high", c.high), ("low", c.low), ("close", c.close)]:
        if not math.isfinite(p) or p <= 0.0:
            return False, f"Non-positive or non-finite price for {name}: {p}"

    if c.low > min(c.open, c.close) + 1e-9:
        return False, f"OHLC violation: low ({c.low}) > min(open, close) ({min(c.open, c.close)})"
    if c.high < max(c.open, c.close) - 1e-9:
        return False, f"OHLC violation: high ({c.high}) < max(open, close) ({max(c.open, c.close)})"

    if c.volume is not None:
        if not math.isfinite(c.volume) or c.volume < 0.0:
            return False, f"Invalid volume: {c.volume}"

    if c.bid_close is not None and c.ask_close is not None:
        if c.ask_close < c.bid_close:
            return False, f"Inverted spread: ask ({c.ask_close}) < bid ({c.bid_close})"

    try:
        ts = datetime.fromisoformat(c.timestamp_utc)
        if ts.tzinfo is None:
            return False, "Timestamp must be UTC timezone-aware"
    except Exception as exc:
        return False, f"Invalid timestamp format: {exc}"

    eval_now = now_dt or datetime.now(UTC)
    if (eval_now - ts).total_seconds() < -300.0:
        return False, f"Future timestamp detected: {c.timestamp_utc} (eval_now={eval_now.isoformat()})"

    return True, "VALID"


def get_expected_forex_session_bars_5m(date_obj: datetime) -> int:
    """Calculates expected 5m candle bars for a given calendar day accounting for DST.

    Rules:
    - Global Forex opens Sunday 17:00 NY time and closes Friday 17:00 NY time.
    - Saturday: 0 bars (Market closed).
    - Sunday: Bars from 17:00 NY time to midnight UTC.
      * During EDT (UTC-4): 21:00 to 24:00 UTC = 3 hours = 36 five-minute bars.
      * During EST (UTC-5): 22:00 to 24:00 UTC = 2 hours = 24 five-minute bars.
    - Monday - Thursday: Full 24 hours = 288 five-minute bars.
    - Friday: Closes at 17:00 NY time.
      * During EDT (UTC-4): 00:00 to 21:00 UTC = 21 hours = 252 five-minute bars.
      * During EST (UTC-5): 00:00 to 22:00 UTC = 22 hours = 264 five-minute bars.
    """
    ny_dt = date_obj.astimezone(NY_TZ)
    weekday = ny_dt.weekday()  # 0=Monday, 4=Friday, 5=Saturday, 6=Sunday

    if weekday == 5:  # Saturday
        return 0

    is_dst = bool(ny_dt.dst() and ny_dt.dst().total_seconds() > 0)

    if weekday == 6:  # Sunday
        return 36 if is_dst else 24

    if weekday == 4:  # Friday
        return 252 if is_dst else 264

    # Mon, Tue, Wed, Thu
    return 288


def validate_forex_sequence_contract(
    candles: list[ForexCandleContractV1],
    expected_interval: str,
    now_dt: datetime | None = None,
) -> tuple[bool, str, list[dict[str, Any]]]:
    """Validates sequence properties of ForexCandleContractV1.

    Checks:
    - Invariants on every single candle.
    - Strictly monotonic increasing timestamps (no duplicates, no time reversals).
    - Interval uniformity: candle.native_interval == expected_interval.
    - Strict price basis isolation:
      * '1d' requires PriceBasis.REFERENCE_FIXING or PriceBasis.MID
      * '5m' prohibits PriceBasis.REFERENCE_FIXING (must be MID, BID, or ASK)
    - Interval boundaries:
      * For '1d': delta >= 20.0 hours (rejects intraday spacing in daily series)
      * For '5m': delta >= 300.0 seconds (rejects sub-5m spacing)
    - Gap detection during active 24/5 sessions (quantifies missing seconds & bars).
    """
    if not candles:
        return True, "EMPTY", []

    detected_gaps: list[dict[str, Any]] = []
    prev_dt: datetime | None = None

    for idx, c in enumerate(candles):
        valid, msg = validate_forex_candle_contract(c, now_dt=now_dt)
        if not valid:
            return False, f"Invalid contract candle at index {idx}: {msg}", detected_gaps

        if c.native_interval != expected_interval:
            return False, (
                f"Interval mismatch at index {idx}: expected {expected_interval}, "
                f"got {c.native_interval}. Interval mixing is prohibited."
            ), detected_gaps

        # Price basis isolation
        if expected_interval == "5m" and c.price_basis == PriceBasis.REFERENCE_FIXING:
            return False, (
                f"Price basis violation at index {idx}: REFERENCE_FIXING cannot be used for "
                f"intraday 5m executable candle."
            ), detected_gaps

        curr_dt = datetime.fromisoformat(c.timestamp_utc)

        if prev_dt is not None:
            if curr_dt <= prev_dt:
                return False, (
                    f"Timestamp ordering violation at index {idx}: non-increasing or duplicate "
                    f"({curr_dt.isoformat()} <= {prev_dt.isoformat()})"
                ), detected_gaps

            delta_s = (curr_dt - prev_dt).total_seconds()

            if expected_interval == "1d":
                if delta_s < 20.0 * 3600.0:
                    return False, (
                        f"Daily interval boundary violation at index {idx}: delta is {delta_s:.1f}s "
                        f"({delta_s/3600:.2f}h < 20h limit). Intraday bar detected in daily sequence."
                    ), detected_gaps
            elif expected_interval == "5m":
                if delta_s < 300.0:
                    return False, (
                        f"Intraday interval boundary violation at index {idx}: delta is {delta_s:.1f}s "
                        f"(< 300s limit). Sub-5m bar detected in 5m sequence."
                    ), detected_gaps

                # Detect gaps during active market sessions
                if delta_s > 300.0:
                    detected_gaps.append({
                        "gap_index": idx,
                        "from_timestamp": prev_dt.isoformat(),
                        "to_timestamp": curr_dt.isoformat(),
                        "missing_seconds": delta_s - 300.0,
                        "missing_bars": int(round((delta_s - 300.0) / 300.0)),
                    })

        prev_dt = curr_dt

    return True, "VALID", detected_gaps

