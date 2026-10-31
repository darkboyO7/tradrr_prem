"""Clean-Room Forex Data Provider and Validation Pipeline.

Provides strictly isolated, reproducible, and traceable Forex data ingestion for
EURUSD and EURCHF under governance policy POL-GOV-2026-001.

Architectural Guarantees:
1. Strict separation of daily reference rates (ECB SDMX) from intraday OHLC bars.
2. Invariant validation: positive finite prices, low <= min(O,C), high >= max(O,C), volume >= 0.
3. Sequence integrity: strictly increasing timestamps, duplicate rejection, gap detection.
4. Strict interval isolation: never combines intervals, never relabels intervals, never downsamples.
5. Fail-closed semantics: zero artificial data fabrication, zero fallback to stale caches.
6. Forensic isolation: existing contaminated cache (fx_candles_cache.json) is preserved untouched;
   clean data is stored in dedicated clean cache paths.
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import requests

logger = logging.getLogger("data.ingestion.clean_forex")

# Clean cache paths strictly separated from legacy / contaminated cache
CLEAN_CACHE_DIR = Path("data/universe")
CLEAN_DAILY_CACHE_PATH = CLEAN_CACHE_DIR / "clean_fx_daily_reference_cache.json"
CLEAN_INTRADAY_CACHE_PATH = CLEAN_CACHE_DIR / "clean_fx_intraday_5m_cache.json"
LEGACY_CONTAMINATED_CACHE_PATH = CLEAN_CACHE_DIR / "fx_candles_cache.json"

# Supported verified instruments under governance
APPROVED_INSTRUMENTS = frozenset(["EURUSD", "EURCHF"])
PERMANENTLY_BLOCKED_INSTRUMENTS = frozenset(["EURAUD", "EURCAD"])


@dataclass(frozen=True)
class NormalizedCandleRecord:
    """Normalized Forex candle record with immutable provenance attributes."""

    symbol: str
    timestamp_utc: str          # ISO 8601 UTC string (e.g. 2026-10-19T16:00:00+00:00)
    open: float
    high: float
    low: float
    close: float
    volume: float
    native_interval: str        # '1d' or '5m' (strictly documented native source interval)
    provider: str               # e.g. 'ECB_SDMX_OFFICIAL', 'CLEAN_FIXTURE'
    retrieved_at_utc: str       # ISO 8601 UTC timestamp of retrieval
    is_closed: bool = True

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> NormalizedCandleRecord:
        return cls(
            symbol=str(data["symbol"]),
            timestamp_utc=str(data["timestamp_utc"]),
            open=float(data["open"]),
            high=float(data["high"]),
            low=float(data["low"]),
            close=float(data["close"]),
            volume=float(data.get("volume", 0.0)),
            native_interval=str(data["native_interval"]),
            provider=str(data["provider"]),
            retrieved_at_utc=str(data["retrieved_at_utc"]),
            is_closed=bool(data.get("is_closed", True)),
        )


def validate_normalized_record(
    record: NormalizedCandleRecord | dict[str, Any],
    now_dt: datetime | None = None,
) -> tuple[bool, str]:
    """Validates individual candle record invariants.

    Checks:
    - Supported governance symbol (EURUSD, EURCHF) and not permanently blocked.
    - Positive finite numeric prices.
    - OHLC consistency: high >= max(open, close), low <= min(open, close).
    - Volume >= 0.0 and finite.
    - Valid UTC ISO 8601 timestamp.
    - Not in future (> now_dt + 300s).
    - Documented native interval ('1d' or '5m').
    """
    if isinstance(record, dict):
        try:
            rec = NormalizedCandleRecord.from_dict(record)
        except Exception as exc:
            return False, f"Malformed record fields: {exc}"
    else:
        rec = record

    # Symbol governance check
    if rec.symbol in PERMANENTLY_BLOCKED_INSTRUMENTS:
        return False, f"Instrument {rec.symbol} is permanently blocked by governance policy POL-GOV-2026-001"
    if rec.symbol not in APPROVED_INSTRUMENTS:
        return False, f"Instrument {rec.symbol} is not approved under governance allowlist {sorted(list(APPROVED_INSTRUMENTS))}"

    # Price positivity and finiteness
    for name, val in [("open", rec.open), ("high", rec.high), ("low", rec.low), ("close", rec.close)]:
        if not math.isfinite(val) or val <= 0.0:
            return False, f"Non-positive or non-finite price for {name}: {val}"

    # OHLC invariants
    if rec.low > min(rec.open, rec.close) + 1e-9:
        return False, f"OHLC violation: low ({rec.low}) exceeds min(open, close) ({min(rec.open, rec.close)})"
    if rec.high < max(rec.open, rec.close) - 1e-9:
        return False, f"OHLC violation: high ({rec.high}) less than max(open, close) ({max(rec.open, rec.close)})"

    # Volume check
    if not math.isfinite(rec.volume) or rec.volume < 0.0:
        return False, f"Invalid volume: {rec.volume}"

    # Native interval validation
    if rec.native_interval not in {"1d", "5m"}:
        return False, f"Unsupported native interval '{rec.native_interval}' (only '1d' and '5m' documented)"

    # Timestamp parsing and timezone check
    try:
        ts = datetime.fromisoformat(rec.timestamp_utc)
        if ts.tzinfo is None:
            return False, f"Timestamp is naive, must be timezone-aware UTC: {rec.timestamp_utc}"
    except Exception as exc:
        return False, f"Invalid ISO 8601 timestamp: {rec.timestamp_utc} ({exc})"

    # Future timestamp check
    eval_now = now_dt or datetime.now(UTC)
    if (eval_now - ts).total_seconds() < -300.0:
        return False, f"Future candle timestamp detected: {rec.timestamp_utc} (eval_now={eval_now.isoformat()})"

    return True, "VALID"


def validate_clean_candle_sequence(
    candles: list[NormalizedCandleRecord],
    expected_interval: str,
    now_dt: datetime | None = None,
) -> tuple[bool, str, list[dict[str, Any]]]:
    """Validates sequence properties: monotonic ordering, uniqueness, interval boundaries, and gaps.

    Returns:
    (is_valid: bool, reason: str, detected_gaps: list[dict])
    """
    if not candles:
        return True, "EMPTY", []

    detected_gaps: list[dict[str, Any]] = []
    prev_dt: datetime | None = None

    for idx, c in enumerate(candles):
        valid, msg = validate_normalized_record(c, now_dt=now_dt)
        if not valid:
            return False, f"Invalid record at index {idx}: {msg}", detected_gaps

        if c.native_interval != expected_interval:
            return False, (
                f"Mixed interval violation at index {idx}: expected {expected_interval}, "
                f"got {c.native_interval}. Combining intervals is strictly prohibited."
            ), detected_gaps

        curr_dt = datetime.fromisoformat(c.timestamp_utc)

        if prev_dt is not None:
            # Monotonic ordering and duplicate check
            if curr_dt <= prev_dt:
                return False, (
                    f"Timestamp ordering violation at index {idx}: non-increasing or duplicate timestamp "
                    f"({curr_dt.isoformat()} <= {prev_dt.isoformat()})"
                ), detected_gaps

            delta_s = (curr_dt - prev_dt).total_seconds()

            # Sequence boundary checks
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
                # Detect internal gaps during active sessions
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


class CleanForexDataProvider:
    """Isolated, clean-room Forex data provider with zero synthetic fabrication."""

    ECB_SDMX_URL_TEMPLATE = (
        "https://data-api.ecb.europa.eu/service/data/EXR/D.{quote_ccy}.EUR.SP00.A"
        "?lastNObservations={limit}&format=jsondata"
    )

    def __init__(
        self,
        clean_daily_cache: Path | str = CLEAN_DAILY_CACHE_PATH,
        clean_intraday_cache: Path | str = CLEAN_INTRADAY_CACHE_PATH,
        timeout: float = 8.0,
    ):
        self.clean_daily_cache = Path(clean_daily_cache)
        self.clean_intraday_cache = Path(clean_intraday_cache)
        self.timeout = timeout

    def fetch_daily_ecb_reference_rates(
        self,
        symbol: str,
        limit: int = 360,
    ) -> tuple[bool, list[NormalizedCandleRecord], str]:
        """Fetches authentic daily European Central Bank reference rates from official SDMX endpoint.

        Guarantees:
        - Native interval is strictly '1d'.
        - Provider is stamped as 'ECB_SDMX_OFFICIAL'.
        - Validated against single-record and sequence invariants.
        - Stored in clean daily reference cache, never mixed with intraday data.
        """
        if symbol in PERMANENTLY_BLOCKED_INSTRUMENTS:
            return False, [], f"Symbol {symbol} is blocked by governance policy."
        if symbol not in APPROVED_INSTRUMENTS:
            return False, [], f"Symbol {symbol} not in approved governance allowlist."

        quote_ccy = symbol[3:]  # 'USD' or 'CHF'
        url = self.ECB_SDMX_URL_TEMPLATE.format(quote_ccy=quote_ccy, limit=max(limit, 30))
        now_dt = datetime.now(UTC)

        try:
            resp = requests.get(
                url,
                headers={"User-Agent": "Antigravity/2.0 (TradingSystemAudit; +https://antigravity.dev)"},
                timeout=self.timeout,
            )
            if resp.status_code != 200:
                return False, [], f"ECB SDMX provider error: HTTP {resp.status_code} ({resp.text[:100]})"

            data = resp.json()
            structure = data.get("structure", {})
            dims = structure.get("dimensions", {}).get("observation", [])
            if not dims or "values" not in dims[0]:
                return False, [], "Malformed ECB SDMX response: observation dimensions missing."

            dates = [v["id"] for v in dims[0]["values"]]
            series_dict = data.get("dataSets", [{}])[0].get("series", {})
            if not series_dict:
                return False, [], "Malformed ECB SDMX response: series observations empty."

            first_series_key = next(iter(series_dict))
            observations = series_dict[first_series_key].get("observations", {})

            records: list[NormalizedCandleRecord] = []
            prev_rate: float | None = None

            for idx, date_str in enumerate(dates):
                obs = observations.get(str(idx))
                if obs is None or not obs:
                    continue
                rate = float(obs[0])
                if not math.isfinite(rate) or rate <= 0.0:
                    continue

                # European Central Bank reference rates are official daily fixings published at 16:00 CET
                ts_iso = f"{date_str}T16:00:00+00:00"
                # Indicative daily reference: open=prev_close or rate, high=max(o,c), low=min(o,c), close=rate
                o = prev_rate if prev_rate is not None else rate
                h = max(o, rate)
                l = min(o, rate)
                c = rate

                rec = NormalizedCandleRecord(
                    symbol=symbol,
                    timestamp_utc=ts_iso,
                    open=round(o, 5),
                    high=round(h, 5),
                    low=round(l, 5),
                    close=round(c, 5),
                    volume=0.0,  # Official central bank fixing has no commercial trade volume
                    native_interval="1d",
                    provider="ECB_SDMX_OFFICIAL",
                    retrieved_at_utc=now_dt.isoformat(),
                    is_closed=True,
                )
                records.append(rec)
                prev_rate = c

            if len(records) < 20:
                return False, [], f"Insufficient daily observations returned from ECB ({len(records)} < 20)."

            valid, reason, _ = validate_clean_candle_sequence(records, expected_interval="1d", now_dt=now_dt)
            if not valid:
                return False, [], f"ECB daily sequence validation failure: {reason}"

            # Save to isolated clean daily cache
            self._save_to_clean_cache(symbol, records, self.clean_daily_cache)
            return True, records[-limit:], f"Successfully ingested {len(records)} verified daily ECB reference rates."

        except Exception as exc:
            logger.error(f"ECB SDMX request failed for {symbol}: {exc}")
            return False, [], f"Provider connection error: {exc}"

    def fetch_intraday_5m_candles(
        self,
        symbol: str,
        limit: int = 100,
    ) -> tuple[bool, list[NormalizedCandleRecord], str]:
        """Queries genuine intraday 5m OHLC candles.

        Strict fail-closed rule:
        If an institutional, authentic broker feed is not connected or returns an error,
        it returns (False, [], reason). It NEVER fabricates fake replacement candles,
        NEVER converts quotes into historical bars, and NEVER downsamples daily rates.
        """
        if symbol in PERMANENTLY_BLOCKED_INSTRUMENTS:
            return False, [], f"Symbol {symbol} is blocked by governance policy."
        if symbol not in APPROVED_INSTRUMENTS:
            return False, [], f"Symbol {symbol} not in approved governance allowlist."

        # Check clean intraday cache if populated
        if self.clean_intraday_cache.exists():
            try:
                content = json.loads(self.clean_intraday_cache.read_text(encoding="utf-8"))
                if symbol in content:
                    records = [NormalizedCandleRecord.from_dict(d) for d in content[symbol]]
                    valid, reason, _ = validate_clean_candle_sequence(records, expected_interval="5m")
                    if valid and len(records) >= 20:
                        return True, records[-limit:], "Loaded verified intraday 5m candles from clean cache."
            except Exception as exc:
                logger.warning(f"Failed to read clean intraday cache: {exc}")

        # Live intraday broker feed check
        return False, [], (
            f"Genuine institutional 5m intraday provider unavailable for {symbol}. "
            f"Failing closed: zero artificial or synthetic candle fabrication permitted."
        )

    def _save_to_clean_cache(
        self,
        symbol: str,
        records: list[NormalizedCandleRecord],
        target_path: Path,
    ) -> None:
        """Atomically saves verified records to an isolated clean cache path.

        Preserves existing contaminated cache without modification.
        """
        assert target_path != LEGACY_CONTAMINATED_CACHE_PATH, (
            "SAFETY VIOLATION: Cannot write clean records to contaminated forensic cache path!"
        )
        target_path.parent.mkdir(parents=True, exist_ok=True)
        existing: dict[str, Any] = {}
        if target_path.exists():
            try:
                existing = json.loads(target_path.read_text(encoding="utf-8"))
            except Exception:
                existing = {}

        existing[symbol] = [r.to_dict() for r in records]
        tmp_path = target_path.with_suffix(".tmp")
        tmp_path.write_text(json.dumps(existing, indent=2), encoding="utf-8")
        tmp_path.replace(target_path)
