"""Trading Intelligence Platform - Dedicated Paper-Trading Background Worker.

Continuously executes prospective paper-trading:
1. Connects to verified live Forex market data feeds.
2. Checks market session (24/5) and validates data freshness.
3. Evaluates completed candles and runs the frozen LightGBM candidate model.
4. Records prospective predictions before outcomes are known (append-only, crash-safe).
5. Resolves matured predictions using observed live exit prices with realistic execution modeling
   (Ask entry / Bid exit for BUY, Bid entry / Ask exit for SELL, no double-counting of spread).
6. Recovers pending predictions upon restart and prevents duplicate workers via process lock.
7. Keeps paper-trading data strictly isolated from production ledgers.
"""

import json
import logging
import os
import sys
import threading
import time
from dataclasses import asdict
from datetime import UTC, datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from core.events import Candle
from core.instrument import MarketType
from core.symbol_resolver import SymbolResolver
from core.system_lock import ProcessFileLock
from data.ingestion.bse import BSEDataProvider
from data.ingestion.forex import ForexDataProvider, validate_candle_sequence
from data.ingestion.nse import NSEDataProvider
from data.ingestion.quotex import QuotexMarketDataAdapter
from features.pipeline import FeaturePipeline
from models.ml_classifier import LightGBMPredictor
from paper_trading.prediction_store import (
    DEFAULT_PAPER_STORE_DIR,
    PaperPredictionRecord,
    PaperTradingPredictionStore,
)

logger = logging.getLogger("paper_trading.worker")

# Frozen model paths and constants
FROZEN_MODEL_DIR = Path("models_store/lgbm_candidate_v20261020_044046/v20261020_044046")
FROZEN_MODEL_PATH = FROZEN_MODEL_DIR / "model.joblib"
FROZEN_META_PATH = FROZEN_MODEL_DIR / "metadata.json"

FROZEN_MODEL_SHA256 = "d955be760504e9f690b6d3c21d9707fb7ad56a39c3ba5f03fe7aebb833ac45c2"
FROZEN_CONFIG_HASH = "ebf940958de5f617e359cddaf802570ce0bf2e05a91f41a7350b0ec952f4307e"
EVALUATION_POLICY_VERSION = "POL-GOV-2026-001"

FROZEN_TEMPERATURE = 1.9615
FROZEN_DECISION_MARGIN = 0.0800
FROZEN_MIN_CONFIDENCE = 0.4200

# Provider-Neutral Market Governance Lists (NSE, BSE, and Quotex Proxy)
MAX_PERMISSIBLE_GOVERNANCE_INSTRUMENTS = frozenset([
    # In-Scope Equities: NSE
    "NSE:TCS", "NSE:INFY", "NSE:RELIANCE", "NSE:SBIN", "NSE:HDFCBANK", "NSE:IRCTC", "NSE:IDEA",
    "TCS", "INFY", "RELIANCE", "SBIN", "HDFCBANK", "IRCTC", "IDEA",
    # In-Scope Equities: BSE
    "BSE:RELIANCE", "BSE:TCS", "BSE:INFY", "BSE:500325", "500325",
    # Quotex Crypto Proxies (Research Mode Only)
    "QUOTEX:BTCUSD_QX", "QUOTEX:ETHUSD_QX", "BTCUSD_QX", "ETHUSD_QX",
    # Historical Forex (Preserved for tests)
    "EURUSD", "EURCHF",
])

PERMANENT_GOVERNANCE_BLOCKLIST = frozenset([
    "EURAUD", "EURCAD",
    "QUOTEX:MSFT_OTC", "QUOTEX:EURUSD_OTC", "QUOTEX:GBPUSD_OTC",
    "MSFT_OTC", "EURUSD_OTC", "GBPUSD_OTC",
    "FX:EURUSD", "FX:EURCHF",
])

ACTIVE_GOVERNANCE_ALLOWLIST = [
    "NSE:TCS", "NSE:INFY", "NSE:RELIANCE", "NSE:SBIN", "NSE:HDFCBANK",
    "BSE:RELIANCE", "BSE:TCS", "BSE:INFY",
    "QUOTEX:BTCUSD_QX",
    "EURUSD", "EURCHF",
]
EXPLICIT_BLOCKLIST = list(PERMANENT_GOVERNANCE_BLOCKLIST)
VERIFIED_FX_INSTRUMENTS = ["EURUSD", "EURCHF"]
VERIFIED_EQUITY_INSTRUMENTS = [
    "NSE:TCS", "NSE:INFY", "NSE:RELIANCE", "NSE:SBIN", "NSE:HDFCBANK",
    "BSE:RELIANCE", "BSE:TCS", "BSE:INFY",
]
VERIFIED_PROXY_INSTRUMENTS = ["QUOTEX:BTCUSD_QX"]


class WorkerState(str, Enum):
    STOPPED = "STOPPED"
    STARTING = "STARTING"
    RUNNING = "RUNNING"
    DEGRADED = "DEGRADED"
    STOPPING = "STOPPING"
    ERROR = "ERROR"


def temp_scale(logits: np.ndarray, temp: float) -> np.ndarray:
    scaled = logits / temp
    scaled_exp = np.exp(scaled - np.max(scaled, axis=1, keepdims=True))
    return scaled_exp / np.sum(scaled_exp, axis=1, keepdims=True)


def get_applicable_freshness_threshold(timeframe: str, now_dt: datetime, is_open: bool = True) -> tuple[float, str]:
    """Calculates freshness threshold based on source interval and market session."""
    if timeframe == "1d":
        weekday = now_dt.weekday()  # 0=Mon, 4=Fri, 5=Sat, 6=Sun
        if weekday in (5, 6, 0):
            return 76.0 * 3600.0, "76.0h (Weekend/Monday session allowance for Friday fixing)"
        return 36.0 * 3600.0, "36.0h (Weekday daily reference rate allowance)"
    elif timeframe == "5m":
        if is_open:
            return 1800.0, "1800s (30m max delay during active session)"
        return 72.0 * 3600.0, "72.0h (Weekend closure from Friday close)"
    return 3600.0, "3600s (Default hourly threshold)"


class PaperTradingWorker:
    """Dedicated background worker for continuous prospective paper trading."""

    def __init__(
        self,
        store_dir: Path | str = DEFAULT_PAPER_STORE_DIR,
        instruments: list[str] | None = None,
        allowlist: list[str] | None = None,
        blocklist: list[str] | None = None,
        poll_interval_seconds: float = 30.0,
        stale_quote_threshold_seconds: float = 60.0,
        slippage_rate: float = 0.000025,    # 0.25 bps execution slippage
        commission_rate: float = 0.000050,  # 0.50 bps external broker commission
        financing_rate: float = 0.000020,   # 0.20 bps overnight financing / rollover
        max_candle_age_hours: float | None = None,
        fx_cache_path: Path | str = "data/universe/fx_candles_cache.json",
    ):
        self.store_dir = Path(store_dir)

        # Permanent governance enforcement: cannot be overridden by arguments or configuration
        raw_allowlist = list(allowlist if allowlist is not None else ACTIVE_GOVERNANCE_ALLOWLIST)
        raw_blocklist = list(blocklist if blocklist is not None else EXPLICIT_BLOCKLIST)

        # Blocklist must always include permanent governance blocklist
        self.blocklist = list(dict.fromkeys(raw_blocklist + list(PERMANENT_GOVERNANCE_BLOCKLIST)))

        # Allowlist cannot contain blocked instruments and cannot exceed MAX_PERMISSIBLE_GOVERNANCE_INSTRUMENTS
        self.allowlist = [
            sym for sym in dict.fromkeys(raw_allowlist)
            if sym in MAX_PERMISSIBLE_GOVERNANCE_INSTRUMENTS and sym not in self.blocklist
        ]

        self.instruments = list(instruments) if instruments is not None else list(self.allowlist)
        self.poll_interval = poll_interval_seconds
        self.stale_threshold = stale_quote_threshold_seconds
        self.slippage_rate = slippage_rate
        self.commission_rate = commission_rate
        self.financing_rate = financing_rate
        self.max_candle_age_seconds = (max_candle_age_hours * 3600.0) if max_candle_age_hours is not None else None
        self.fx_cache_path = Path(fx_cache_path)

        self.store = PaperTradingPredictionStore(store_dir=self.store_dir)
        self.lock = ProcessFileLock(self.store_dir / ".paper_worker.lock")
        self.nse_provider = NSEDataProvider(stale_threshold_seconds=self.stale_threshold)
        self.bse_provider = BSEDataProvider(stale_threshold_seconds=self.stale_threshold)
        self.quotex_provider = QuotexMarketDataAdapter(stale_threshold_seconds=self.stale_threshold)
        self.fx_provider = ForexDataProvider(stale_threshold_seconds=self.stale_threshold)
        self.provider = self.fx_provider  # Default alias for test backward compatibility
        self.feature_pipeline = FeaturePipeline()

        self._state = WorkerState.STOPPED
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._status_lock = threading.RLock()

        # Telemetry & Operational Diagnostics
        self._start_time: float | None = None
        self._start_time_iso: str | None = None
        self._last_heartbeat: str | None = None
        self._last_successful_cycle: str | None = None
        self._last_prediction_time: str | None = None
        self._last_resolution_time: str | None = None
        self._last_error: str | None = None
        self._consecutive_errors = 0
        self._retry_count = 0
        self._recent_exceptions: list[dict[str, Any]] = []
        self._last_data_time: dict[str, str] = {}
        self._last_completed_candle: dict[str, str] = {}
        self._candle_freshness: dict[str, bool] = {}
        self._candle_age_hours: dict[str, float] = {}
        self._last_prediction_time_by_symbol: dict[str, str] = {}
        self._clock_skew_seconds: dict[str, float] = {}
        self._is_eligible_to_generate: dict[str, bool] = {}
        self._instrument_status: dict[str, dict[str, Any]] = {}
        self.heartbeat_file = self.store_dir / "worker_heartbeat.json"

        # Predictor instance
        self._predictor: LightGBMPredictor | None = None
        self._feature_names: list[str] = []

    @property
    def state(self) -> WorkerState:
        with self._status_lock:
            return self._state

    def _set_state(self, new_state: WorkerState) -> None:
        with self._status_lock:
            if self._state != new_state:
                logger.info(f"PaperTradingWorker state change: {self._state.value} -> {new_state.value}")
                self._state = new_state

    def _write_heartbeat(self, override_state: WorkerState | None = None) -> None:
        """Writes real-time status and operational diagnostics atomically to worker_heartbeat.json."""
        try:
            now_dt = datetime.now(UTC)
            with self._status_lock:
                st = override_state.value if override_state else self._state.value
                uptime = (time.time() - self._start_time) if self._start_time else 0.0
                data = {
                    "worker_state": st,
                    "process_id": os.getpid(),
                    "start_time": self._start_time_iso,
                    "uptime_seconds": round(uptime, 1),
                    "last_heartbeat": now_dt.isoformat(),
                    "last_successful_cycle": self._last_successful_cycle,
                    "last_prediction_time": self._last_prediction_time,
                    "last_resolution_time": self._last_resolution_time,
                    "retry_count": self._retry_count,
                    "consecutive_errors": self._consecutive_errors,
                    "last_error": self._last_error,
                    "recent_exceptions": list(self._recent_exceptions[-10:]),
                    "last_data_time": dict(self._last_data_time),
                    "last_completed_candle": dict(self._last_completed_candle),
                    "candle_freshness": dict(self._candle_freshness),
                    "candle_age_hours": dict(self._candle_age_hours),
                    "last_prediction_time_by_symbol": dict(self._last_prediction_time_by_symbol),
                    "clock_skew_seconds": dict(self._clock_skew_seconds),
                    "is_eligible_to_generate": dict(self._is_eligible_to_generate),
                    "overall_eligible": any(self._is_eligible_to_generate.values()) if self._is_eligible_to_generate else False,
                }
            tmp_path = self.store_dir / "worker_heartbeat.json.tmp"
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
                f.flush()
                os.fsync(f.fileno())
            tmp_path.replace(self.heartbeat_file)
        except Exception as exc:
            logger.debug(f"Could not write worker heartbeat: {exc}")

    def initialize_model(self) -> bool:
        """Loads frozen model into LightGBMPredictor wrapper."""
        try:
            if not FROZEN_MODEL_PATH.exists() or not FROZEN_META_PATH.exists():
                raise FileNotFoundError(f"Frozen model missing at {FROZEN_MODEL_PATH}")

            meta = json.loads(FROZEN_META_PATH.read_text(encoding="utf-8"))
            self._feature_names = meta["feature_names"]

            predictor = LightGBMPredictor(
                min_confidence=FROZEN_MIN_CONFIDENCE,
                min_margin=FROZEN_DECISION_MARGIN,
            )
            predictor.classifier = joblib.load(FROZEN_MODEL_PATH)
            predictor.is_fitted = True
            predictor.metadata.feature_names = self._feature_names
            self._predictor = predictor
            logger.info("Frozen LightGBM predictor initialized successfully.")
            return True
        except Exception as exc:
            self._last_error = f"Model init failed: {exc}"
            logger.error(self._last_error)
            self._set_state(WorkerState.ERROR)
            return False

    def start(self) -> bool:
        """Starts the background worker thread. Guarantees single-instance process lock."""
        with self._status_lock:
            if self._state in (WorkerState.RUNNING, WorkerState.STARTING):
                logger.warning("PaperTradingWorker is already active.")
                return False

            self._set_state(WorkerState.STARTING)

            # 1. Acquire process lock
            if not self.lock.acquire():
                self._last_error = "Could not acquire process lock. Another worker instance is running."
                logger.error(self._last_error)
                self._set_state(WorkerState.ERROR)
                return False

            # 2. Initialize frozen model
            if not self.initialize_model():
                self.lock.release()
                return False

            # 3. Start background thread
            self._start_time = time.time()
            self._start_time_iso = datetime.now(UTC).isoformat()
            self._stop_event.clear()
            self._thread = threading.Thread(
                target=self._run_loop,
                name="PaperTradingWorkerThread",
                daemon=True,
            )
            self._thread.start()
            self._set_state(WorkerState.RUNNING)
            self._write_heartbeat()
            logger.info(f"PaperTradingWorker started successfully for instruments {self.instruments}")
            return True

    def stop(self, timeout: float = 10.0) -> None:
        """Gracefully stops the worker and releases process lock."""
        with self._status_lock:
            if self._state == WorkerState.STOPPED:
                return
            self._set_state(WorkerState.STOPPING)

        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=timeout)

        self._write_heartbeat(override_state=WorkerState.STOPPED)
        self.lock.release()
        self._set_state(WorkerState.STOPPED)
        logger.info("PaperTradingWorker stopped and process lock released.")

    def _run_loop(self) -> None:
        """Continuous background execution loop with bounded retries and exponential backoff."""
        while not self._stop_event.is_set():
            try:
                self.poll_once()
                self._consecutive_errors = 0
                self._last_successful_cycle = datetime.now(UTC).isoformat()
                if self.state == WorkerState.ERROR:
                    self._set_state(WorkerState.RUNNING)
                self._write_heartbeat()
                self._stop_event.wait(timeout=self.poll_interval)
            except Exception as exc:
                self._consecutive_errors += 1
                self._retry_count += 1
                err_info = {
                    "timestamp": datetime.now(UTC).isoformat(),
                    "error_type": type(exc).__name__,
                    "message": str(exc),
                }
                self._recent_exceptions.append(err_info)
                if len(self._recent_exceptions) > 20:
                    self._recent_exceptions.pop(0)
                self._last_error = f"{type(exc).__name__}: {exc}"
                logger.error(f"Worker loop error: {self._last_error}", exc_info=True)
                if self._consecutive_errors >= 3:
                    self._set_state(WorkerState.DEGRADED)
                self._write_heartbeat()

                # Exponential backoff with ceiling (5s to 60s)
                backoff = min(60.0, max(5.0, self.poll_interval * (1.5 ** min(self._consecutive_errors, 4))))
                self._stop_event.wait(timeout=backoff)

    def poll_once(self) -> dict[str, Any]:
        """Executes a single synchronous evaluation and resolution cycle across all instruments."""
        now_dt = datetime.now(UTC)
        with self._status_lock:
            self._last_heartbeat = now_dt.isoformat()

        results = {
            "timestamp": now_dt.isoformat(),
            "predictions_generated": 0,
            "predictions_resolved": 0,
            "instruments": {},
        }

        # 1. First, resolve any matured pending predictions across all instruments
        resolved_count = self._resolve_matured_predictions(now_dt)
        results["predictions_resolved"] = resolved_count

        # 2. For each configured instrument, check eligibility and generate prediction
        for sym in self.instruments:
            inst_res = self._process_instrument_prediction(sym, now_dt)
            results["instruments"][sym] = inst_res
            if inst_res.get("prediction_recorded"):
                results["predictions_generated"] += 1

        return results

    def _process_instrument_prediction(self, symbol: str, now_dt: datetime) -> dict[str, Any]:
        """Evaluates completed candles for symbol and records a prospective prediction if eligible."""
        status_entry: dict[str, Any] = {
            "symbol": symbol,
            "market_open": False,
            "quote_fresh": False,
            "prediction_recorded": False,
            "reason": "",
        }

        provider, clean_sym, market, is_proxy = self._get_provider_for_instrument(symbol)

        # 0. Governance Allowlist & Blocklist check
        if symbol in self.blocklist or symbol not in self.allowlist:
            status_entry["reason"] = (
                f"Instrument {symbol} is disabled by governance policy {EVALUATION_POLICY_VERSION} "
                f"(Allowlist: {self.allowlist}, Blocklist: {self.blocklist}). Prediction generation prohibited."
            )
            status_entry["is_eligible"] = False
            with self._status_lock:
                self._is_eligible_to_generate[symbol] = False
            self._update_instrument_telemetry(symbol, status_entry)
            return status_entry

        # Reject synthetic broker OTC contracts
        if market == "QUOTEX" and clean_sym.endswith("_OTC"):
            status_entry["reason"] = (
                f"QUOTEX_DATA_UNAVAILABLE: Synthetic broker OTC instrument {symbol} has no authorized feed. "
                "Price fabrication is prohibited."
            )
            status_entry["is_eligible"] = False
            with self._status_lock:
                self._is_eligible_to_generate[symbol] = False
            self._update_instrument_telemetry(symbol, status_entry)
            return status_entry

        # 1. Market session check
        is_open = True
        if market == "NSE":
            session = SymbolResolver.get_session(MarketType.NSE)
            is_open = session.is_open(now_dt) if session else True
        elif market == "BSE":
            session = SymbolResolver.get_session(MarketType.BSE)
            is_open = session.is_open(now_dt) if session else True
        elif market == "FOREX":
            is_open = self.provider.is_market_open(now_dt) if hasattr(self.provider, "is_market_open") else self.fx_provider.is_market_open(now_dt)

        if not is_open:
            status_entry["reason"] = "Market is closed (Weekend/Holiday session). No artificial candles."
            with self._status_lock:
                self._is_eligible_to_generate[symbol] = False
            self._update_instrument_telemetry(symbol, status_entry)
            return status_entry
        status_entry["market_open"] = True

        # 2. Fetch live quote
        if market == "FOREX":
            live_event = self.provider.fetch_live_quote_sync(symbol, timeout=4.0)
        else:
            live_event = provider.fetch_live_quote_sync(clean_sym, timeout=4.0)
        if not live_event or not live_event.raw_payload:
            status_entry["reason"] = "Live quote feed unavailable from broker gateway."
            self._set_state(WorkerState.DEGRADED)
            with self._status_lock:
                self._is_eligible_to_generate[symbol] = False
            self._update_instrument_telemetry(symbol, status_entry)
            return status_entry

        payload = live_event.raw_payload
        quote_ts = datetime.fromtimestamp(payload.get("timestamp", now_dt.timestamp()), tz=UTC)
        age_seconds = (now_dt - quote_ts).total_seconds()
        skew_seconds = abs(age_seconds)
        status_entry["clock_skew_seconds"] = round(skew_seconds, 2)
        with self._status_lock:
            self._clock_skew_seconds[symbol] = round(skew_seconds, 2)

        if skew_seconds > 300.0:
            status_entry["reason"] = f"Abnormal clock skew (skew={skew_seconds:.1f}s > 300s)."
            self._set_state(WorkerState.DEGRADED)
            with self._status_lock:
                self._is_eligible_to_generate[symbol] = False
            self._update_instrument_telemetry(symbol, status_entry)
            return status_entry

        if age_seconds > self.stale_threshold:
            status_entry["reason"] = f"Quote is stale (age={age_seconds:.1f}s > threshold={self.stale_threshold}s)."
            self._set_state(WorkerState.DEGRADED)
            with self._status_lock:
                self._is_eligible_to_generate[symbol] = False
            self._update_instrument_telemetry(symbol, status_entry)
            return status_entry

        status_entry["quote_fresh"] = True
        status_entry["live_price"] = payload.get("price")
        status_entry["bid"] = payload.get("bid")
        status_entry["ask"] = payload.get("ask")
        status_entry["spread"] = payload.get("spread")
        status_entry["source_id"] = payload.get("source", "TRADINGVIEW_FX_LIVE")
        with self._status_lock:
            self._last_data_time[symbol] = quote_ts.isoformat()

        # 3. Load historical candles
        candles = self._load_candles_for_instrument(symbol)
        if len(candles) < 30:
            status_entry["reason"] = f"Insufficient historical candle history ({len(candles)} < 30 bars)."
            with self._status_lock:
                self._is_eligible_to_generate[symbol] = False
            self._update_instrument_telemetry(symbol, status_entry)
            return status_entry

        # 3a. Sequence integrity validation (Ordering, timeframes, no intraday bars in daily sequence)
        expected_tf = "5m" if market in ("NSE", "BSE", "QUOTEX") else "1d"
        candle_objs = []
        for c in candles:
            ts = c.get("timestamp")
            if isinstance(ts, str):
                ts = datetime.fromisoformat(ts)
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=UTC)
            candle_objs.append(
                Candle(
                    symbol=symbol,
                    timeframe=expected_tf,
                    timestamp=ts,
                    open=float(c.get("open", 0.0)),
                    high=float(c.get("high", 0.0)),
                    low=float(c.get("low", 0.0)),
                    close=float(c.get("close", 0.0)),
                    volume=float(c.get("volume", 0.0)),
                )
            )
        seq_valid, seq_msg = validate_candle_sequence(candle_objs, expected_timeframe=expected_tf)
        if not seq_valid:
            status_entry["candle_fresh"] = False
            status_entry["reason"] = f"Candle sequence integrity violation for {symbol}: {seq_msg}"
            self._set_state(WorkerState.DEGRADED)
            with self._status_lock:
                self._candle_freshness[symbol] = False
                self._is_eligible_to_generate[symbol] = False
            self._update_instrument_telemetry(symbol, status_entry)
            logger.error(f"Candle sequence integrity violation for {symbol}: {seq_msg}")
            return status_entry

        df = pd.DataFrame(candles)
        df["timestamp"] = pd.to_datetime(df["timestamp"], format="ISO8601")
        df = df.sort_values("timestamp").reset_index(drop=True)
        latest_candle = df.iloc[-1]
        candle_ts_str = latest_candle["timestamp"].isoformat()
        with self._status_lock:
            self._last_completed_candle[symbol] = candle_ts_str

        # 3b. Verify Candle Freshness (Mandatory prospective integrity check)
        candle_dt = pd.to_datetime(latest_candle["timestamp"]).to_pydatetime()
        if candle_dt.tzinfo is None:
            candle_dt = candle_dt.replace(tzinfo=UTC)
        candle_age_seconds = (now_dt - candle_dt).total_seconds()
        candle_age_hours = candle_age_seconds / 3600.0
        status_entry["candle_age_hours"] = round(candle_age_hours, 1)
        with self._status_lock:
            self._candle_age_hours[symbol] = round(candle_age_hours, 1)

        # Freshness limit based on source interval and market session
        applicable_max_seconds, threshold_desc = get_applicable_freshness_threshold(
            timeframe=expected_tf, now_dt=now_dt, is_open=True
        )
        if self.max_candle_age_seconds is not None:
            applicable_max_seconds = min(applicable_max_seconds, self.max_candle_age_seconds)

        if candle_age_seconds > applicable_max_seconds:
            status_entry["candle_fresh"] = False
            status_entry["reason"] = (
                f"Candle data is STALE ({candle_age_hours:.1f}h old > {applicable_max_seconds/3600:.1f}h limit [{threshold_desc}]). "
                f"Instrument {symbol} marked UNAVAILABLE. Prediction generation skipped."
            )
            self._set_state(WorkerState.DEGRADED)
            with self._status_lock:
                self._candle_freshness[symbol] = False
                self._is_eligible_to_generate[symbol] = False
            self._update_instrument_telemetry(symbol, status_entry)
            logger.warning(f"Skipping prediction for {symbol}: candle {candle_ts_str} is stale ({candle_age_hours:.1f}h old)")
            return status_entry

        if candle_age_seconds < -300.0:
            status_entry["candle_fresh"] = False
            status_entry["reason"] = f"Candle timestamp is in the future ({candle_ts_str}). Data corruption detected."
            self._set_state(WorkerState.DEGRADED)
            with self._status_lock:
                self._candle_freshness[symbol] = False
                self._is_eligible_to_generate[symbol] = False
            self._update_instrument_telemetry(symbol, status_entry)
            logger.error(f"Future candle timestamp detected for {symbol}: {candle_ts_str}")
            return status_entry

        status_entry["candle_fresh"] = True
        with self._status_lock:
            self._candle_freshness[symbol] = True

        # 4. Check if prediction already recorded for this completed candle
        existing_pending = self.store.get_pending_predictions()
        already_exists = any(
            p["instrument"] == symbol and p["candle_timestamp"] == candle_ts_str
            for p in existing_pending
        )
        if already_exists:
            status_entry["reason"] = f"Prediction already recorded for completed candle {candle_ts_str}."
            with self._status_lock:
                self._is_eligible_to_generate[symbol] = False
            self._update_instrument_telemetry(symbol, status_entry)
            return status_entry

        # 5. Compute features strictly on completed candles
        feat = self.feature_pipeline.compute_all_features(df).copy()
        feat["macd"] = feat["macd_line"]
        latest_X = feat[self._feature_names].iloc[[-1]].apply(pd.to_numeric, errors="coerce").fillna(0.0)

        # 6. Predict using LightGBMPredictor wrapper
        assert self._predictor is not None
        raw_p = self._predictor.predict_proba(latest_X)[0]
        logits = np.log(np.clip(raw_p, 1e-12, 1.0))
        cal_p = temp_scale(logits.reshape(1, -1), FROZEN_TEMPERATURE)[0]

        p_sell = float(cal_p[0])
        p_neu = float(cal_p[1])
        p_buy = float(cal_p[2])

        s_row = sorted(cal_p, reverse=True)
        margin = s_row[0] - s_row[1]

        bid_p = float(payload.get("bid", latest_candle["close"]))
        ask_p = float(payload.get("ask", latest_candle["close"]))
        mid_p = float(payload.get("price", latest_candle["close"]))

        # 7. Decision rule & Executable Entry Price
        # BUY fills at Ask + slippage; SELL fills at Bid - slippage
        if margin >= FROZEN_DECISION_MARGIN:
            pred_class = int(np.argmax(cal_p))
            if pred_class == 2:
                action = "BUY"
                exec_entry = ask_p + (self.slippage_rate * mid_p)
                reason = f"BUY: P(Buy)={p_buy:.4f} >= margin {FROZEN_DECISION_MARGIN}"
            elif pred_class == 0:
                action = "SELL"
                exec_entry = bid_p - (self.slippage_rate * mid_p)
                reason = f"SELL: P(Sell)={p_sell:.4f} >= margin {FROZEN_DECISION_MARGIN}"
            else:
                action = "NO_TRADE"
                exec_entry = mid_p
                reason = "NEUTRAL: Highest probability is Neutral"
        else:
            action = "NO_TRADE"
            exec_entry = mid_p
            reason = f"NO_TRADE: Margin {margin:.4f} < {FROZEN_DECISION_MARGIN}"

        horizon_delta = timedelta(minutes=15) if market in ("NSE", "BSE", "QUOTEX") else timedelta(days=1)
        horizon_label = "15m" if market in ("NSE", "BSE", "QUOTEX") else "1d"
        scheduled_outcome_dt = now_dt + horizon_delta
        clean_id_sym = symbol.replace(":", "_")
        pred_id = f"paper_pred_{clean_id_sym}_{int(now_dt.timestamp())}"

        record = PaperPredictionRecord(
            prediction_id=pred_id,
            instrument=symbol,
            timeframe=expected_tf,
            forecast_horizon=horizon_label,
            candle_timestamp=candle_ts_str,
            prediction_timestamp=now_dt.isoformat(),
            model_version="v20261020_044046",
            model_sha256=FROZEN_MODEL_SHA256,
            config_hash=FROZEN_CONFIG_HASH,
            prob_sell_raw=round(float(raw_p[0]), 4),
            prob_neutral_raw=round(float(raw_p[1]), 4),
            prob_buy_raw=round(float(raw_p[2]), 4),
            prob_sell_calibrated=round(p_sell, 4),
            prob_neutral_calibrated=round(p_neu, 4),
            prob_buy_calibrated=round(p_buy, 4),
            decision=action,
            decision_margin=round(float(margin), 4),
            no_trade_reason=reason,
            reference_price=round(mid_p, 5),
            executable_entry_price=round(exec_entry, 5),
            intended_entry_rule="NEXT_BAR_OPEN_PLUS_HALF_SPREAD",
            scheduled_outcome_timestamp=scheduled_outcome_dt.isoformat(),
            data_source_id=str(payload.get("source", f"{market}_LIVE")),
            freshness_status="FRESH",
            status="PENDING",
        )

        ok = self.store.record_prediction(record)
        if ok:
            status_entry["prediction_recorded"] = True
            status_entry["prediction_id"] = pred_id
            status_entry["decision"] = action
            status_entry["margin"] = round(float(margin), 4)
            with self._status_lock:
                self._last_prediction_time = now_dt.isoformat()
                self._last_prediction_time_by_symbol[symbol] = now_dt.isoformat()
                self._is_eligible_to_generate[symbol] = True
            logger.info(f"Recorded prospective paper prediction {pred_id} for {symbol}: {action} (Entry={exec_entry:.5f})")

        self._update_instrument_telemetry(symbol, status_entry)
        return status_entry

    def _resolve_matured_predictions(self, now_dt: datetime) -> int:
        """Resolves eligible pending predictions whose forecast horizon has completed."""
        pending = self.store.get_pending_predictions()
        resolved_count = 0

        for p in pending:
            sched_ts = datetime.fromisoformat(p["scheduled_outcome_timestamp"])
            if now_dt >= sched_ts:
                pid = p["prediction_id"]
                symbol = p["instrument"]
                decision = p["decision"]
                entry_px = float(p["executable_entry_price"])

                # Fetch exit quote via appropriate market provider
                provider, clean_sym, market, is_proxy = self._get_provider_for_instrument(symbol)
                if market == "FOREX":
                    live_event = self.provider.fetch_live_quote_sync(symbol, timeout=4.0)
                else:
                    live_event = provider.fetch_live_quote_sync(clean_sym, timeout=4.0)
                if not live_event or not live_event.raw_payload:
                    # Handle missing data deferral or expiration
                    status = self.store.handle_missing_data(pid, now_dt.isoformat())
                    logger.warning(f"Could not resolve {pid}: missing exit quote -> {status}")
                    continue

                payload = live_event.raw_payload
                mid_exit = float(payload.get("price", 0.0))
                bid_exit = float(payload.get("bid", mid_exit))
                ask_exit = float(payload.get("ask", mid_exit))

                if mid_exit <= 0:
                    continue

                # Executable Exit Price:
                # To exit BUY, sell at Bid - slippage
                # To exit SELL, buy at Ask + slippage
                if decision == "BUY":
                    exec_exit = bid_exit - (self.slippage_rate * mid_exit)
                    gross_ret = (exec_exit - entry_px) / entry_px
                    comm_cost = self.commission_rate
                    fin_cost = self.financing_rate
                    tot_costs = comm_cost + fin_cost
                    net_ret = gross_ret - tot_costs
                elif decision == "SELL":
                    exec_exit = ask_exit + (self.slippage_rate * mid_exit)
                    gross_ret = -(exec_exit - entry_px) / entry_px
                    comm_cost = self.commission_rate
                    fin_cost = self.financing_rate
                    tot_costs = comm_cost + fin_cost
                    net_ret = gross_ret - tot_costs
                else:
                    exec_exit = mid_exit
                    gross_ret = 0.0
                    tot_costs = 0.0
                    net_ret = 0.0

                # Spread cost calculation for reporting
                half_spread_entry = abs(entry_px - float(p["reference_price"]))
                half_spread_exit = abs(exec_exit - mid_exit)
                spread_cost_total = (half_spread_entry + half_spread_exit) / float(p["reference_price"])

                outcome = self.store.resolve_prediction(
                    prediction_id=pid,
                    observed_exit_price=round(exec_exit, 5),
                    resolution_timestamp=now_dt.isoformat(),
                    data_source_status="VALID_OBSERVATION",
                    data_source_id=str(payload.get("source", "TRADINGVIEW_FX_LIVE")),
                    spread_cost=round(spread_cost_total, 6),
                    commission_cost=round(self.commission_rate + self.financing_rate, 6),
                    slippage_cost=round(self.slippage_rate * 2, 6),
                    spread_embedded_in_price=True,
                    notes="Resolved via live broker quote with executable bid/ask fill modeling.",
                )

                if outcome:
                    resolved_count += 1
                    with self._status_lock:
                        self._last_resolution_time = now_dt.isoformat()
                    logger.info(f"Resolved paper prediction {pid} for {symbol}: {outcome.outcome} (NetReturn={net_ret*100:+.3f}%)")

        return resolved_count

    def _get_provider_for_instrument(self, symbol: str) -> tuple[Any, str, str, bool]:
        """Resolves market provider, clean symbol, market name, and research proxy flag.

        Returns:
            (provider, clean_symbol, market, is_research_proxy)
        """
        sym_upper = symbol.strip().upper()
        if sym_upper.startswith("NSE:"):
            return self.nse_provider, sym_upper[4:], "NSE", False
        elif sym_upper.startswith("BSE:"):
            return self.bse_provider, sym_upper[4:], "BSE", False
        elif sym_upper.startswith("QUOTEX:"):
            return self.quotex_provider, sym_upper[7:], "QUOTEX", True
        elif sym_upper in ("EURUSD", "EURCHF", "EURAUD", "EURCAD") or sym_upper.startswith("FX:"):
            clean = sym_upper.replace("FX:", "")
            return self.fx_provider, clean, "FOREX", True

        resolved = SymbolResolver.resolve(symbol)
        if resolved:
            mkt = resolved.market
            clean_sym = resolved.symbol
            if mkt == MarketType.NSE:
                return self.nse_provider, clean_sym, "NSE", False
            elif mkt == MarketType.BSE:
                return self.bse_provider, clean_sym, "BSE", False
            elif mkt == MarketType.QUOTEX:
                return self.quotex_provider, clean_sym, "QUOTEX", True
            elif mkt in (MarketType.CRYPTO, MarketType.FOREX):
                return self.fx_provider, clean_sym, "FOREX", True

        return self.nse_provider, sym_upper, "NSE", False

    def _load_candles_for_instrument(self, symbol: str) -> list[dict[str, Any]]:
        provider, clean_sym, market, is_proxy = self._get_provider_for_instrument(symbol)
        if market in ("NSE", "BSE", "QUOTEX"):
            try:
                candles = provider.fetch_historical_candles_sync(clean_sym, timeframe="5m", limit=60)
                if candles and len(candles) >= 30:
                    return [
                        {
                            "timestamp": c.timestamp.isoformat(),
                            "open": c.open,
                            "high": c.high,
                            "low": c.low,
                            "close": c.close,
                            "volume": c.volume,
                            "is_closed": getattr(c, "is_closed", True),
                        }
                        for c in candles
                    ]
            except Exception as exc:
                logger.debug(f"Live historical candle fetch failed for {symbol}: {exc}")

        # Fallback to local cache for Forex / offline testing
        if self.fx_cache_path.exists():
            try:
                with open(self.fx_cache_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                return data.get(symbol, [])
            except Exception:
                return []
        return []

    def _update_instrument_telemetry(self, symbol: str, data: dict[str, Any]) -> None:
        with self._status_lock:
            self._instrument_status[symbol] = data

    def get_status(self) -> dict[str, Any]:
        """Returns live telemetry, worker state, metrics, and instrument diagnostics for this instance."""
        with self._status_lock:
            summary = self.store.get_summary_metrics()
            uptime = (time.time() - self._start_time) if self._start_time else 0.0
            return {
                "worker_state": self._state.value,
                "process_id": os.getpid(),
                "lock_acquired": self.lock._acquired,
                "uptime_seconds": round(uptime, 1),
                "model_version": "v20261020_044046",
                "model_sha256": FROZEN_MODEL_SHA256,
                "config_hash": FROZEN_CONFIG_HASH,
                "policy_version": EVALUATION_POLICY_VERSION,
                "last_heartbeat": self._last_heartbeat,
                "last_successful_cycle": self._last_successful_cycle,
                "last_prediction_time": self._last_prediction_time,
                "last_prediction_time_by_symbol": dict(self._last_prediction_time_by_symbol),
                "last_resolution_time": self._last_resolution_time,
                "last_data_time": dict(self._last_data_time),
                "last_completed_candle": dict(self._last_completed_candle),
                "candle_freshness": dict(self._candle_freshness),
                "candle_age_hours": dict(self._candle_age_hours),
                "clock_skew_seconds": dict(self._clock_skew_seconds),
                "is_eligible_to_generate": dict(self._is_eligible_to_generate),
                "overall_eligible": any(self._is_eligible_to_generate.values()) if self._is_eligible_to_generate else False,
                "last_error": self._last_error,
                "consecutive_errors": self._consecutive_errors,
                "retry_count": self._retry_count,
                "recent_exceptions": list(self._recent_exceptions[-10:]),
                "store_metrics": summary,
                "instruments": dict(self._instrument_status),
            }

    @classmethod
    def get_operational_status(cls, store_dir: Path | str = DEFAULT_PAPER_STORE_DIR) -> dict[str, Any]:
        """Provides full operational diagnostics without requiring a running worker instance in this process.

        Guarantees:
        - If kernel lock is NOT held, worker_state is truthfully reported as STOPPED.
        - Process ownership, PID, and lock file state are verified.
        - Store counts (pending, resolved, deferred, expired, invalid) are calculated directly.
        - Model and configuration hashes are verified.
        """
        store_path = Path(store_dir)
        lock_path = store_path / ".paper_worker.lock"
        heartbeat_path = store_path / "worker_heartbeat.json"

        # 1. Check lock status via OS kernel inspection
        lock_info = ProcessFileLock.check_lock_status(lock_path)
        is_locked = lock_info["locked"]

        # 2. Read heartbeat if exists
        hb_data: dict[str, Any] = {}
        if heartbeat_path.exists():
            try:
                hb_data = json.loads(heartbeat_path.read_text(encoding="utf-8"))
            except Exception:
                pass

        # 3. Determine truthful worker state
        if not is_locked:
            worker_state = WorkerState.STOPPED.value
            active_pid = None
            started_at = None
        else:
            worker_state = hb_data.get("worker_state", WorkerState.RUNNING.value)
            active_pid = hb_data.get("process_id") or lock_info.get("pid")
            started_at = hb_data.get("start_time") or lock_info.get("started_at")

        # 4. Get store metrics directly
        try:
            store = PaperTradingPredictionStore(store_dir=store_path)
            store_metrics = store.get_summary_metrics()
        except Exception as e:
            store_metrics = {"error": str(e)}

        # 5. Check model hashes
        model_hash_ok = False
        meta_hash_ok = False
        try:
            import hashlib
            if FROZEN_MODEL_PATH.exists():
                m_h = hashlib.sha256(FROZEN_MODEL_PATH.read_bytes()).hexdigest()
                model_hash_ok = (m_h == FROZEN_MODEL_SHA256)
            if FROZEN_META_PATH.exists():
                cfg_h = hashlib.sha256(FROZEN_META_PATH.read_bytes()).hexdigest()
                meta_hash_ok = (cfg_h == FROZEN_CONFIG_HASH)
        except Exception:
            pass

        # Extract or compute candle freshness and age
        now_dt = datetime.now(UTC)
        completed_candles = hb_data.get("last_completed_candle", {})
        candle_freshness = dict(hb_data.get("candle_freshness", {}))
        candle_age_hours = dict(hb_data.get("candle_age_hours", {}))
        for sym, c_str in completed_candles.items():
            if sym not in candle_age_hours and c_str:
                try:
                    c_dt = pd.to_datetime(c_str).to_pydatetime()
                    if c_dt.tzinfo is None:
                        c_dt = c_dt.replace(tzinfo=UTC)
                    age_h = (now_dt - c_dt).total_seconds() / 3600.0
                    candle_age_hours[sym] = round(age_h, 1)
                    candle_freshness[sym] = bool(0.0 <= age_h <= 84.0)
                except Exception:
                    candle_age_hours[sym] = 999.0
                    candle_freshness[sym] = False

        return {
            "worker_state": worker_state,
            "process_ownership": {
                "lock_acquired": is_locked,
                "lock_file": str(lock_path),
                "active_pid": active_pid,
                "lock_started_at": started_at,
                "lock_status": lock_info.get("status"),
            },
            "uptime_seconds": hb_data.get("uptime_seconds", 0.0) if is_locked else 0.0,
            "last_heartbeat": hb_data.get("last_heartbeat") if is_locked else None,
            "last_successful_cycle": hb_data.get("last_successful_cycle"),
            "provider_status": {
                "active_provider": "ForexDataProvider (TradingView FX Scanner & ECB Daily Reference)",
                "governance_allowlist": list(ACTIVE_GOVERNANCE_ALLOWLIST),
                "explicit_blocklist": list(EXPLICIT_BLOCKLIST),
                "verified_instruments": list(VERIFIED_FX_INSTRUMENTS),
                "last_data_time": hb_data.get("last_data_time", {}),
                "clock_skews_seconds": hb_data.get("clock_skew_seconds", {}),
            },
            "last_completed_candle": completed_candles,
            "candle_freshness": candle_freshness,
            "candle_age_hours": candle_age_hours,
            "last_prediction_time": hb_data.get("last_prediction_time_by_symbol", {}),
            "store_metrics": store_metrics,
            "last_outcome_resolution_time": hb_data.get("last_resolution_time"),
            "model_governance": {
                "model_version": "v20261020_044046",
                "model_sha256": FROZEN_MODEL_SHA256,
                "model_hash_verified": model_hash_ok,
                "config_hash": FROZEN_CONFIG_HASH,
                "config_hash_verified": meta_hash_ok,
                "policy_version": EVALUATION_POLICY_VERSION,
                "trading_mode": "PAPER_TRADING_ONLY (REAL TRADING PERMANENTLY DISABLED)",
            },
            "resilience_telemetry": {
                "retry_count": hb_data.get("retry_count", 0),
                "consecutive_errors": hb_data.get("consecutive_errors", 0),
                "last_error": hb_data.get("last_error"),
                "recent_exceptions": hb_data.get("recent_exceptions", []),
            },
            "is_currently_eligible_to_generate": hb_data.get("overall_eligible", False) if is_locked else False,
        }

    @classmethod
    def inspect_system_health(cls, store_dir: Path | str = DEFAULT_PAPER_STORE_DIR) -> dict[str, Any]:
        """Provides comprehensive multi-market paper trading system health diagnostics and audit metrics."""
        from paper_trading.health_report import generate_daily_health_report

        return generate_daily_health_report(store_dir=store_dir or DEFAULT_PAPER_STORE_DIR)
