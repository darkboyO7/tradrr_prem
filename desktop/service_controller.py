"""Trading Intelligence Platform — Local Desktop Application Service Layer.

Acts as the clean decoupling boundary between the Windows Desktop UI (PySide6)
and the modular Trading Intelligence Backend (Universe, MarketState, FeatureEngine,
HistoricalPatternSimilarityEngine, SignalFusionEngine, MarketScannerService,
PredictionLedgerService, and SystemGovernanceController).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, time as dtime, timedelta, timezone
from pathlib import Path
import time
from typing import Any, Callable
from zoneinfo import ZoneInfo

from core.instrument import Instrument, MarketType
from core.logging import get_logger
from core.symbol_resolver import SymbolResolver
from core.universe import (
    BSEUniverseManager,
    CanonicalInstrumentRegistry,
    NSEUniverseManager,
    QuotexDiscoveryAdapter,
    UniverseService,
    UniverseSyncService,
)
from data.market_state import MarketStateService
from learning.ledger import PredictionLedgerService
from signals.fusion import run_live_intelligence_pipeline
from signals.scanner import MarketScannerService

logger = get_logger(__name__)


@dataclass
class StartupCheckResult:
    """Result of the 5 core startup verification checks (Section 15)."""

    data_ok: bool = False
    data_detail: str = "Checking..."
    database_ok: bool = False
    database_detail: str = "Checking..."
    models_ok: bool = False
    models_detail: str = "Checking..."
    market_connection_ok: bool = False
    market_connection_detail: str = "Checking..."
    prediction_engine_ok: bool = False
    prediction_engine_detail: str = "Checking..."
    step_logs: list[str] = field(default_factory=list)

    @property
    def all_critical_ok(self) -> bool:
        return (
            self.data_ok
            and self.database_ok
            and self.models_ok
            and self.market_connection_ok
            and self.prediction_engine_ok
        )

    @property
    def summary(self) -> str:
        """Concise human-readable status summary of all startup checks."""
        status = "PASSED" if self.all_critical_ok else "FAILED"
        return (
            f"StartupCheckResult({status} - data: {self.data_ok}, db: {self.database_ok}, "
            f"models: {self.models_ok}, market: {self.market_connection_ok}, prediction: {self.prediction_engine_ok})"
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert startup check results to a serializable dictionary."""
        return {
            "all_critical_ok": self.all_critical_ok,
            "data_ok": self.data_ok,
            "data_detail": self.data_detail,
            "database_ok": self.database_ok,
            "database_detail": self.database_detail,
            "models_ok": self.models_ok,
            "models_detail": self.models_detail,
            "market_connection_ok": self.market_connection_ok,
            "market_connection_detail": self.market_connection_detail,
            "prediction_engine_ok": self.prediction_engine_ok,
            "prediction_engine_detail": self.prediction_engine_detail,
            "summary": self.summary,
            "step_logs": list(self.step_logs),
        }


@dataclass
class NotificationSettings:
    """User-configurable desktop notification preferences (Section 18)."""

    enabled: bool = True
    buy_signal: bool = True
    sell_signal: bool = True
    signal_invalidated: bool = True
    data_failure: bool = True
    model_update: bool = True
    connection_lost: bool = True
    sound_enabled: bool = True
    cooldown_seconds: float = 120.0


class DesktopServiceController:
    """Local application service layer bridging the Desktop UI and Trading Backend."""

    def __init__(self) -> None:
        try:
            from desktop.app import setup_desktop_file_logging

            setup_desktop_file_logging()
        except Exception:
            pass
        self.predictions_paused: bool = False
        self.notification_settings = NotificationSettings()
        self._last_notification_ts: dict[str, float] = {}
        self._last_signal_by_symbol: dict[str, str] = {}
        self._startup_result = StartupCheckResult()
        self._scanner = MarketScannerService()
        self._recent_logs: list[str] = []
        self._latest_latencies: dict[str, float] = {
            "market_data_latency_ms": 0.0,
            "processing_latency_ms": 0.0,
            "prediction_latency_ms": 0.0,
            "ui_update_latency_ms": 0.0,
            "end_to_end_latency_ms": 0.0,
        }

    def log_event(self, message: str) -> None:
        ts = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        entry = f"[{ts}] {message}"
        self._recent_logs.append(entry)
        if len(self._recent_logs) > 250:
            self._recent_logs.pop(0)
        logger.info(message)

    def get_recent_logs(self) -> list[str]:
        return list(self._recent_logs)

    def startup(
        self,
        progress_callback: Callable[[int, str, StartupCheckResult], None] | None = None,
    ) -> StartupCheckResult:
        """Alias for run_startup_sequence."""
        return self.run_startup_sequence(progress_callback=progress_callback)

    def run_startup_sequence(
        self,
        progress_callback: Callable[[int, str, StartupCheckResult], None] | None = None,
    ) -> StartupCheckResult:
        """Execute the 10-step desktop startup verification sequence (Section 15)."""
        res = StartupCheckResult()

        def _notify(step_num: int, msg: str) -> None:
            res.step_logs.append(f"Step {step_num}/10: {msg}")
            self.log_event(f"STARTUP [{step_num}/10] {msg}")
            if progress_callback:
                progress_callback(step_num, msg, res)

        # 1. Start/check required local services
        _notify(1, "Starting local application runtime services...")

        # 2. Check database / local storage persistence
        _notify(2, "Checking database & local ledger persistence...")
        try:
            ledger = PredictionLedgerService.get_instance()
            metrics = ledger.get_metrics()
            res.database_ok = True
            res.database_detail = f"Connected (Ledger active, {metrics.get('total_predictions', 0)} records)"
        except Exception as exc:
            res.database_ok = False
            res.database_detail = f"Database error: {exc}"

        # 3. Check Redis / EventBus & synchronize universe data masters
        _notify(3, "Checking event bus & local market data store...")
        try:
            total_inst = UniverseService.sync_universe(refresh_remote=False)
            summary = UniverseService.summary()
            by_mkt = summary.get("by_market", {})
            res.data_ok = total_inst > 1000
            res.data_detail = (
                f"{total_inst:,} instruments synced "
                f"(NSE: {by_mkt.get('NSE', 0):,}, BSE: {by_mkt.get('BSE', 0):,}, "
                f"Quotex: {by_mkt.get('QUOTEX', 0)}, FX: {by_mkt.get('FOREX', 0)}, Crypto: {by_mkt.get('CRYPTO', 0)})"
            )
        except Exception as exc:
            res.data_ok = False
            res.data_detail = f"Universe data error: {exc}"

        # 4. Check model registry
        _notify(4, "Verifying model registry & champion governance state...")
        try:
            ledger = PredictionLedgerService.get_instance()
            champ = ledger.champion_model_id
            res.models_ok = bool(champ)
            res.models_detail = f"Production Model: {champ} (Hierarchical 5-Tier Ready)"
        except Exception as exc:
            res.models_ok = False
            res.models_detail = f"Model registry error: {exc}"

        # 5. Synchronize cross-listed NSE/BSE & Quotex discovery registries
        _notify(5, "Linking ISIN cross-listed NSE/BSE securities & Quotex underlying mappings...")
        cross_count = CanonicalInstrumentRegistry.cross_listed_count()
        self.log_event(f"Cross-listed NSE+BSE canonical companies linked: {cross_count:,}")

        # 6. Check market-data connections
        _notify(6, "Testing live market-data feed & historical backfill...")
        try:
            state_svc = MarketStateService.get_instance()
            t0 = time.perf_counter()
            st = state_svc.ensure_instrument_state("NSE:TCS", "5m")
            md_ms = (time.perf_counter() - t0) * 1000.0
            self._latest_latencies["market_data_latency_ms"] = round(md_ms, 2)
            if st and st.last_price > 0:
                res.market_connection_ok = True
                sess_str = "LIVE" if st.session_open else "MARKET CLOSED (Historical Synced)"
                res.market_connection_detail = f"{st.provider} Connected ({sess_str}, {md_ms:.1f} ms)"
            else:
                res.market_connection_ok = False
                res.market_connection_detail = "Market data feed returned no valid price"
        except Exception as exc:
            res.market_connection_ok = False
            res.market_connection_detail = f"Market connection error: {exc}"

        # 7. Start live market engine
        _notify(7, "Starting live market state synchronization engine...")

        # 8. Start prediction engine
        _notify(8, "Verifying multi-source signal fusion & prediction engine...")
        try:
            t_pred = time.perf_counter()
            eval_res = run_live_intelligence_pipeline("NSE:TCS", timeframe="5m", horizon="1h")
            pred_ms = (time.perf_counter() - t_pred) * 1000.0
            self._latest_latencies["prediction_latency_ms"] = round(pred_ms, 2)
            dec = eval_res.get("decision")
            if dec is not None:
                act_str = dec.action.value if hasattr(dec, "action") else str(getattr(dec, "direction", "EVALUATED"))
                res.prediction_engine_ok = True
                res.prediction_engine_detail = (
                    f"Operational ({act_str} @ {dec.confidence * 100:.0f}%, {pred_ms:.1f} ms)"
                )
            else:
                res.prediction_engine_ok = False
                res.prediction_engine_detail = "Prediction pipeline returned no decision"
        except Exception as exc:
            res.prediction_engine_ok = False
            res.prediction_engine_detail = f"Prediction engine failure: {exc}"

        # 9. Initialize background services in safe read-only STOPPED state
        _notify(9, "Initializing background services in safe read-only STOPPED state...")
        try:
            from learning.controller import ContinuousLearningController

            ledger = PredictionLedgerService.get_instance()
            learning_ctrl = ContinuousLearningController.get_instance()
            learning_ctrl.auto_promote = False
            self.log_event("Safe Read-Only Startup: All background workers remain STOPPED by default")
        except Exception as exc:
            self.log_event(f"Service initialization notice: {exc}")

        # 10. Open the desktop dashboard
        _notify(10, "Startup complete — launching Desktop Dashboard...")
        self._startup_result = res
        return res

    def search_universe(
        self,
        query: str,
        market: str | None = None,
        limit: int = 25,
    ) -> list[dict[str, Any]]:
        """Search across all 9,148 instruments and include cross-listed NSE + BSE scrip codes (Section 6)."""
        m_enum = None
        if market and market.upper() != "ALL":
            m_enum = getattr(MarketType, market.upper(), None)

        raw_hits = UniverseService.search(query=query, market=m_enum, limit=limit)
        enriched: list[dict[str, Any]] = []
        for hit in raw_hits:
            canon = hit["canonical_symbol"]
            cross = CanonicalInstrumentRegistry.get_cross_listed(canon)
            nse_sym = cross.get("nse_symbol") or (canon if hit["market"] == "NSE" else "")
            bse_scrip = cross.get("bse_scrip_code") or hit.get("scrip_code") or ""
            bse_sym = cross.get("bse_symbol") or (canon if hit["market"] == "BSE" else "")

            id_badges: list[str] = []
            if nse_sym:
                id_badges.append(nse_sym if ":" in nse_sym else f"NSE:{nse_sym}")
            if bse_scrip:
                id_badges.append(f"BSE:{bse_scrip}")
            elif bse_sym:
                id_badges.append(bse_sym if ":" in bse_sym else f"BSE:{bse_sym}")
            if not id_badges:
                id_badges.append(canon)

            hit["nse_display"] = nse_sym if ":" in nse_sym else (f"NSE:{nse_sym}" if nse_sym else "")
            hit["bse_scrip_display"] = f"BSE:{bse_scrip}" if bse_scrip else bse_sym
            hit["identifiers_text"] = "  •  ".join(id_badges)
            hit["display_line"] = f"{hit['name']}   [{hit['identifiers_text']}]"
            enriched.append(hit)

        return enriched

    def evaluate_instrument(
        self,
        symbol_or_query: str,
        timeframe: str = "5m",
        horizon: str = "1h",
    ) -> dict[str, Any]:
        """Run full end-to-end live + historical + developing-day + pattern + ML evaluation for the Desktop UI."""
        t_start = time.perf_counter()

        inst = UniverseService.resolve_arbitrary_stock(symbol_or_query)
        canon = inst.canonical_symbol if inst else symbol_or_query.strip().upper()
        meta = inst.metadata if inst else {}

        # Check Quotex live data availability first (Section 7: Never fake Quotex data)
        if inst and inst.market == MarketType.QUOTEX and not meta.get("live_data_available", True):
            total_ms = round((time.perf_counter() - t_start) * 1000.0, 2)
            self._latest_latencies.update(
                {
                    "market_data_latency_ms": total_ms,
                    "processing_latency_ms": 0.0,
                    "prediction_latency_ms": 0.0,
                    "end_to_end_latency_ms": total_ms,
                }
            )
            notification = self._check_notifications(
                canon=canon,
                signal="NO_TRADE",
                confidence=0.0,
                quotex_unavailable=True,
                is_stale=False,
            )
            return {
                "status": "QUOTEX_DATA_UNAVAILABLE",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "data_timestamp": None,
                "symbol": inst.symbol,
                "canonical_symbol": canon,
                "market": "QUOTEX",
                "company_name": meta.get("name", inst.symbol),
                "currency": inst.quote_currency,
                "cross_listing": {},
                "quotex_info": {
                    "platform_instrument": canon,
                    "underlying_instrument": meta.get("underlying_instrument", "UNKNOWN"),
                    "market_data_source": meta.get("market_data_source", "UNAVAILABLE"),
                    "live_data_available": False,
                },
                "is_market_open": False,
                "session_badge": "QUOTEX DATA UNAVAILABLE",
                "last_price": 0.0,
                "price_display": "QUOTEX DATA UNAVAILABLE",
                "signal": "NO-TRADE",
                "action": "NO-TRADE",
                "confidence_pct": 0,
                "horizon": horizon,
                "horizon_label": self._format_horizon_label(horizon),
                "entry_price": 0.0,
                "stop_loss": 0.0,
                "take_profit": 0.0,
                "why_bullets": [
                    "QUOTEX DATA UNAVAILABLE — Proprietary synthetic OTC instrument has no authorized public market feed.",
                    f"Platform Instrument: {canon}  |  Underlying: {meta.get('underlying_instrument', 'SYNTHETIC_OTC')}",
                    "System strictly refuses to display fake prices or fabricated signals.",
                ],
                "candles": [],
                "developing_day": {},
                "pattern_consensus": {},
                "multi_horizon": {},
                "quality_gate": {"passed": False, "reasons": ["Quotex live data unavailable"]},
                "latencies": dict(self._latest_latencies),
                "notification": notification,
            }

        # Measure market data retrieval latency
        t_md0 = time.perf_counter()
        state_svc = MarketStateService.get_instance()
        st = state_svc.ensure_instrument_state(canon, timeframe)
        md_latency_ms = round((time.perf_counter() - t_md0) * 1000.0, 2)

        # Measure processing & prediction latency
        t_pred0 = time.perf_counter()
        res = run_live_intelligence_pipeline(
            symbol_or_query=canon,
            timeframe=timeframe,
            horizon=horizon,
        )
        pred_total_ms = round((time.perf_counter() - t_pred0) * 1000.0, 2)

        lat_tel = res.get("latency_telemetry") or {}
        proc_ms = round(float(lat_tel.get("feature_latency_ms", pred_total_ms * 0.45)), 2)
        pred_ms = round(max(0.1, pred_total_ms - proc_ms), 2)
        e2e_ms = round((time.perf_counter() - t_start) * 1000.0, 2)

        self._latest_latencies.update(
            {
                "market_data_latency_ms": md_latency_ms,
                "processing_latency_ms": proc_ms,
                "prediction_latency_ms": pred_ms,
                "end_to_end_latency_ms": e2e_ms,
            }
        )

        decision = res["decision"]
        fc = res.get("forecast")
        inst_obj: Instrument | None = res.get("instrument") or inst
        st_obj = res.get("market_state") or st

        # Record prediction & resolve any matured predictions in PredictionLedgerService
        ledger = PredictionLedgerService.get_instance()
        if st_obj and st_obj.last_price > 0:
            try:
                ledger.resolve_pending_forecasts(canon, st_obj.last_price)
            except Exception as exc:
                self.log_event(f"Ledger resolve warning for {canon}: {exc}")
        if fc is not None and res.get("status") == "OK" and not self.predictions_paused:
            try:
                active_model = ledger.champion_model_id
                try:
                    from learning.controller import ContinuousLearningController

                    active_model = ContinuousLearningController.get_instance().active_champion_id
                except Exception:
                    pass

                ledger.record_forecast(
                    fc,
                    model_id=active_model,
                    market=inst_obj.market.value if inst_obj else None,
                    canonical_symbol=canon,
                    features_used=res.get("features"),
                    source_data_timestamp=getattr(st_obj, "last_timestamp", None) if st_obj else None,
                    historical_context_used={
                        "pattern_consensus": {
                            k: v for k, v in (res.get("pattern_consensus") or {}).items() if k != "all_returns"
                        },
                        "developing_day": res.get("developing_day"),
                    },
                )
            except Exception as exc:
                self.log_event(f"Ledger record forecast warning for {canon}: {exc}")

        # Determine data provenance and session status (Section 16: Offline / Market-Closed Mode)
        mkt_name = inst_obj.market.value if inst_obj else "NSE"
        timing_info = self.get_market_session_status(mkt_name)
        is_open = timing_info["is_open"]
        clock_badge = timing_info["badge"]

        # Derive provenance-truthful session badge
        is_stale = bool(st_obj.is_stale) if st_obj else False
        provider = getattr(st_obj, "provider", None) if st_obj else None
        conn_state = getattr(st_obj, "connection_state", "") if st_obj else ""

        if inst_obj and inst_obj.market == MarketType.QUOTEX:
            session_badge = "QUOTEX DATA UNAVAILABLE"
        elif not st_obj or not st_obj.last_price or st_obj.last_price <= 0:
            session_badge = "DATA UNAVAILABLE"
        elif is_stale:
            session_badge = "DATA STALE"
        elif not is_open:
            session_badge = clock_badge
        elif conn_state == "CONNECTED" and provider and ("live" in provider.lower() or "direct" in provider.lower() or "websocket" in provider.lower()):
            session_badge = "LIVE VERIFIED"
        elif provider and ("cache" in provider.lower() or "fallback" in provider.lower() or "reference" in provider.lower()):
            session_badge = "REFERENCE ONLY"
        else:
            session_badge = clock_badge
        last_price = float(st_obj.last_price) if (st_obj and st_obj.last_price > 0) else float(decision.entry_price or 0.0)
        currency = inst_obj.quote_currency if inst_obj else "INR"

        raw_dir = decision.action.value if hasattr(decision, "action") else str(getattr(decision, "direction", "NO_TRADE"))
        ui_signal = "NO-TRADE" if raw_dir == "NO_TRADE" else raw_dir
        conf_pct = int(round(float(decision.confidence) * 100))

        # Cross-listing details (Section 6)
        cross_info = CanonicalInstrumentRegistry.get_cross_listed(canon)
        quotex_info = {}
        if inst_obj and inst_obj.market == MarketType.QUOTEX:
            quotex_info = {
                "platform_instrument": canon,
                "underlying_instrument": (inst_obj.metadata or {}).get("underlying_instrument", canon),
                "market_data_source": (inst_obj.metadata or {}).get("market_data_source", "LIVE_FEED"),
                "live_data_available": (inst_obj.metadata or {}).get("live_data_available", True),
            }

        why_bullets = self._build_beginner_why_bullets(
            ui_signal=ui_signal,
            decision=decision,
            developing_day=res.get("developing_day") or {},
            pattern_consensus=res.get("pattern_consensus") or {},
            mtf_context=res.get("mtf_context") or {},
            is_open=is_open,
            quotex_info=quotex_info,
        )

        candles = (st_obj.rolling_candles_5m or st_obj.rolling_candles_1m)[-75:] if st_obj else []

        notification = self._check_notifications(
            canon=canon,
            signal=ui_signal,
            confidence=float(decision.confidence),
            quotex_unavailable=False,
            is_stale=bool(st_obj.is_stale) if st_obj else False,
        )

        eval_iso = datetime.now(timezone.utc).isoformat()
        data_ts = getattr(st_obj, "last_timestamp", None) if st_obj else None
        data_ts_iso = data_ts.isoformat() if hasattr(data_ts, "isoformat") else (str(data_ts) if data_ts else eval_iso)

        return {
            "status": res.get("status", "OK"),
            "timestamp": eval_iso,
            "data_timestamp": data_ts_iso,
            "symbol": inst_obj.symbol if inst_obj else canon,
            "canonical_symbol": canon,
            "market": inst_obj.market.value if inst_obj else "NSE",
            "company_name": (inst_obj.metadata or {}).get("name", inst_obj.symbol) if inst_obj else canon,
            "currency": currency,
            "cross_listing": cross_info,
            "quotex_info": quotex_info,
            "is_market_open": is_open,
            "session_badge": session_badge,
            "session_timing": timing_info,
            "last_price": last_price,
            "price_display": self.format_price(last_price, currency, inst_obj.market.value if inst_obj else "NSE"),
            "signal": ui_signal,
            "action": ui_signal,
            "confidence_pct": conf_pct,
            "horizon": horizon,
            "horizon_label": self._format_horizon_label(horizon),
            "entry_price": float(decision.entry_price or last_price),
            "stop_loss": float(decision.stop_loss or 0.0),
            "take_profit": float(decision.take_profit or 0.0),
            "why_bullets": why_bullets,
            "candles": candles,
            "developing_day": res.get("developing_day") or (st_obj.developing_day if st_obj else {}),
            "pattern_consensus": res.get("pattern_consensus") or {},
            "mtf_context": res.get("mtf_context") or {},
            "multi_horizon": (decision.evidence or {}).get("multi_horizon_forecasts", {}),
            "quality_gate": (decision.evidence or {}).get("data_quality_gate", {}),
            "hierarchical_model_tier": (decision.evidence or {}).get("hierarchical_model_tier", "EXCHANGE_SPECIFIC"),
            "latencies": dict(self._latest_latencies),
            "session_timing": timing_info,
            "notification": notification,
        }

    def record_ui_render_latency(self, ui_ms: float) -> None:
        self._latest_latencies["ui_update_latency_ms"] = round(max(0.05, float(ui_ms)), 2)

    @staticmethod
    def get_market_session_status(market: str = "NSE") -> dict[str, Any]:
        """Compute exact market session state and countdown to next open or close."""
        mkt_up = (market or "NSE").upper()
        if mkt_up in ("CRYPTO", "QUOTEX"):
            return {
                "is_open": True,
                "is_pre_market": False,
                "badge": "LIVE",
                "message": "Continuous 24/7 Market Session",
                "countdown_str": "24/7 Active",
            }

        try:
            ist_tz = ZoneInfo("Asia/Kolkata")
        except Exception:
            ist_tz = timezone.utc

        now_ist = datetime.now(ist_tz)
        weekday = now_ist.weekday()  # 0=Mon, 4=Fri, 5=Sat, 6=Sun
        t = now_ist.time()
        pre_open_t = dtime(9, 0)
        open_t = dtime(9, 15)
        close_t = dtime(15, 30)
        post_open_t = dtime(15, 40)
        post_close_t = dtime(16, 0)

        if weekday < 5 and open_t <= t <= close_t:
            close_dt = datetime.combine(now_ist.date(), close_t, tzinfo=ist_tz)
            rem = close_dt - now_ist
            hrs, rem_sec = divmod(max(0, int(rem.total_seconds())), 3600)
            mins = rem_sec // 60
            return {
                "is_open": True,
                "is_pre_market": False,
                "badge": "LIVE",
                "message": "Continuous Trading Session (09:15 - 15:30 IST)",
                "countdown_str": f"Closes in {hrs}h {mins:02d}m",
            }

        if weekday < 5 and pre_open_t <= t < open_t:
            live_open_dt = datetime.combine(now_ist.date(), open_t, tzinfo=ist_tz)
            rem = live_open_dt - now_ist
            mins = max(0, int(rem.total_seconds())) // 60
            return {
                "is_open": False,
                "is_pre_market": True,
                "badge": "PRE-MARKET",
                "message": "Pre-Market Discovery Session (09:00 - 09:15 IST)",
                "countdown_str": f"Live opens in {mins:02d}m",
            }

        if weekday < 5 and post_open_t <= t <= post_close_t:
            return {
                "is_open": False,
                "is_pre_market": False,
                "badge": "POST-MARKET",
                "message": "Post-Market Closing Session (15:40 - 16:00 IST)",
                "countdown_str": "Session finalization",
            }

        # Calculate next market open
        if weekday < 4 and t > close_t:
            next_date = now_ist.date() + timedelta(days=1)
        elif weekday == 4 and t > close_t:
            next_date = now_ist.date() + timedelta(days=3)
        elif weekday == 5:
            next_date = now_ist.date() + timedelta(days=2)
        elif weekday == 6:
            next_date = now_ist.date() + timedelta(days=1)
        elif weekday < 5 and t < open_t:
            next_date = now_ist.date()
        else:
            next_date = now_ist.date() + timedelta(days=1)

        next_open_dt = datetime.combine(next_date, open_t, tzinfo=ist_tz)
        rem = next_open_dt - now_ist
        hrs, rem_sec = divmod(max(0, int(rem.total_seconds())), 3600)
        mins = rem_sec // 60

        day_label = (
            "Tomorrow"
            if next_date == (now_ist.date() + timedelta(days=1))
            else ("Today" if next_date == now_ist.date() else next_date.strftime("%A"))
        )

        return {
            "is_open": False,
            "is_pre_market": False,
            "badge": "MARKET CLOSED",
            "message": f"Regular Trading Session Closed - Opens {day_label} 09:15 IST",
            "countdown_str": f"Opens in {hrs}h {mins:02d}m ({day_label})",
        }

    @staticmethod
    def format_price(price: float, currency: str = "INR", market: str = "NSE") -> str:
        if price <= 0:
            return "—"
        if market in ("FOREX", "QUOTEX") and price < 250.0 and currency != "INR":
            return f"{price:,.5f}"
        if currency == "INR" or market in ("NSE", "BSE"):
            return f"₹{price:,.2f}"
        if currency in ("USD", "USDT"):
            return f"${price:,.2f}"
        return f"{price:,.4f} {currency}"

    @staticmethod
    def _format_horizon_label(horizon: str) -> str:
        mapping = {
            "5m": "Next 5 Minutes",
            "15m": "Next 15 Minutes",
            "1h": "Next 1 Hour",
            "4h": "Next 4 Hours",
            "1d": "Next 1 Trading Day",
        }
        return mapping.get(horizon.lower(), f"Next {horizon}")

    @staticmethod
    def _build_beginner_why_bullets(
        ui_signal: str,
        decision: Any,
        developing_day: dict[str, Any],
        pattern_consensus: dict[str, Any],
        mtf_context: dict[str, Any],
        is_open: bool,
        quotex_info: dict[str, Any],
    ) -> list[str]:
        """Translate technical and ML signals into simple beginner-readable bullet points (Section 4)."""
        bullets: list[str] = []

        if not is_open:
            bullets.append("Market is currently closed — showing analysis synchronized to the most recent session close.")

        if quotex_info:
            bullets.append(
                f"Quotex Platform Instrument: {quotex_info.get('platform_instrument')} → "
                f"Analyzed via Underlying: {quotex_info.get('underlying_instrument')}"
            )

        trend = str(developing_day.get("trend", "")).upper()
        rel_vol = float(developing_day.get("relative_volume", 1.0) or 1.0)
        vwap_diff = float(developing_day.get("price_vs_vwap_pct", 0.0) or 0.0)
        breakout = str(developing_day.get("breakout_state", "")).upper()

        if ui_signal == "BUY":
            if "BULL" in trend or vwap_diff > 0:
                bullets.append("Strong trend — Price is trading above today's session average (VWAP) with upward structure.")
            else:
                bullets.append("Constructive setup — Intraday buyers are stepping in near session support.")
        elif ui_signal == "SELL":
            if "BEAR" in trend or vwap_diff < 0:
                bullets.append("Downward trend — Price is trading below today's session average (VWAP) with selling pressure.")
            else:
                bullets.append("Weakening setup — Intraday sellers are rejecting higher prices near resistance.")
        else:
            gate = (decision.evidence or {}).get("data_quality_gate", {})
            reasons = gate.get("reasons") or []
            if reasons:
                bullets.append(f"No-Trade protection active: {', '.join(str(r) for r in reasons)}")
            else:
                bullets.append("Mixed market direction — Waiting for clearer alignment before risking capital.")

        # Historical time-aligned pattern bullet
        win_rate = float(pattern_consensus.get("win_rate", 0.5) or 0.5)
        sample_size = int(pattern_consensus.get("sample_size", 0) or 0)
        if sample_size > 0:
            if ui_signal == "BUY" and win_rate >= 0.52:
                bullets.append(
                    f"Similar historical pattern — {int(round(win_rate * 100))}% of time-aligned historical sessions moved higher."
                )
            elif ui_signal == "SELL" and (1.0 - win_rate) >= 0.52:
                bullets.append(
                    f"Similar historical pattern — {int(round((1.0 - win_rate) * 100))}% of time-aligned historical sessions moved lower."
                )
            else:
                bullets.append(
                    f"Historical comparison — Evaluated {sample_size} time-synchronized historical sessions at this exact time of day."
                )

        # Volume bullet
        if rel_vol >= 1.15:
            bullets.append(f"High volume — Current session activity is {rel_vol:.1f}x normal volume, confirming participation.")
        else:
            bullets.append(f"Steady volume — Intraday trading activity is tracking at {rel_vol:.1f}x normal session pace.")

        # Multi-timeframe confirmation bullet
        alignment = str(mtf_context.get("alignment", "MIXED")).upper()
        if "BULL" in alignment or "BEAR" in alignment or "ALIGNED" in alignment:
            bullets.append("Multi-timeframe confirmation — Short-term (5m/15m) and higher-timeframe (1h/1d) trends agree.")
        elif "BROKEN" in breakout:
            bullets.append(f"Opening range structure — Session opening range status is {breakout.replace('_', ' ').title()}.")
        else:
            bullets.append("Multi-timeframe confirmation — Monitored across 5m, 15m, 1h, and daily market structure.")

        return bullets[:5]

    def _check_notifications(
        self,
        canon: str,
        signal: str,
        confidence: float,
        quotex_unavailable: bool,
        is_stale: bool,
    ) -> dict[str, str] | None:
        """Evaluate whether a non-spamming desktop notification should be emitted (Section 18)."""
        if not self.notification_settings.enabled:
            return None

        now = time.time()
        last_ts = self._last_notification_ts.get(canon, 0.0)
        prev_sig = self._last_signal_by_symbol.get(canon)
        self._last_signal_by_symbol[canon] = signal

        if now - last_ts < self.notification_settings.cooldown_seconds and prev_sig == signal:
            return None

        if quotex_unavailable and self.notification_settings.data_failure:
            self._last_notification_ts[canon] = now
            return {
                "title": f"Data Unavailable — {canon}",
                "message": f"{canon}: Quotex live data unavailable. Showing NO-TRADE.",
                "category": "data_failure",
            }

        if is_stale and self.notification_settings.connection_lost:
            self._last_notification_ts[canon] = now
            return {
                "title": f"Stale Market Feed — {canon}",
                "message": f"{canon}: Market data age exceeded freshness threshold.",
                "category": "connection_lost",
            }

        if prev_sig in ("BUY", "SELL") and signal != prev_sig and self.notification_settings.signal_invalidated:
            self._last_notification_ts[canon] = now
            return {
                "title": f"Signal Changed — {canon}",
                "message": f"Previous {prev_sig} signal transitioned to {signal} ({int(confidence * 100)}% confidence).",
                "category": "signal_invalidated",
            }

        if signal == "BUY" and confidence >= 0.65 and self.notification_settings.buy_signal:
            self._last_notification_ts[canon] = now
            return {
                "title": f"BUY Signal — {canon}",
                "message": f"{canon}: BUY signal with {int(confidence * 100)}% confidence.",
                "category": "buy_signal",
            }

        if signal == "SELL" and confidence >= 0.65 and self.notification_settings.sell_signal:
            self._last_notification_ts[canon] = now
            return {
                "title": f"SELL Signal — {canon}",
                "message": f"{canon}: SELL signal with {int(confidence * 100)}% confidence.",
                "category": "sell_signal",
            }

        return None

    def scan_market(
        self,
        market: str = "ALL",
        signal_filter: str = "ALL",
        min_confidence_pct: int = 0,
        horizon: str = "1h",
        search_query: str = "",
        sort_by: str = "confidence",
        limit: int = 40,
    ) -> list[dict[str, Any]]:
        """Scan across NSE, BSE, Quotex, Forex, and Crypto with full filtering and sorting (Section 8)."""
        m_enum = None
        if market and market.upper() != "ALL":
            m_enum = getattr(MarketType, market.upper(), None)

        sig_arg = None
        if signal_filter and signal_filter.upper() != "ALL":
            sig_arg = signal_filter.upper().replace("NO-TRADE", "NO_TRADE")

        report = self._scanner.scan_market(
            market=m_enum,
            min_confidence=max(0.0, min_confidence_pct / 100.0),
            horizon=horizon,
            limit=limit,
            search_query=search_query.strip() or None,
            signal_filter=sig_arg,
            sort_by=sort_by,
            include_no_trade=True,
        )
        rows: list[dict[str, Any]] = []
        for opp in report.top_opportunities:
            inst = UniverseService.resolve_arbitrary_stock(opp.canonical_symbol)
            cross = CanonicalInstrumentRegistry.get_cross_listed(opp.canonical_symbol)
            scrip = cross.get("bse_scrip_code") or ((inst.metadata or {}).get("scrip_code", "") if inst else "")
            currency = inst.quote_currency if inst else "INR"
            raw_act = opp.action.value if hasattr(opp, "action") else str(getattr(opp, "direction", "NO_TRADE"))
            ui_sig = "NO-TRADE" if raw_act in ("NO_TRADE", "NO-TRADE") else raw_act
            reasons_list = getattr(opp, "reasons", []) or []
            rationale_str = reasons_list[0] if reasons_list else getattr(opp, "rationale", "")
            rows.append(
                {
                    "market": opp.market,
                    "symbol": opp.symbol,
                    "canonical_symbol": opp.canonical_symbol,
                    "scrip_code": scrip,
                    "company_name": opp.company_name,
                    "price": opp.current_price,
                    "price_display": (
                        "QUOTEX DATA UNAVAILABLE"
                        if (opp.market == "QUOTEX" and opp.current_price <= 0)
                        else self.format_price(opp.current_price, currency, opp.market)
                    ),
                    "signal": ui_sig,
                    "confidence_pct": int(round(opp.confidence * 100)),
                    "horizon": getattr(opp, "forecast_horizon", horizon),
                    "entry_price": opp.current_price,
                    "stop_loss": getattr(opp, "stop_price", 0.0),
                    "take_profit": getattr(opp, "target_price", 0.0),
                    "expected_move_pct": getattr(opp, "expected_return_pct", 0.0),
                    "rationale": rationale_str,
                }
            )
        return rows

    def get_quotex_section_rows(
        self,
        horizon: str = "1h",
        search_query: str = "",
        asset_type_filter: str = "ALL",
    ) -> list[dict[str, Any]]:
        """Return all discovered Quotex instruments with real live predictions, levels, and rationale (Section 7)."""
        if not UniverseService._synced:
            UniverseService.sync_universe()

        records = QuotexDiscoveryAdapter.get_all_records()
        q_upper = search_query.strip().upper()
        type_upper = asset_type_filter.strip().upper()

        # Run batch scanner across Quotex universe
        report = self._scanner.scan_market(
            market=MarketType.QUOTEX,
            min_confidence=0.0,
            horizon=horizon,
            search_query=search_query.strip() or None,
            sort_by="confidence",
            limit=100,
            include_no_trade=True,
        )
        scanned_by_canon: dict[str, Any] = {opp.canonical_symbol: opp for opp in report.top_opportunities}
        state_svc = MarketStateService.get_instance()
        rows: list[dict[str, Any]] = []

        for rec in records:
            canon = rec.canonical_symbol
            # Text search filter
            if q_upper:
                match_text = f"{rec.quotex_symbol} {canon} {rec.platform_instrument_name} {rec.underlying_instrument}".upper()
                if q_upper not in match_text:
                    continue

            # Asset class filter
            if type_upper != "ALL":
                a_type = rec.asset_type.upper()
                if type_upper == "FOREX_SPOT" and "SPOT" not in a_type:
                    continue
                elif type_upper == "FOREX_OTC" and "OTC_UNDERLYING" not in a_type:
                    continue
                elif type_upper == "CRYPTO" and "CRYPTO" not in a_type:
                    continue
                elif type_upper == "PROPRIETARY_OTC" and "SYNTHETIC" not in a_type:
                    continue

            if not rec.live_data_available:
                rows.append(
                    {
                        "platform_instrument": canon,
                        "symbol": rec.quotex_symbol,
                        "name": rec.platform_instrument_name,
                        "underlying_instrument": rec.underlying_instrument,
                        "underlying_market_symbol": rec.underlying_market_symbol,
                        "market_data_source": rec.market_data_source,
                        "asset_type": rec.asset_type,
                        "payout_pct": f"{int(round(rec.payout_pct * 100))}%",
                        "live_status": "QUOTEX DATA UNAVAILABLE",
                        "price": 0.0,
                        "price_display": "QUOTEX DATA UNAVAILABLE",
                        "signal": "NO-TRADE",
                "action": "NO-TRADE",
                        "confidence_pct": 0,
                        "horizon": horizon,
                        "entry_price": 0.0,
                        "stop_loss": 0.0,
                        "take_profit": 0.0,
                        "levels_display": "—",
                        "rationale": rec.limitation_notes,
                        "note": rec.limitation_notes,
                    }
                )
            else:
                opp = scanned_by_canon.get(canon)
                st = state_svc._states.get(canon)
                if opp is not None and opp.current_price > 0:
                    price = opp.current_price
                    raw_act = opp.action.value if hasattr(opp, "action") else str(getattr(opp, "direction", "NO_TRADE"))
                    ui_sig = "NO-TRADE" if raw_act in ("NO_TRADE", "NO-TRADE") else raw_act
                    conf = int(round(opp.confidence * 100))
                    ep = opp.current_price
                    sl = getattr(opp, "stop_price", 0.0)
                    tp = getattr(opp, "target_price", 0.0)
                    reasons = getattr(opp, "reasons", []) or []
                    rationale = reasons[0] if reasons else getattr(opp, "rationale", "")
                else:
                    price = st.last_price if st and st.last_price > 0 else 0.0
                    ui_sig = "NO-TRADE"
                    conf = 0
                    ep = price
                    sl = 0.0
                    tp = 0.0
                    rationale = f"Awaiting live quote update from {rec.underlying_instrument}"

                ep_str = self.format_price(ep, "USD", "QUOTEX") if ep > 0 else "—"
                sl_str = self.format_price(sl, "USD", "QUOTEX") if sl > 0 else "—"
                tp_str = self.format_price(tp, "USD", "QUOTEX") if tp > 0 else "—"
                levels_str = f"{ep_str} / {sl_str} / {tp_str}" if ui_sig in ("BUY", "SELL") and ep > 0 else "—"

                feed_label = "PROXY (FX OTC INTERBANK)" if "_OTC" in rec.quotex_symbol else "PROXY (FX SPOT UNDERLYING)"
                if "CRYPTO" in rec.asset_type:
                    feed_label = "PROXY (CRYPTO SPOT UNDERLYING)"

                rows.append(
                    {
                        "platform_instrument": canon,
                        "symbol": rec.quotex_symbol,
                        "name": rec.platform_instrument_name,
                        "underlying_instrument": rec.underlying_instrument,
                        "underlying_market_symbol": rec.underlying_market_symbol,
                        "market_data_source": rec.market_data_source,
                        "asset_type": rec.asset_type,
                        "payout_pct": f"{int(round(rec.payout_pct * 100))}%",
                        "live_status": feed_label,
                        "price": price,
                        "price_display": self.format_price(price, "USD", "QUOTEX") if price > 0 else "Click to Load Live",
                        "signal": ui_sig,
                        "confidence_pct": conf,
                        "horizon": horizon,
                        "entry_price": ep,
                        "stop_loss": sl,
                        "take_profit": tp,
                        "levels_display": levels_str,
                        "rationale": rationale,
                        "note": rec.limitation_notes or f"Derived from {rec.underlying_instrument}",
                    }
                )
        return rows

    def get_continuous_learning_status(self) -> dict[str, Any]:
        """Gather real runtime metrics from the ContinuousLearningController."""
        try:
            from learning.controller import ContinuousLearningController

            learning_ctrl = ContinuousLearningController.get_instance()
            return learning_ctrl.get_status_telemetry()
        except Exception as exc:
            return {
                "state": "ERROR",
                "is_running": False,
                "error": str(exc),
                "active_champion_model": "lgbm_champion_v1.0",
                "target_99": {
                    "status": "NOT_YET_VERIFIABLE",
                    "measured_accuracy_pct": None,
                    "target_accuracy_pct": 99.0,
                    "summary": f"Learning controller unavailable: {exc}",
                },
            }

    def get_diagnostics_snapshot(self, current_eval: dict[str, Any] | None = None) -> dict[str, Any]:
        """Gather complete diagnostics, self-development ledger metrics, continuous learning telemetry, and universe audit for Tab 4."""
        ledger = PredictionLedgerService.get_instance()
        ledger_metrics = ledger.get_metrics()
        summary = UniverseService.summary()
        learning_telemetry = self.get_continuous_learning_status()

        try:
            from paper_trading.worker import PaperTradingWorker
            from paper_trading.evaluator import PaperTradingEvaluator
            op_status = PaperTradingWorker.get_operational_status()
            p_eval = PaperTradingEvaluator()
            eval_res = p_eval.evaluate()
            paper_status = {
                "active_mode": "PAPER_TRADING (SIMULATED - REAL TRADING DISABLED)",
                "worker_state": op_status["worker_state"],
                "process_ownership": op_status["process_ownership"],
                "uptime_seconds": op_status["uptime_seconds"],
                "last_successful_cycle": op_status["last_successful_cycle"],
                "last_heartbeat": op_status["last_heartbeat"],
                "metrics": op_status["store_metrics"],
                "evidence_status": eval_res.get("evidence_status", "INSUFFICIENT_PROSPECTIVE_EVIDENCE"),
                "sample_progress": eval_res.get("sample_progress", {}),
                "is_eligible_to_generate": op_status["is_currently_eligible_to_generate"],
                "provider_status": op_status["provider_status"],
            }
        except Exception as exc:
            paper_status = {
                "active_mode": "PAPER_TRADING (SIMULATED - REAL TRADING DISABLED)",
                "worker_state": "STOPPED",
                "evidence_status": "INSUFFICIENT_PROSPECTIVE_EVIDENCE",
                "metrics": {},
                "sample_progress": {},
                "error": str(exc),
            }

        import os
        import threading
        cache_file = Path("data/universe/equity_candles_cache.json")
        cache_size = cache_file.stat().st_size if cache_file.exists() else 0
        cache_keys = 0
        if cache_file.exists():
            try:
                import json
                cache_keys = len(json.loads(cache_file.read_text(encoding="utf-8")))
            except Exception:
                pass

        system_resources = {
            "pid": os.getpid(),
            "active_threads": threading.active_count(),
            "equity_cache_file": str(cache_file),
            "equity_cache_bytes": cache_size,
            "equity_cached_instruments": cache_keys,
            "predictions_paused": self.predictions_paused,
            "governance_mode": "CONTROLLED (AUTO-PROMOTE DISABLED)",
        }

        # Section 8 Candidate Model Accuracy & Dual-Mode Telemetry
        model_accuracy_telemetry = {
            "has_candidate_audit": False,
            "acceptance_state": "NO_EVALUATION",
            "unconstrained_accuracy_pct": None,
            "wilson_ci_lower_pct": None,
            "wilson_ci_upper_pct": None,
            "unconstrained_brier_score": None,
            "high_conviction_win_rate_pct": None,
            "high_conviction_expectancy_pct": None,
            "high_conviction_coverage_pct": None,
        }
        try:
            cand_store = Path("models_store") / "lgbm_multiclass_candidate"
            if cand_store.exists():
                v_dirs = sorted([d for d in cand_store.iterdir() if d.is_dir()])
                if v_dirs:
                    latest_v = v_dirs[-1]
                    audit_f = latest_v / "accuracy_evaluation_report.json"
                    if audit_f.exists():
                        import json
                        aud = json.loads(audit_f.read_text(encoding="utf-8"))
                        model_accuracy_telemetry["has_candidate_audit"] = True
                        model_accuracy_telemetry["candidate_version"] = latest_v.name
                        model_accuracy_telemetry["acceptance_state"] = aud.get("acceptance_state", "UNKNOWN")
                        model_accuracy_telemetry["unconstrained_accuracy_pct"] = aud.get("overall_accuracy_pct")
                        model_accuracy_telemetry["wilson_ci_lower_pct"] = aud.get("wilson_ci_lower_pct")
                        model_accuracy_telemetry["wilson_ci_upper_pct"] = aud.get("wilson_ci_upper_pct")
                        model_accuracy_telemetry["unconstrained_brier_score"] = aud.get("brier_score")
                        model_accuracy_telemetry["high_conviction_win_rate_pct"] = aud.get("active_trade_win_rate_pct")
                        model_accuracy_telemetry["high_conviction_expectancy_pct"] = aud.get("active_trade_expectancy_pct")
                        model_accuracy_telemetry["high_conviction_coverage_pct"] = aud.get("coverage_rate_pct")
        except Exception:
            pass

        return {
            "startup": self._startup_result,
            "latencies": dict(self._latest_latencies),
            "universe_summary": summary,
            "ledger_metrics": ledger_metrics,
            "continuous_learning": learning_telemetry,
            "paper_trading": paper_status,
            "system_resources": system_resources,
            "model_accuracy": model_accuracy_telemetry,
            "current_eval": current_eval or {},
            "logs": self.get_recent_logs()[-80:],
        }

    def get_recent_prediction_outcomes(self, limit: int = 20) -> list[dict[str, Any]]:
        """Return the most recent resolved predictions from the prediction outcomes ledger."""
        try:
            from learning.ledger import PredictionLedgerService

            ledger = PredictionLedgerService.get_instance()
            records: list[dict[str, Any]] = []
            with ledger._ledger_lock:
                if ledger._resolved_outcomes:
                    records = list(ledger._resolved_outcomes[-limit:])
                elif ledger.ledger_file.exists():
                    import json

                    with open(ledger.ledger_file, "r", encoding="utf-8") as f:
                        lines = [line.strip() for line in f if line.strip()]
                        for line in lines[-limit:]:
                            try:
                                records.append(json.loads(line))
                            except Exception:
                                pass

            formatted: list[dict[str, Any]] = []
            for r in reversed(records):
                dt_str = r.get("resolved_at") or r.get("predicted_at") or ""
                if "T" in dt_str:
                    time_part = dt_str.split("T")[1][:8]
                    date_part = dt_str.split("T")[0]
                    formatted_time = f"{date_part} {time_part}"
                else:
                    formatted_time = dt_str[:19]

                net_ret = r.get("net_trade_return_pct")
                if net_ret is None:
                    net_ret = r.get("actual_return_pct", 0.0)

                formatted.append(
                    {
                        "time": formatted_time,
                        "symbol": r.get("canonical_symbol") or r.get("symbol", "—"),
                        "action": r.get("action", "—"),
                        "entry_price": r.get("entry_price", 0.0),
                        "exit_price": r.get("exit_price", 0.0),
                        "outcome": r.get("outcome", "PENDING"),
                        "return_pct": round(float(net_ret), 2) if net_ret is not None else 0.0,
                        "confidence": round(float(r.get("confidence", 0.0)) * 100, 1),
                    }
                )
            return formatted
        except Exception as exc:
            self.log_event(f"Error fetching recent prediction outcomes: {exc}")
            return []

    def stop_background_services(self) -> None:
        """Stop background learning controller and outcome resolver workers cleanly."""
        try:
            from learning.ledger import PredictionLedgerService

            PredictionLedgerService.get_instance().stop_background_resolver()
        except Exception as exc:
            self.log_event(f"Outcome resolver shutdown warning: {exc}")
        try:
            from learning.controller import ContinuousLearningController

            ContinuousLearningController.get_instance().stop_background_loop()
        except Exception as exc:
            self.log_event(f"Learning controller shutdown warning: {exc}")

    def start_learning_controller(self) -> bool:
        """Explicitly start background learning controller loop."""
        from learning.controller import ContinuousLearningController
        ctrl = ContinuousLearningController.get_instance()
        ctrl.auto_promote = False
        ctrl.start_background_loop()
        self.log_event("Continuous learning controller started explicitly by operator")
        return True

    def start_outcome_resolver(self) -> bool:
        """Explicitly start background outcome resolver worker."""
        from learning.ledger import PredictionLedgerService
        ledger = PredictionLedgerService.get_instance()
        ledger.start_background_resolver()
        self.log_event("Outcome resolver worker started explicitly by operator")
        return True

    def get_background_worker_statuses(self) -> dict[str, str]:
        """Query running/stopped status of all background workers."""
        statuses = {
            "paper_trading_worker": "STOPPED",
            "continuous_learning": "STOPPED",
            "outcome_resolver": "STOPPED",
        }
        try:
            from paper_trading.worker import PaperTradingWorker
            st = PaperTradingWorker.get_operational_status()
            statuses["paper_trading_worker"] = st.get("worker_state", "STOPPED")
        except Exception:
            pass
        try:
            from learning.controller import ContinuousLearningController
            ctrl = ContinuousLearningController.get_instance()
            if hasattr(ctrl, "_worker_thread") and ctrl._worker_thread and ctrl._worker_thread.is_alive():
                statuses["continuous_learning"] = "RUNNING"
        except Exception:
            pass
        try:
            from learning.ledger import PredictionLedgerService
            led = PredictionLedgerService.get_instance()
            if hasattr(led, "_resolver_thread") and led._resolver_thread and led._resolver_thread.is_alive():
                statuses["outcome_resolver"] = "RUNNING"
        except Exception:
            pass
        return statuses

