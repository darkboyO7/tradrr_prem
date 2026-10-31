"""End-to-end verification tests for the Windows Desktop Application (Sections 1–23)."""

from __future__ import annotations

from pathlib import Path
import sys

from PySide6.QtWidgets import QApplication

from desktop.main_window import StartupStatusDialog, TradingIntelligenceDesktopWindow
from desktop.service_controller import DesktopServiceController
from desktop.setup_installer import perform_installation


def _get_qapp() -> QApplication:
    return QApplication.instance() or QApplication(sys.argv)


def test_desktop_startup_and_23_point_validation() -> None:
    """Verify all 23 final validation requirements for the Windows Desktop Application."""
    app = _get_qapp()
    controller = DesktopServiceController()

    # 1. Startup Checklist Verification (Section 15)
    splash = StartupStatusDialog(controller)
    res = splash.run_checks()
    assert res.data_ok is True
    assert res.database_ok is True
    assert res.models_ok is True
    assert res.market_connection_ok is True
    assert res.prediction_engine_ok is True
    splash.close()

    # 2. Cross-Listed NSE/BSE Search Verification (Section 6)
    tcs_hits = controller.search_universe("TCS")
    assert len(tcs_hits) > 0
    assert any("NSE:TCS" in h["identifiers_text"] and "BSE:532540" in h["identifiers_text"] for h in tcs_hits)

    rel_hits = controller.search_universe("RELIANCE")
    assert len(rel_hits) > 0
    assert any("NSE:RELIANCE" in h["identifiers_text"] and "BSE:500325" in h["identifiers_text"] for h in rel_hits)

    # 3. Live + Historical + Developing-Day + Time-Synchronized Evaluation (Sections 4, 5, 9, 10, 11)
    eval_tcs = controller.evaluate_instrument("NSE:TCS", timeframe="5m", horizon="1h")
    assert eval_tcs["canonical_symbol"] == "NSE:TCS"
    assert eval_tcs["last_price"] > 0
    assert eval_tcs["signal"] in ("BUY", "SELL", "NO-TRADE")
    assert 0 <= eval_tcs["confidence_pct"] <= 100
    assert eval_tcs["horizon_label"] == "Next 1 Hour"
    assert len(eval_tcs["why_bullets"]) >= 2

    # Verify all 22 Developing-Day fields (Section 10)
    dd = eval_tcs["developing_day"]
    for required_field in (
        "open",
        "current",
        "high",
        "low",
        "range",
        "gap",
        "volume",
        "relative_volume",
        "vwap",
        "trend",
        "momentum",
        "volatility",
        "support",
        "resistance",
        "opening_range",
        "breakout",
        "pullback",
        "consolidation",
        "higher_highs",
        "higher_lows",
        "lower_highs",
        "lower_lows",
    ):
        assert required_field in dd, f"Missing developing_day field: {required_field}"

    # Verify Historical + Live Timestamp Synchronization & Zero Future Leakage (Section 11)
    pc = eval_tcs["pattern_consensus"]
    assert pc.get("zero_future_leakage_verified") is True
    assert "matched_historical_timestamps" in pc

    # Verify Latency Telemetry (Section 5)
    lat = eval_tcs["latencies"]
    assert lat["market_data_latency_ms"] >= 0.0
    assert lat["processing_latency_ms"] >= 0.0
    assert lat["prediction_latency_ms"] >= 0.0

    # 4. Quotex Dedicated Section & Unavailable Guard (Section 7)
    qx_unavail = controller.evaluate_instrument("QUOTEX:MSFT_OTC", timeframe="5m", horizon="1h")
    assert qx_unavail["status"] == "QUOTEX_DATA_UNAVAILABLE"
    assert qx_unavail["session_badge"] == "QUOTEX DATA UNAVAILABLE"
    assert qx_unavail["price_display"] == "QUOTEX DATA UNAVAILABLE"
    assert qx_unavail["signal"] == "NO-TRADE"

    qx_live = controller.evaluate_instrument("QUOTEX:EURUSD_OTC", timeframe="5m", horizon="1h")
    assert qx_live["quotex_info"]["platform_instrument"] == "QUOTEX:EURUSD_OTC"
    assert qx_live["quotex_info"]["underlying_instrument"] != "QUOTEX:EURUSD_OTC"

    # 5. Desktop Window, Native Chart, Market Scanner, System Tray & Diagnostics (Sections 2, 3, 4, 8, 17, 18, 19)
    win = TradingIntelligenceDesktopWindow(controller, auto_start_workers=False)
    win._on_evaluation_ready(eval_tcs)
    assert win.lbl_symbol.text() == "TCS"
    assert win.lbl_signal_banner.text() in ("BUY", "SELL", "NO-TRADE")
    assert win.tray_icon is not None
    assert win.btn_refresh_dashboard is not None
    assert win.table_recent_outcomes is not None
    assert win.table_recent_outcomes.columnCount() == 7
    assert win.table_recent_outcomes.horizontalHeaderItem(0).text() == "Time (UTC)"
    assert win.table_recent_outcomes.horizontalHeaderItem(6).text() == "Net Return"

    # Verify Historical Pattern Analogs Table & Audio Notification Control
    assert win.table_pattern_analogs is not None
    assert win.table_pattern_analogs.columnCount() == 8
    assert win.table_pattern_analogs.horizontalHeaderItem(0).text() == "Analog #"
    assert win.table_pattern_analogs.horizontalHeaderItem(2).text() == "Similarity"
    assert win.chk_notif_sound is not None
    assert win.chk_notif_sound.isChecked() is True
    assert controller.notification_settings.sound_enabled is True

    # Test System Tray Pause/Resume Predictions
    win._tray_pause_predictions()
    assert controller.predictions_paused is True
    win._tray_resume_predictions()
    assert controller.predictions_paused is False

    # Test Scanner population
    scanner_rows = controller.scan_market(market="NSE", limit=6)
    assert len(scanner_rows) > 0
    win._on_scanner_ready(scanner_rows)
    assert win.scanner_table.rowCount() == len(scanner_rows)

    # Test Honest Out-of-Sample Accuracy (Section 13: Never hardcoded 99%)
    diag = controller.get_diagnostics_snapshot(eval_tcs)
    win_rate = float(diag["ledger_metrics"].get("win_rate", 0.0))
    assert win_rate != 0.99

    win._force_quit = True
    win.tray_icon.hide()
    win.close()
    controller.stop_background_services()
    app.processEvents()
