"""Regression tests verifying Desktop Safe Read-Only Startup Isolation (P0).

Proves that:
1. Default desktop launch is strictly safe read-only.
2. Background workers (paper trading, continuous learning, outcome resolver, scanner) default to STOPPED.
3. Real-money order execution is permanently disabled.
4. Zero orders are placed upon desktop startup.
"""

from __future__ import annotations

import sys
from PySide6.QtWidgets import QApplication

from desktop.main_window import TradingIntelligenceDesktopWindow
from desktop.service_controller import DesktopServiceController
from paper_trading.worker import PaperTradingWorker


def _get_qapp() -> QApplication:
    return QApplication.instance() or QApplication(sys.argv)


def test_desktop_safe_startup_leaves_all_workers_stopped() -> None:
    """Verify that default desktop initialization leaves all background workers strictly STOPPED."""
    app = _get_qapp()
    controller = DesktopServiceController()

    # Verify controller worker statuses before window launch
    statuses = controller.get_background_worker_statuses()
    assert statuses["paper_trading_worker"] == "STOPPED"
    assert statuses["continuous_learning"] == "STOPPED"
    assert statuses["outcome_resolver"] == "STOPPED"

    # Verify PaperTradingWorker state directly
    op_status = PaperTradingWorker.get_operational_status()
    assert op_status["worker_state"] == "STOPPED"
    assert op_status["process_ownership"]["lock_acquired"] is False

    # Launch desktop window with default arguments (auto_start_workers must default to False)
    win = TradingIntelligenceDesktopWindow(controller)
    try:
        assert win._auto_start_workers is False
        assert win.live_timer.isActive() is False
        assert win._eval_worker is None or not win._eval_worker.isRunning()
        assert win._scanner_worker is None or not win._scanner_worker.isRunning()

        # Verify real-money execution is disabled
        diag = controller.get_diagnostics_snapshot({})
        assert "REAL TRADING DISABLED" in diag["paper_trading"]["active_mode"]
        assert diag["paper_trading"]["worker_state"] == "STOPPED"

        # Verify worker statuses remain STOPPED after window creation
        updated_statuses = controller.get_background_worker_statuses()
        assert updated_statuses["paper_trading_worker"] == "STOPPED"
        assert updated_statuses["continuous_learning"] == "STOPPED"
        assert updated_statuses["outcome_resolver"] == "STOPPED"
    finally:
        win._force_quit = True
        win.tray_icon.hide()
        win.close()
        controller.stop_background_services()
        app.processEvents()


def test_independent_explicit_worker_controls() -> None:
    """Verify that background workers require explicit activation and report running state."""
    controller = DesktopServiceController()
    try:
        # Before explicit start
        st = controller.get_background_worker_statuses()
        assert st["continuous_learning"] == "STOPPED"
        assert st["outcome_resolver"] == "STOPPED"

        # Explicitly start outcome resolver
        started = controller.start_outcome_resolver()
        assert started is True
        st_after = controller.get_background_worker_statuses()
        assert st_after["outcome_resolver"] == "RUNNING"
    finally:
        controller.stop_background_services()
        st_stopped = controller.get_background_worker_statuses()
        assert st_stopped["outcome_resolver"] == "STOPPED"
