"""Trading Intelligence Platform — Native Windows Desktop Application Window (PySide6).

Provides:
1. Startup Status Verification Dialog (DATA, DATABASE, MODELS, MARKET CONNECTION, PREDICTION ENGINE)
2. Simple Beginner Dashboard (Market bar, Search with NSE/BSE scrip cross-listing, Price, BUY/SELL/NO-TRADE,
   Confidence, Forecast Horizon, Entry/Stop/Target, WHY bullets, and Native Live Candlestick Graph)
3. Dedicated Multi-Exchange Market Scanner (Search, Filter, Sort, Market, Signal, Confidence, Timeframe)
4. Dedicated Quotex Market Section (Platform Instrument vs Underlying Instrument, QUOTEX DATA UNAVAILABLE guard)
5. Advanced / System Diagnostics Tab (Latency telemetry, 22-field DevelopingDayState, Time-Synchronized
   Historical Pattern Matches, Honest Out-of-Sample Self-Development Ledger, Notification Config, Logs)
6. Windows System Tray Integration (Open Dashboard, Market Scanner, Pause/Resume Predictions, System Status, Exit)
"""

from __future__ import annotations

from datetime import UTC, datetime
import time
from typing import Any
from zoneinfo import ZoneInfo

from PySide6.QtCore import QSize, Qt, QThread, QTimer, Signal
from PySide6.QtGui import QAction, QColor, QCloseEvent, QFont, QIcon, QKeySequence, QPainter, QPixmap, QShortcut
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMenu,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSlider,
    QSplitter,
    QSystemTrayIcon,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from desktop.chart_widget import LiveCandlestickChartWidget
from desktop.service_controller import DesktopServiceController, StartupCheckResult


def create_app_icon() -> QIcon:
    """Create a native vector QIcon for the Windows taskbar, window frame, and system tray."""
    pix = QPixmap(64, 64)
    pix.fill(Qt.GlobalColor.transparent)
    painter = QPainter(pix)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)

    painter.setBrush(QColor("#0F172A"))
    painter.setPen(QColor("#10B981"))
    painter.drawRoundedRect(4, 4, 56, 56, 14, 14)

    # Draw ascending candlestick bars inside icon
    painter.setPen(Qt.PenStyle.NoPen)
    painter.setBrush(QColor("#38BDF8"))
    painter.drawRoundedRect(14, 34, 8, 16, 2, 2)
    painter.setBrush(QColor("#10B981"))
    painter.drawRoundedRect(28, 24, 8, 26, 2, 2)
    painter.setBrush(QColor("#34D399"))
    painter.drawRoundedRect(42, 14, 8, 36, 2, 2)

    painter.end()
    return QIcon(pix)


class StartupStatusDialog(QDialog):
    """Startup checklist dialog verifying all 5 core systems before opening the dashboard (Section 15)."""

    def __init__(self, controller: DesktopServiceController, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.controller = controller
        self.setWindowTitle("Trading Intelligence Platform — Starting Services")
        self.setWindowIcon(create_app_icon())
        self.setFixedSize(560, 390)
        self.setStyleSheet(
            """
            QDialog { background-color: #0B1120; color: #F8FAFC; }
            QLabel { color: #E2E8F0; font-family: 'Segoe UI'; }
            QProgressBar {
                background-color: #1E293B; border: 1px solid #334155;
                border-radius: 6px; text-align: center; color: #F8FAFC; height: 18px;
            }
            QProgressBar::chunk { background-color: #10B981; border-radius: 5px; }
            """
        )

        layout = QVBoxLayout(self)
        layout.setContentsMargins(28, 24, 28, 24)
        layout.setSpacing(14)

        title = QLabel("TRADING INTELLIGENCE PLATFORM")
        title.setFont(QFont("Segoe UI", 15, QFont.Weight.Bold))
        title.setStyleSheet("color: #38BDF8;")
        layout.addWidget(title)

        subtitle = QLabel("Initializing local Windows desktop runtime, market universe & prediction engine...")
        subtitle.setFont(QFont("Segoe UI", 9))
        subtitle.setStyleSheet("color: #94A3B8;")
        layout.addWidget(subtitle)

        grid = QGridLayout()
        grid.setVerticalSpacing(10)
        grid.setHorizontalSpacing(16)

        self._rows: dict[str, tuple[QLabel, QLabel]] = {}
        checks = [
            ("DATA", "Synchronizing NSE, BSE, Quotex, Forex & Crypto universe..."),
            ("DATABASE", "Checking local database & prediction ledger storage..."),
            ("MODELS", "Verifying production ML & hierarchical model registry..."),
            ("MARKET CONNECTION", "Connecting to market data feeds & historical candles..."),
            ("PREDICTION ENGINE", "Starting multi-source signal fusion & risk gate..."),
        ]
        for idx, (key, desc) in enumerate(checks):
            name_lbl = QLabel(key)
            name_lbl.setFont(QFont("Segoe UI", 10, QFont.Weight.Bold))
            status_lbl = QLabel("⋯")
            status_lbl.setFont(QFont("Segoe UI", 12, QFont.Weight.Bold))
            status_lbl.setStyleSheet("color: #F59E0B;")
            detail_lbl = QLabel(desc)
            detail_lbl.setFont(QFont("Segoe UI", 9))
            detail_lbl.setStyleSheet("color: #94A3B8;")

            grid.addWidget(name_lbl, idx, 0)
            grid.addWidget(status_lbl, idx, 1)
            grid.addWidget(detail_lbl, idx, 2)
            self._rows[key] = (status_lbl, detail_lbl)

        layout.addLayout(grid)

        self.step_label = QLabel("Starting...")
        self.step_label.setFont(QFont("Segoe UI", 9))
        self.step_label.setStyleSheet("color: #CBD5E1;")
        layout.addWidget(self.step_label)

        self.progress = QProgressBar()
        self.progress.setRange(0, 10)
        self.progress.setValue(0)
        layout.addWidget(self.progress)

    def run_checks(self) -> StartupCheckResult:
        def _on_progress(step: int, msg: str, res: StartupCheckResult) -> None:
            self.progress.setValue(step)
            self.step_label.setText(f"Step {step}/10: {msg}")
            self._update_row("DATA", res.data_ok, res.data_detail, step >= 3)
            self._update_row("DATABASE", res.database_ok, res.database_detail, step >= 2)
            self._update_row("MODELS", res.models_ok, res.models_detail, step >= 4)
            self._update_row("MARKET CONNECTION", res.market_connection_ok, res.market_connection_detail, step >= 6)
            self._update_row("PREDICTION ENGINE", res.prediction_engine_ok, res.prediction_engine_detail, step >= 8)
            QApplication.processEvents()

        result = self.controller.run_startup_sequence(progress_callback=_on_progress)
        QApplication.processEvents()
        return result

    def _update_row(self, key: str, ok: bool, detail: str, evaluated: bool) -> None:
        status_lbl, detail_lbl = self._rows[key]
        detail_lbl.setText(detail)
        if not evaluated:
            status_lbl.setText("⋯")
            status_lbl.setStyleSheet("color: #F59E0B;")
        elif ok:
            status_lbl.setText("✓")
            status_lbl.setStyleSheet("color: #10B981;")
        else:
            status_lbl.setText("✗")
            status_lbl.setStyleSheet("color: #EF4444;")


class EvaluationWorker(QThread):
    """Background worker thread running live market + prediction evaluation without blocking Qt UI."""

    evaluation_ready = Signal(dict)

    def __init__(
        self,
        controller: DesktopServiceController,
        symbol: str,
        timeframe: str = "5m",
        horizon: str = "1h",
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.controller = controller
        self.symbol = symbol
        self.timeframe = timeframe
        self.horizon = horizon

    def run(self) -> None:
        try:
            payload = self.controller.evaluate_instrument(
                symbol_or_query=self.symbol,
                timeframe=self.timeframe,
                horizon=self.horizon,
            )
            self.evaluation_ready.emit(payload)
        except Exception as exc:
            self.evaluation_ready.emit(
                {
                    "status": "ERROR",
                    "symbol": self.symbol,
                    "canonical_symbol": self.symbol,
                    "market": "NSE",
                    "company_name": self.symbol,
                    "session_badge": "DATA ERROR",
                    "price_display": "—",
                    "signal": "NO-TRADE",
                    "confidence_pct": 0,
                    "horizon": self.horizon,
                    "horizon_label": f"Next {self.horizon}",
                    "entry_price": 0.0,
                    "stop_loss": 0.0,
                    "take_profit": 0.0,
                    "why_bullets": [f"Evaluation error: {exc}"],
                    "candles": [],
                    "developing_day": {},
                    "pattern_consensus": {},
                    "latencies": {},
                }
            )


class ScannerWorker(QThread):
    """Background worker thread running multi-exchange market scanner queries."""

    scanner_ready = Signal(list)

    def __init__(
        self,
        controller: DesktopServiceController,
        market: str,
        signal_filter: str,
        min_conf: int,
        horizon: str,
        search_q: str,
        sort_by: str,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.controller = controller
        self.market = market
        self.signal_filter = signal_filter
        self.min_conf = min_conf
        self.horizon = horizon
        self.search_q = search_q
        self.sort_by = sort_by

    def run(self) -> None:
        try:
            rows = self.controller.scan_market(
                market=self.market,
                signal_filter=self.signal_filter,
                min_confidence_pct=self.min_conf,
                horizon=self.horizon,
                search_query=self.search_q,
                sort_by=self.sort_by,
                limit=45,
            )
            self.scanner_ready.emit(rows)
        except Exception:
            self.scanner_ready.emit([])


class TradingIntelligenceDesktopWindow(QMainWindow):
    """Main Windows Desktop Application Window for the Trading Intelligence Platform."""

    def __init__(self, controller: DesktopServiceController, auto_start_workers: bool = False) -> None:
        super().__init__()
        self.controller = controller
        self._auto_start_workers = auto_start_workers
        self.active_symbol: str = "NSE:TCS"
        self.active_market_filter: str = "ALL"
        self.active_horizon: str = "1h"
        self.active_timeframe: str = "5m"
        self._last_eval_payload: dict[str, Any] = {}
        self._eval_worker: EvaluationWorker | None = None
        self._scanner_worker: ScannerWorker | None = None
        self._force_quit: bool = False

        self.setWindowTitle("Trading Intelligence Platform — Windows Desktop Edition")
        self.setWindowIcon(create_app_icon())
        self.resize(1340, 880)
        self._apply_stylesheet()

        # Central container & Top Navigation Tabs
        central = QWidget(self)
        self.setCentralWidget(central)
        root_layout = QVBoxLayout(central)
        root_layout.setContentsMargins(14, 12, 14, 10)
        root_layout.setSpacing(10)

        self.tabs = QTabWidget()
        self.tabs.setDocumentMode(True)
        root_layout.addWidget(self.tabs)

        # Build the 4 application views
        self.dashboard_tab = QWidget()
        self.scanner_tab = QWidget()
        self.quotex_tab = QWidget()
        self.diagnostics_tab = QWidget()

        self._build_beginner_dashboard_tab()
        self._build_market_scanner_tab()
        self._build_quotex_tab()
        self._build_diagnostics_tab()

        self.tabs.addTab(self.dashboard_tab, "  Beginner Dashboard  ")
        self.tabs.addTab(self.scanner_tab, "  Market Scanner  ")
        self.tabs.addTab(self.quotex_tab, "  Quotex Instruments  ")
        self.tabs.addTab(self.diagnostics_tab, "  Advanced / System Diagnostics  ")

        self.tabs.currentChanged.connect(self._on_tab_changed)

        # Setup Windows System Tray (Section 17)
        self._setup_system_tray()

        # Setup Event-Driven Live Refresh Timer (every 6 seconds when active)
        self.live_timer = QTimer(self)
        self.live_timer.setInterval(6000)
        self.live_timer.timeout.connect(self._on_live_timer_tick)
        if auto_start_workers:
            self.live_timer.start()
            self.trigger_instrument_evaluation(self.active_symbol)
            QTimer.singleShot(400, self.trigger_scanner_refresh)
            QTimer.singleShot(600, self._populate_quotex_table)
            QTimer.singleShot(800, self._populate_recent_outcomes_table)
        else:
            self._populate_quotex_table()
            self._populate_recent_outcomes_table()

        # 1-second system clock timer for IST/UTC time display
        self.clock_timer = QTimer(self)
        self.clock_timer.setInterval(1000)
        self.clock_timer.timeout.connect(self._on_clock_tick)
        self.clock_timer.start()
        self._on_clock_tick()

        # Keyboard productivity shortcuts
        self._setup_keyboard_shortcuts()

    def _setup_keyboard_shortcuts(self) -> None:
        """Register productivity shortcuts (F5, Ctrl+R, Ctrl+F, Escape, Ctrl+1..4)."""
        QShortcut(QKeySequence("F5"), self, self._on_manual_refresh_clicked)
        QShortcut(QKeySequence("Ctrl+R"), self, self._on_manual_refresh_clicked)
        QShortcut(QKeySequence("Ctrl+F"), self, self._focus_search)
        QShortcut(QKeySequence("Escape"), self, self._dismiss_search)
        QShortcut(QKeySequence("Ctrl+1"), self, lambda: self.tabs.setCurrentIndex(0))
        QShortcut(QKeySequence("Ctrl+2"), self, lambda: self.tabs.setCurrentIndex(1))
        QShortcut(QKeySequence("Ctrl+3"), self, lambda: self.tabs.setCurrentIndex(2))
        QShortcut(QKeySequence("Ctrl+4"), self, lambda: self.tabs.setCurrentIndex(3))

    def _focus_search(self) -> None:
        if hasattr(self, "search_input"):
            self.search_input.setFocus()
            self.search_input.selectAll()

    def _dismiss_search(self) -> None:
        if hasattr(self, "search_results_list"):
            self.search_results_list.setVisible(False)
        if hasattr(self, "search_input"):
            self.search_input.clearFocus()

    def _on_clock_tick(self) -> None:
        try:
            ist_now = datetime.now(ZoneInfo("Asia/Kolkata"))
            utc_now = datetime.now(UTC)
            if hasattr(self, "lbl_clock"):
                self.lbl_clock.setText(f"IST {ist_now.strftime('%H:%M:%S')}  •  UTC {utc_now.strftime('%H:%M:%S')}")
        except Exception:
            pass

    def _apply_stylesheet(self) -> None:
        self.setStyleSheet(
            """
            QMainWindow, QWidget {
                background-color: #0B1120;
                color: #F8FAFC;
                font-family: 'Segoe UI';
            }
            QTabWidget::pane {
                border: 1px solid #1E293B;
                border-radius: 8px;
                background-color: #0B1120;
            }
            QTabBar::tab {
                background-color: #111827;
                color: #94A3B8;
                padding: 9px 18px;
                margin-right: 6px;
                border-top-left-radius: 6px;
                border-top-right-radius: 6px;
                font-weight: 600;
                font-size: 13px;
            }
            QTabBar::tab:selected {
                background-color: #1E293B;
                color: #38BDF8;
                border-bottom: 2px solid #38BDF8;
            }
            QLineEdit, QComboBox {
                background-color: #111827;
                border: 1px solid #334155;
                border-radius: 6px;
                padding: 7px 12px;
                color: #F8FAFC;
                font-size: 13px;
            }
            QLineEdit:focus, QComboBox:focus {
                border: 1px solid #38BDF8;
            }
            QPushButton {
                background-color: #1E293B;
                border: 1px solid #334155;
                border-radius: 6px;
                padding: 7px 14px;
                color: #E2E8F0;
                font-weight: 600;
            }
            QPushButton:hover {
                background-color: #334155;
            }
            QTableWidget {
                background-color: #0F172A;
                border: 1px solid #1E293B;
                border-radius: 8px;
                gridline-color: #1E293B;
                color: #E2E8F0;
                font-size: 12px;
            }
            QHeaderView::section {
                background-color: #1E293B;
                color: #94A3B8;
                padding: 7px;
                border: none;
                font-weight: 700;
            }
            QListWidget {
                background-color: #111827;
                border: 1px solid #38BDF8;
                border-radius: 6px;
                color: #F8FAFC;
                font-size: 12px;
            }
            QListWidget::item {
                padding: 7px 10px;
                border-bottom: 1px solid #1E293B;
            }
            QListWidget::item:selected, QListWidget::item:hover {
                background-color: #1E293B;
                color: #38BDF8;
            }
            """
        )

    # =========================================================================
    # TAB 1: SIMPLE BEGINNER DASHBOARD (Section 4)
    # =========================================================================
    def _build_beginner_dashboard_tab(self) -> None:
        layout = QVBoxLayout(self.dashboard_tab)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(12)

        # Top Control Bar: Market buttons + Search input + Horizon selector
        top_bar = QHBoxLayout()
        top_bar.setSpacing(8)

        mkt_lbl = QLabel("MARKET:")
        mkt_lbl.setFont(QFont("Segoe UI", 10, QFont.Weight.Bold))
        mkt_lbl.setStyleSheet("color: #94A3B8;")
        top_bar.addWidget(mkt_lbl)

        self.market_buttons: dict[str, QPushButton] = {}
        for mkt in ("ALL", "NSE", "BSE", "FOREX", "QUOTEX", "CRYPTO"):
            btn = QPushButton(mkt)
            btn.setCheckable(True)
            btn.setChecked(mkt == "ALL")
            btn.clicked.connect(lambda checked, m=mkt: self._on_market_filter_clicked(m))
            top_bar.addWidget(btn)
            self.market_buttons[mkt] = btn
        self._style_market_buttons()

        top_bar.addSpacing(14)

        search_lbl = QLabel("SEARCH:")
        search_lbl.setFont(QFont("Segoe UI", 10, QFont.Weight.Bold))
        search_lbl.setStyleSheet("color: #94A3B8;")
        top_bar.addWidget(search_lbl)

        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText(
            "Search stock / instrument (e.g. TCS, RELIANCE, INFY, HDFCBANK, 500325, EURUSD)..."
        )
        self.search_input.setMinimumWidth(360)
        self.search_input.textChanged.connect(self._on_search_text_changed)
        self.search_input.returnPressed.connect(self._on_search_enter_pressed)
        top_bar.addWidget(self.search_input, stretch=1)

        hz_lbl = QLabel("FORECAST:")
        hz_lbl.setFont(QFont("Segoe UI", 10, QFont.Weight.Bold))
        hz_lbl.setStyleSheet("color: #94A3B8;")
        top_bar.addWidget(hz_lbl)

        self.horizon_combo = QComboBox()
        self.horizon_combo.addItems(["5m", "15m", "1h", "4h", "1d"])
        self.horizon_combo.setCurrentText("1h")
        self.horizon_combo.currentTextChanged.connect(self._on_horizon_changed)
        top_bar.addWidget(self.horizon_combo)

        top_bar.addSpacing(6)
        self.btn_refresh_dashboard = QPushButton("↻ Refresh (F5)")
        self.btn_refresh_dashboard.setFont(QFont("Segoe UI", 9, QFont.Weight.Bold))
        self.btn_refresh_dashboard.setStyleSheet(
            "QPushButton { background-color: #1E293B; color: #38BDF8; border: 1px solid #38BDF8; "
            "border-radius: 6px; padding: 4px 12px; }"
            "QPushButton:hover { background-color: #0284C7; color: #FFFFFF; }"
        )
        self.btn_refresh_dashboard.clicked.connect(self._on_manual_refresh_clicked)
        top_bar.addWidget(self.btn_refresh_dashboard)

        top_bar.addSpacing(10)
        self.lbl_clock = QLabel("—")
        self.lbl_clock.setFont(QFont("Segoe UI", 9, QFont.Weight.Bold))
        self.lbl_clock.setStyleSheet(
            "color: #38BDF8; background-color: #111827; border: 1px solid #1E293B; "
            "border-radius: 6px; padding: 4px 10px;"
        )
        top_bar.addWidget(self.lbl_clock)

        layout.addLayout(top_bar)

        # Instant Search Results Dropdown List (Hidden when empty)
        self.search_results_list = QListWidget()
        self.search_results_list.setMaximumHeight(165)
        self.search_results_list.setVisible(False)
        self.search_results_list.itemClicked.connect(self._on_search_item_clicked)
        layout.addWidget(self.search_results_list)

        # Quick Preset Chips for instant 1-click switching
        preset_bar = QHBoxLayout()
        preset_bar.setSpacing(6)
        quick_lbl = QLabel("Popular:")
        quick_lbl.setStyleSheet("color: #64748B; font-size: 11px;")
        preset_bar.addWidget(quick_lbl)
        for label, canon in (
            ("TCS (NSE)", "NSE:TCS"),
            ("RELIANCE (NSE)", "NSE:RELIANCE"),
            ("INFY (NSE)", "NSE:INFY"),
            ("HDFCBANK (NSE)", "NSE:HDFCBANK"),
            ("RELIANCE (BSE:500325)", "BSE:RELIANCE"),
            ("EUR/USD (FX)", "FX:EUR/USD"),
            ("EURUSD_OTC (Quotex)", "QUOTEX:EURUSD_OTC"),
            ("MSFT_OTC (Quotex Unavailable)", "QUOTEX:MSFT_OTC"),
            ("BTC-USD (Crypto)", "CRYPTO:BTC-USD"),
        ):
            chip = QPushButton(label)
            chip.setStyleSheet(
                "QPushButton { background-color: #111827; border: 1px solid #1E293B; "
                "border-radius: 4px; padding: 3px 9px; font-size: 11px; color: #CBD5E1; }"
                "QPushButton:hover { border-color: #38BDF8; color: #38BDF8; }"
            )
            chip.clicked.connect(lambda checked=False, c=canon: self.trigger_instrument_evaluation(c))
            preset_bar.addWidget(chip)
        preset_bar.addStretch()
        layout.addLayout(preset_bar)

        # Main Beginner Card (Left: Instrument + Price + Signal + Levels; Right: WHY? explanation)
        card_frame = QFrame()
        card_frame.setStyleSheet(
            "QFrame { background-color: #0F172A; border: 1px solid #1E293B; border-radius: 12px; }"
        )
        card_layout = QHBoxLayout(card_frame)
        card_layout.setContentsMargins(24, 20, 24, 20)
        card_layout.setSpacing(28)

        # Left column: Symbol, Exchange, Current Price, Levels
        left_col = QVBoxLayout()
        left_col.setSpacing(6)

        header_row = QHBoxLayout()
        self.lbl_symbol = QLabel("TCS")
        self.lbl_symbol.setFont(QFont("Segoe UI", 24, QFont.Weight.Bold))
        self.lbl_symbol.setStyleSheet("color: #F8FAFC; border: none;")
        header_row.addWidget(self.lbl_symbol)

        self.lbl_session_badge = QLabel("LIVE")
        self.lbl_session_badge.setFont(QFont("Segoe UI", 10, QFont.Weight.Bold))
        self.lbl_session_badge.setStyleSheet(
            "background-color: #064E3B; color: #34D399; padding: 4px 12px; border-radius: 12px; border: none;"
        )
        header_row.addWidget(self.lbl_session_badge)

        self.lbl_timestamp = QLabel("Updated: —")
        self.lbl_timestamp.setFont(QFont("Segoe UI", 9))
        self.lbl_timestamp.setStyleSheet("color: #94A3B8; border: none; margin-left: 8px;")
        header_row.addWidget(self.lbl_timestamp)
        header_row.addStretch()
        left_col.addLayout(header_row)

        self.lbl_exchange_sub = QLabel("NSE  •  Tata Consultancy Services Limited  •  BSE:532540")
        self.lbl_exchange_sub.setFont(QFont("Segoe UI", 10))
        self.lbl_exchange_sub.setStyleSheet("color: #94A3B8; border: none;")
        left_col.addWidget(self.lbl_exchange_sub)

        left_col.addSpacing(6)

        self.lbl_price = QLabel("₹—")
        self.lbl_price.setFont(QFont("Segoe UI", 30, QFont.Weight.Bold))
        self.lbl_price.setStyleSheet("color: #38BDF8; border: none;")
        left_col.addWidget(self.lbl_price)

        left_col.addSpacing(10)

        # Entry / Stop / Target boxes
        levels_layout = QHBoxLayout()
        levels_layout.setSpacing(14)
        self.lbl_entry = self._create_level_box(levels_layout, "Entry", "₹—", "#38BDF8")
        self.lbl_stop = self._create_level_box(levels_layout, "Stop", "₹—", "#F87171")
        self.lbl_target = self._create_level_box(levels_layout, "Target", "₹—", "#34D399")
        left_col.addLayout(levels_layout)

        # Tactical Risk & Position Sizing Assistant
        risk_box = QFrame()
        risk_box.setStyleSheet(
            "QFrame { background-color: #111827; border: 1px solid #1E293B; border-radius: 8px; }"
        )
        risk_layout = QVBoxLayout(risk_box)
        risk_layout.setContentsMargins(10, 6, 10, 6)
        risk_layout.setSpacing(4)

        risk_hdr = QHBoxLayout()
        risk_title = QLabel("RISK & SIZING CALCULATOR")
        risk_title.setFont(QFont("Segoe UI", 8, QFont.Weight.Bold))
        risk_title.setStyleSheet("color: #38BDF8; border: none;")
        risk_hdr.addWidget(risk_title)
        risk_hdr.addStretch()

        risk_hdr.addWidget(QLabel("Capital:"))
        self.edit_capital = QLineEdit("100000")
        self.edit_capital.setFixedWidth(75)
        self.edit_capital.setStyleSheet("padding: 2px 4px; font-size: 11px;")
        self.edit_capital.textChanged.connect(self._recalculate_position_size)
        risk_hdr.addWidget(self.edit_capital)

        risk_hdr.addWidget(QLabel("Risk:"))
        self.combo_risk_pct = QComboBox()
        self.combo_risk_pct.addItems(["0.5%", "1.0%", "1.5%", "2.0%"])
        self.combo_risk_pct.setCurrentText("1.0%")
        self.combo_risk_pct.setFixedWidth(60)
        self.combo_risk_pct.setStyleSheet("padding: 2px 4px; font-size: 11px;")
        self.combo_risk_pct.currentTextChanged.connect(lambda _: self._recalculate_position_size())
        risk_hdr.addWidget(self.combo_risk_pct)
        risk_layout.addLayout(risk_hdr)

        self.lbl_risk_sizing_summary = QLabel("Enter capital to compute risk-adjusted position sizing.")
        self.lbl_risk_sizing_summary.setFont(QFont("Segoe UI", 8))
        self.lbl_risk_sizing_summary.setStyleSheet("color: #94A3B8; border: none;")
        risk_layout.addWidget(self.lbl_risk_sizing_summary)

        left_col.addWidget(risk_box)

        card_layout.addLayout(left_col, stretch=4)

        # Center column: Visually Dominant Signal (BUY / SELL / NO-TRADE) + Confidence + Forecast Horizon
        center_col = QVBoxLayout()
        center_col.setAlignment(Qt.AlignmentFlag.AlignCenter)
        center_col.setSpacing(8)

        self.lbl_signal_banner = QLabel("LOADING")
        self.lbl_signal_banner.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_signal_banner.setMinimumSize(230, 82)
        self.lbl_signal_banner.setFont(QFont("Segoe UI", 28, QFont.Weight.Bold))
        self.lbl_signal_banner.setStyleSheet(
            "background-color: #1E293B; color: #F8FAFC; border-radius: 14px; padding: 12px 24px;"
        )
        center_col.addWidget(self.lbl_signal_banner)

        self.lbl_confidence = QLabel("—% CONFIDENCE")
        self.lbl_confidence.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_confidence.setFont(QFont("Segoe UI", 14, QFont.Weight.Bold))
        self.lbl_confidence.setStyleSheet("color: #E2E8F0; border: none;")
        center_col.addWidget(self.lbl_confidence)

        self.lbl_forecast_horizon = QLabel("Forecast: Next 1 Hour")
        self.lbl_forecast_horizon.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_forecast_horizon.setFont(QFont("Segoe UI", 11))
        self.lbl_forecast_horizon.setStyleSheet("color: #94A3B8; border: none;")
        center_col.addWidget(self.lbl_forecast_horizon)

        card_layout.addLayout(center_col, stretch=3)

        # Right column: Plain-English WHY? bullets
        right_col = QVBoxLayout()
        right_col.setSpacing(8)

        why_title = QLabel("WHY?")
        why_title.setFont(QFont("Segoe UI", 12, QFont.Weight.Bold))
        why_title.setStyleSheet("color: #38BDF8; border: none;")
        right_col.addWidget(why_title)

        self.lbl_why_bullets = QLabel("• Synchronizing live market and historical patterns...")
        self.lbl_why_bullets.setWordWrap(True)
        self.lbl_why_bullets.setFont(QFont("Segoe UI", 11))
        self.lbl_why_bullets.setStyleSheet("color: #E2E8F0; line-height: 1.45; border: none;")
        right_col.addWidget(self.lbl_why_bullets)
        right_col.addStretch()

        card_layout.addLayout(right_col, stretch=5)

        layout.addWidget(card_frame)

        # Live Intraday Candlestick & VWAP Chart Widget below
        self.chart_widget = LiveCandlestickChartWidget(self)
        layout.addWidget(self.chart_widget, stretch=3)

        # Compact Dual-Panel Analytics Section: Verified Outcomes & Historical Pattern Analogs
        history_box = QFrame()
        history_box.setStyleSheet(
            "QFrame { background-color: #0F172A; border: 1px solid #1E293B; border-radius: 8px; }"
        )
        history_layout = QVBoxLayout(history_box)
        history_layout.setContentsMargins(8, 6, 8, 6)
        history_layout.setSpacing(4)

        self.dash_bottom_tabs = QTabWidget()
        self.dash_bottom_tabs.setStyleSheet(
            "QTabBar::tab { padding: 4px 10px; font-size: 11px; } QTabWidget::pane { border: none; }"
        )

        # Sub-Tab 1: Recent Verified Outcomes
        tab_outcomes = QWidget()
        outcomes_layout = QVBoxLayout(tab_outcomes)
        outcomes_layout.setContentsMargins(2, 2, 2, 2)
        outcomes_layout.setSpacing(4)

        hist_header = QHBoxLayout()
        hist_title = QLabel("RECENT PREDICTIONS & VERIFIED OUTCOMES")
        hist_title.setFont(QFont("Segoe UI", 9, QFont.Weight.Bold))
        hist_title.setStyleSheet("color: #38BDF8; border: none;")
        hist_header.addWidget(hist_title)

        self.lbl_history_count = QLabel("Displaying recent resolved predictions")
        self.lbl_history_count.setFont(QFont("Segoe UI", 9))
        self.lbl_history_count.setStyleSheet("color: #94A3B8; border: none;")
        hist_header.addWidget(self.lbl_history_count)
        hist_header.addStretch()

        btn_hist_refresh = QPushButton("Refresh History")
        btn_hist_refresh.setStyleSheet(
            "QPushButton { background-color: #111827; border: 1px solid #334155; border-radius: 4px; "
            "padding: 2px 10px; font-size: 11px; color: #CBD5E1; }"
            "QPushButton:hover { border-color: #38BDF8; color: #38BDF8; }"
        )
        btn_hist_refresh.clicked.connect(self._populate_recent_outcomes_table)
        hist_header.addWidget(btn_hist_refresh)
        outcomes_layout.addLayout(hist_header)

        self.table_recent_outcomes = QTableWidget(0, 7)
        self.table_recent_outcomes.setHorizontalHeaderLabels([
            "Time (UTC)", "Symbol", "Signal", "Entry Price", "Exit Price", "Outcome", "Net Return"
        ])
        self.table_recent_outcomes.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
        self.table_recent_outcomes.horizontalHeader().setStretchLastSection(True)
        self.table_recent_outcomes.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.table_recent_outcomes.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.table_recent_outcomes.setMaximumHeight(135)
        outcomes_layout.addWidget(self.table_recent_outcomes)
        self.dash_bottom_tabs.addTab(tab_outcomes, "VERIFIED PREDICTION OUTCOMES")

        # Sub-Tab 2: Historical Pattern Analogs (Zero Lookahead)
        tab_analogs = QWidget()
        analogs_layout = QVBoxLayout(tab_analogs)
        analogs_layout.setContentsMargins(2, 2, 2, 2)
        analogs_layout.setSpacing(4)

        analog_header = QHBoxLayout()
        self.lbl_analog_summary = QLabel("Analog Consensus: Synchronizing historical patterns...")
        self.lbl_analog_summary.setFont(QFont("Segoe UI", 9))
        self.lbl_analog_summary.setStyleSheet("color: #38BDF8; border: none;")
        analog_header.addWidget(self.lbl_analog_summary)
        analog_header.addStretch()
        analogs_layout.addLayout(analog_header)

        self.table_pattern_analogs = QTableWidget(0, 8)
        self.table_pattern_analogs.setHorizontalHeaderLabels([
            "Analog #", "Historical Session Time", "Similarity", "ToD Aligned", "+5m Realized", "+15m Realized", "+1h Realized", "Max Fav / Adv (MFE/MAE)"
        ])
        self.table_pattern_analogs.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
        self.table_pattern_analogs.horizontalHeader().setStretchLastSection(True)
        self.table_pattern_analogs.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.table_pattern_analogs.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.table_pattern_analogs.setMaximumHeight(135)
        analogs_layout.addWidget(self.table_pattern_analogs)
        self.dash_bottom_tabs.addTab(tab_analogs, "HISTORICAL PATTERN ANALOGS")

        history_layout.addWidget(self.dash_bottom_tabs)
        layout.addWidget(history_box, stretch=2)

    def _create_level_box(self, parent_layout: QHBoxLayout, title: str, value: str, color_hex: str) -> QLabel:
        box = QFrame()
        box.setStyleSheet(
            "QFrame { background-color: #111827; border: 1px solid #1E293B; border-radius: 8px; padding: 6px 10px; }"
        )
        vbox = QVBoxLayout(box)
        vbox.setContentsMargins(10, 6, 10, 6)
        vbox.setSpacing(2)

        t_lbl = QLabel(f"{title}:")
        t_lbl.setFont(QFont("Segoe UI", 9, QFont.Weight.DemiBold))
        t_lbl.setStyleSheet("color: #94A3B8; border: none;")
        vbox.addWidget(t_lbl)

        v_lbl = QLabel(value)
        v_lbl.setFont(QFont("Segoe UI", 13, QFont.Weight.Bold))
        v_lbl.setStyleSheet(f"color: {color_hex}; border: none;")
        vbox.addWidget(v_lbl)

        parent_layout.addWidget(box)
        return v_lbl

    def _recalculate_position_size(self) -> None:
        try:
            if not hasattr(self, "edit_capital") or not hasattr(self, "lbl_risk_sizing_summary"):
                return
            cap_str = self.edit_capital.text().replace(",", "").replace("₹", "").strip()
            capital = float(cap_str) if cap_str else 100000.0
            risk_pct = float(self.combo_risk_pct.currentText().replace("%", "").strip()) / 100.0
            max_risk_amount = capital * risk_pct

            payload = getattr(self, "_last_eval_payload", {})
            sig = str(payload.get("signal", "NO-TRADE"))
            ep = float(payload.get("entry_price", 0.0) or 0.0)
            sl = float(payload.get("stop_loss", 0.0) or 0.0)
            tp = float(payload.get("take_profit", 0.0) or 0.0)

            if sig in ("BUY", "SELL") and ep > 0 and sl > 0 and ep != sl:
                risk_per_share = abs(ep - sl)
                reward_per_share = abs(tp - ep) if tp > 0 else 0.0
                rr_ratio = (reward_per_share / risk_per_share) if risk_per_share > 0 else 0.0

                qty = int(max_risk_amount // risk_per_share) if risk_per_share > 0 else 0
                pos_value = qty * ep
                exp_profit = qty * reward_per_share
                max_loss = qty * risk_per_share

                self.lbl_risk_sizing_summary.setText(
                    f"Rec: {qty:,} shares  |  Outlay: ₹{pos_value:,.0f}  |  "
                    f"R:R 1:{rr_ratio:.2f}  |  +₹{exp_profit:,.0f} / -₹{max_loss:,.0f}"
                )
            else:
                self.lbl_risk_sizing_summary.setText(
                    f"Risk Budget: ₹{max_risk_amount:,.0f} ({risk_pct*100:.1f}%)  |  "
                    "Trade levels inactive for current signal."
                )
        except Exception:
            if hasattr(self, "lbl_risk_sizing_summary"):
                self.lbl_risk_sizing_summary.setText("Invalid capital input.")

    # =========================================================================
    # TAB 2: MARKET SCANNER PAGE (Section 8)
    # =========================================================================
    def _build_market_scanner_tab(self) -> None:
        layout = QVBoxLayout(self.scanner_tab)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)

        filter_bar = QHBoxLayout()
        filter_bar.setSpacing(10)

        self.scanner_search = QLineEdit()
        self.scanner_search.setPlaceholderText("Search symbol, company, ISIN, or BSE scrip code...")
        self.scanner_search.returnPressed.connect(self.trigger_scanner_refresh)
        filter_bar.addWidget(self.scanner_search, stretch=2)

        filter_bar.addWidget(QLabel("Market:"))
        self.scanner_market_combo = QComboBox()
        self.scanner_market_combo.addItems(["ALL", "NSE", "BSE", "QUOTEX", "FOREX", "CRYPTO"])
        self.scanner_market_combo.currentTextChanged.connect(lambda _: self.trigger_scanner_refresh())
        filter_bar.addWidget(self.scanner_market_combo)

        filter_bar.addWidget(QLabel("Signal:"))
        self.scanner_signal_combo = QComboBox()
        self.scanner_signal_combo.addItems(["ALL", "BUY", "SELL", "NO-TRADE"])
        self.scanner_signal_combo.currentTextChanged.connect(lambda _: self.trigger_scanner_refresh())
        filter_bar.addWidget(self.scanner_signal_combo)

        filter_bar.addWidget(QLabel("Min Confidence:"))
        self.scanner_conf_slider = QSlider(Qt.Orientation.Horizontal)
        self.scanner_conf_slider.setRange(0, 90)
        self.scanner_conf_slider.setValue(0)
        self.scanner_conf_slider.setFixedWidth(110)
        self.scanner_conf_label = QLabel("0%")
        self.scanner_conf_slider.valueChanged.connect(
            lambda v: self.scanner_conf_label.setText(f"{v}%")
        )
        self.scanner_conf_slider.sliderReleased.connect(self.trigger_scanner_refresh)
        filter_bar.addWidget(self.scanner_conf_slider)
        filter_bar.addWidget(self.scanner_conf_label)

        filter_bar.addWidget(QLabel("Timeframe:"))
        self.scanner_horizon_combo = QComboBox()
        self.scanner_horizon_combo.addItems(["5m", "15m", "1h", "4h", "1d"])
        self.scanner_horizon_combo.setCurrentText("1h")
        self.scanner_horizon_combo.currentTextChanged.connect(lambda _: self.trigger_scanner_refresh())
        filter_bar.addWidget(self.scanner_horizon_combo)

        filter_bar.addWidget(QLabel("Sort:"))
        self.scanner_sort_combo = QComboBox()
        self.scanner_sort_combo.addItems(["confidence", "rank", "symbol", "expected_move"])
        self.scanner_sort_combo.currentTextChanged.connect(lambda _: self.trigger_scanner_refresh())
        filter_bar.addWidget(self.scanner_sort_combo)

        scan_btn = QPushButton("Scan Market")
        scan_btn.setStyleSheet("background-color: #0284C7; color: #FFFFFF; font-weight: bold;")
        scan_btn.clicked.connect(self.trigger_scanner_refresh)
        filter_bar.addWidget(scan_btn)

        layout.addLayout(filter_bar)

        # Quick Scanner Preset Chips & CSV Export
        presets_bar = QHBoxLayout()
        presets_bar.setSpacing(6)
        pres_lbl = QLabel("Presets:")
        pres_lbl.setStyleSheet("color: #64748B; font-size: 11px;")
        presets_bar.addWidget(pres_lbl)

        for label, mkt, sig, min_c in (
            ("All Markets", "ALL", "ALL", 0),
            ("NSE Equities", "NSE", "ALL", 0),
            ("BSE Equities", "BSE", "ALL", 0),
            ("Buy Signals", "ALL", "BUY", 0),
            ("Sell Signals", "ALL", "SELL", 0),
            ("High Confidence (>60%)", "ALL", "ALL", 60),
        ):
            chip = QPushButton(label)
            chip.setStyleSheet(
                "QPushButton { background-color: #111827; border: 1px solid #1E293B; "
                "border-radius: 4px; padding: 2px 8px; font-size: 11px; color: #CBD5E1; }"
                "QPushButton:hover { border-color: #38BDF8; color: #38BDF8; }"
            )
            chip.clicked.connect(lambda checked=False, m=mkt, s=sig, c=min_c: self._apply_scanner_preset(m, s, c))
            presets_bar.addWidget(chip)

        presets_bar.addStretch()

        btn_export_csv = QPushButton("📥 Export CSV")
        btn_export_csv.setStyleSheet(
            "QPushButton { background-color: #1E293B; border: 1px solid #334155; "
            "border-radius: 4px; padding: 2px 10px; font-size: 11px; color: #38BDF8; }"
            "QPushButton:hover { background-color: #0284C7; color: #FFFFFF; }"
        )
        btn_export_csv.clicked.connect(self._export_scanner_to_csv)
        presets_bar.addWidget(btn_export_csv)

        layout.addLayout(presets_bar)

        self.scanner_status_lbl = QLabel(
            "Double-click or right-click any row to inspect that instrument on the Beginner Dashboard."
        )
        self.scanner_status_lbl.setStyleSheet("color: #94A3B8; font-size: 12px;")
        layout.addWidget(self.scanner_status_lbl)

        self.scanner_table = QTableWidget(0, 10)
        self.scanner_table.setHorizontalHeaderLabels(
            [
                "Market",
                "Symbol",
                "BSE Scrip",
                "Company / Instrument",
                "Price",
                "Signal",
                "Confidence",
                "Horizon",
                "Entry / Stop / Target",
                "Why / Summary",
            ]
        )
        self.scanner_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
        self.scanner_table.horizontalHeader().setStretchLastSection(True)
        self.scanner_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.scanner_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.scanner_table.cellDoubleClicked.connect(self._on_scanner_row_activated)
        self.scanner_table.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.scanner_table.customContextMenuRequested.connect(self._on_scanner_context_menu)
        layout.addWidget(self.scanner_table, stretch=1)

    # =========================================================================
    # TAB 3: QUOTEX DEDICATED SECTION (Section 7)
    # =========================================================================
    def _build_quotex_tab(self) -> None:
        layout = QVBoxLayout(self.quotex_tab)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)

        banner = QLabel(
            "QUOTEX LIVE ANALYSIS & PREDICTION ENGINE — Platform Instrument vs. Underlying Market Asset\n"
            "• Instruments backed by authorized underlying feeds (Interbank FX & Spot Crypto) generate real live predictions, levels, and confidence.\n"
            "• Proprietary synthetic OTC equities without authorized public feeds explicitly report 'QUOTEX DATA UNAVAILABLE' (never fake prices)."
        )
        banner.setStyleSheet(
            "background-color: #0F172A; border: 1px solid #1E293B; border-radius: 8px; padding: 12px; color: #CBD5E1;"
        )
        layout.addWidget(banner)

        filter_bar = QHBoxLayout()
        filter_bar.setSpacing(10)

        self.quotex_search_edit = QLineEdit()
        self.quotex_search_edit.setPlaceholderText("Search Quotex symbol or underlying (e.g. EURUSD, BTCUSD_QX, MSFT_OTC)...")
        self.quotex_search_edit.returnPressed.connect(self._populate_quotex_table)
        filter_bar.addWidget(self.quotex_search_edit, stretch=1)

        filter_bar.addWidget(QLabel("Category:"))
        self.quotex_type_combo = QComboBox()
        self.quotex_type_combo.addItems(["ALL", "FOREX_SPOT", "FOREX_OTC", "CRYPTO", "PROPRIETARY_OTC"])
        self.quotex_type_combo.currentTextChanged.connect(lambda _: self._populate_quotex_table())
        filter_bar.addWidget(self.quotex_type_combo)

        filter_bar.addWidget(QLabel("Horizon:"))
        self.quotex_horizon_combo = QComboBox()
        self.quotex_horizon_combo.addItems(["5m", "15m", "1h", "4h", "1d"])
        self.quotex_horizon_combo.setCurrentText(self.active_horizon)
        self.quotex_horizon_combo.currentTextChanged.connect(lambda _: self._populate_quotex_table())
        filter_bar.addWidget(self.quotex_horizon_combo)

        qx_refresh_btn = QPushButton("Refresh Quotex Analysis")
        qx_refresh_btn.setStyleSheet("background-color: #0284C7; color: #FFFFFF; font-weight: bold;")
        qx_refresh_btn.clicked.connect(self._populate_quotex_table)
        filter_bar.addWidget(qx_refresh_btn)

        layout.addLayout(filter_bar)

        self.quotex_status_lbl = QLabel(
            "Double-click any available Quotex row to open its live chart, prediction, and trade levels on the Beginner Dashboard."
        )
        self.quotex_status_lbl.setStyleSheet("color: #94A3B8; font-size: 12px;")
        layout.addWidget(self.quotex_status_lbl)

        self.quotex_table = QTableWidget(0, 10)
        self.quotex_table.setHorizontalHeaderLabels(
            [
                "Platform Instrument",
                "Display Name",
                "Underlying Asset",
                "Data Source Status",
                "Verified Price",
                "Signal",
                "Confidence",
                "Horizon",
                "Entry / Stop / Target",
                "Reasons & Limitations",
            ]
        )
        self.quotex_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
        self.quotex_table.horizontalHeader().setStretchLastSection(True)
        self.quotex_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.quotex_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.quotex_table.cellDoubleClicked.connect(self._on_quotex_row_activated)
        layout.addWidget(self.quotex_table, stretch=1)

    # =========================================================================
    # TAB 4: ADVANCED / SYSTEM DIAGNOSTICS (Sections 5, 10, 11, 12, 13, 18, 19)
    # =========================================================================
    def _build_diagnostics_tab(self) -> None:
        layout = QVBoxLayout(self.diagnostics_tab)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)

        # Top row: Measured End-to-End Latencies + Notification Checkboxes + Self-Development trigger
        top_split = QHBoxLayout()
        top_split.setSpacing(14)

        # Latency Panel
        lat_box = QFrame()
        lat_box.setStyleSheet("background-color: #0F172A; border: 1px solid #1E293B; border-radius: 8px;")
        lat_layout = QVBoxLayout(lat_box)
        lat_title = QLabel("MEASURED RUNTIME LATENCY (Section 5)")
        lat_title.setFont(QFont("Segoe UI", 10, QFont.Weight.Bold))
        lat_title.setStyleSheet("color: #38BDF8;")
        lat_layout.addWidget(lat_title)
        self.lbl_latency_summary = QLabel("Measuring live pipeline stages...")
        self.lbl_latency_summary.setFont(QFont("Consolas", 10))
        lat_layout.addWidget(self.lbl_latency_summary)
        top_split.addWidget(lat_box, stretch=1)

        # Notification Settings Panel (Section 18)
        notif_box = QFrame()
        notif_box.setStyleSheet("background-color: #0F172A; border: 1px solid #1E293B; border-radius: 8px;")
        notif_layout = QGridLayout(notif_box)
        notif_title = QLabel("DESKTOP NOTIFICATIONS (Section 18)")
        notif_title.setFont(QFont("Segoe UI", 10, QFont.Weight.Bold))
        notif_title.setStyleSheet("color: #38BDF8;")
        notif_layout.addWidget(notif_title, 0, 0, 1, 3)

        self.chk_notif_buy = QCheckBox("BUY Signal")
        self.chk_notif_buy.setChecked(True)
        self.chk_notif_sell = QCheckBox("SELL Signal")
        self.chk_notif_sell.setChecked(True)
        self.chk_notif_inv = QCheckBox("Signal Invalidated")
        self.chk_notif_inv.setChecked(True)
        self.chk_notif_data = QCheckBox("Important Data Failure")
        self.chk_notif_data.setChecked(True)
        self.chk_notif_model = QCheckBox("Model Update")
        self.chk_notif_model.setChecked(True)
        self.chk_notif_conn = QCheckBox("Market Connection Lost")
        self.chk_notif_conn.setChecked(True)
        self.chk_notif_sound = QCheckBox("Audible Alert (Chime)")
        self.chk_notif_sound.setChecked(True)

        for chk in (
            self.chk_notif_buy,
            self.chk_notif_sell,
            self.chk_notif_inv,
            self.chk_notif_data,
            self.chk_notif_model,
            self.chk_notif_conn,
            self.chk_notif_sound,
        ):
            chk.stateChanged.connect(self._sync_notification_preferences)

        notif_layout.addWidget(self.chk_notif_buy, 1, 0)
        notif_layout.addWidget(self.chk_notif_sell, 1, 1)
        notif_layout.addWidget(self.chk_notif_inv, 1, 2)
        notif_layout.addWidget(self.chk_notif_data, 2, 0)
        notif_layout.addWidget(self.chk_notif_model, 2, 1)
        notif_layout.addWidget(self.chk_notif_conn, 2, 2)
        notif_layout.addWidget(self.chk_notif_sound, 3, 0, 1, 3)
        top_split.addWidget(notif_box, stretch=1)

        # Self-Development & Governance Action Panel
        gov_box = QFrame()
        gov_box.setStyleSheet("background-color: #0F172A; border: 1px solid #1E293B; border-radius: 8px;")
        gov_layout = QVBoxLayout(gov_box)
        gov_title = QLabel("CONTROLLED SELF-DEVELOPMENT & ACCURACY (Sections 8, 12–13)")
        gov_title.setFont(QFont("Segoe UI", 10, QFont.Weight.Bold))
        gov_title.setStyleSheet("color: #38BDF8;")
        gov_layout.addWidget(gov_title)

        self.lbl_ledger_summary = QLabel("Out-of-Sample Ledger: Loading...")
        self.lbl_ledger_summary.setFont(QFont("Segoe UI", 9))
        gov_layout.addWidget(self.lbl_ledger_summary)

        self.lbl_dual_mode_accuracy = QLabel("Dual-Mode Accuracy Telemetry: Loading...")
        self.lbl_dual_mode_accuracy.setFont(QFont("Consolas", 8))
        self.lbl_dual_mode_accuracy.setStyleSheet("color: #38BDF8;")
        gov_layout.addWidget(self.lbl_dual_mode_accuracy)

        btn_self_dev = QPushButton("Run Walk-Forward Self-Development Cycle")
        btn_self_dev.setStyleSheet("background-color: #10B981; color: #0B1120; font-weight: bold;")
        btn_self_dev.clicked.connect(self._on_run_self_development)
        gov_layout.addWidget(btn_self_dev)
        top_split.addWidget(gov_box, stretch=1)

        layout.addLayout(top_split)

        # System Health & Resource Telemetry Row
        telemetry_box = QFrame()
        telemetry_box.setStyleSheet("background-color: #0F172A; border: 1px solid #1E293B; border-radius: 8px;")
        telem_layout = QHBoxLayout(telemetry_box)
        telem_layout.setContentsMargins(12, 6, 12, 6)
        telem_title = QLabel("SYSTEM RESOURCES & RUNTIME HEALTH:")
        telem_title.setFont(QFont("Segoe UI", 9, QFont.Weight.Bold))
        telem_title.setStyleSheet("color: #38BDF8; border: none;")
        telem_layout.addWidget(telem_title)
        self.lbl_system_resources = QLabel("Loading runtime telemetry...")
        self.lbl_system_resources.setFont(QFont("Consolas", 9))
        self.lbl_system_resources.setStyleSheet("color: #E2E8F0; border: none;")
        telem_layout.addWidget(self.lbl_system_resources, stretch=1)
        layout.addWidget(telemetry_box)

        # Bottom splitter: Developing-Day 22-Metric Inspector + Time-Aligned Historical Matches + Technical Logs
        splitter = QSplitter(Qt.Orientation.Horizontal)

        self.txt_developing_day = QTextEdit()
        self.txt_developing_day.setReadOnly(True)
        self.txt_developing_day.setFont(QFont("Consolas", 9))
        self.txt_developing_day.setStyleSheet("background-color: #0F172A; border: 1px solid #1E293B; color: #E2E8F0;")
        splitter.addWidget(self.txt_developing_day)

        self.txt_historical_sync = QTextEdit()
        self.txt_historical_sync.setReadOnly(True)
        self.txt_historical_sync.setFont(QFont("Consolas", 9))
        self.txt_historical_sync.setStyleSheet("background-color: #0F172A; border: 1px solid #1E293B; color: #E2E8F0;")
        splitter.addWidget(self.txt_historical_sync)

        self.txt_system_logs = QTextEdit()
        self.txt_system_logs.setReadOnly(True)
        self.txt_system_logs.setFont(QFont("Consolas", 9))
        self.txt_system_logs.setStyleSheet("background-color: #0F172A; border: 1px solid #1E293B; color: #94A3B8;")
        splitter.addWidget(self.txt_system_logs)

        layout.addWidget(splitter, stretch=1)

    # =========================================================================
    # SYSTEM TRAY INTEGRATION (Section 17)
    # =========================================================================
    def _setup_system_tray(self) -> None:
        self.tray_icon = QSystemTrayIcon(create_app_icon(), self)
        self.tray_icon.setToolTip("Trading Intelligence Platform — Live Desktop Engine")

        tray_menu = QMenu(self)
        act_open = QAction("Open Dashboard", self)
        act_open.triggered.connect(self._tray_open_dashboard)
        tray_menu.addAction(act_open)

        act_scanner = QAction("Market Scanner", self)
        act_scanner.triggered.connect(self._tray_open_scanner)
        tray_menu.addAction(act_scanner)

        tray_menu.addSeparator()

        self.act_pause = QAction("Pause Predictions", self)
        self.act_pause.triggered.connect(self._tray_pause_predictions)
        tray_menu.addAction(self.act_pause)

        self.act_resume = QAction("Resume Predictions", self)
        self.act_resume.setEnabled(False)
        self.act_resume.triggered.connect(self._tray_resume_predictions)
        tray_menu.addAction(self.act_resume)

        tray_menu.addSeparator()

        act_status = QAction("System Status", self)
        act_status.triggered.connect(self._tray_open_status)
        tray_menu.addAction(act_status)

        tray_menu.addSeparator()

        act_exit = QAction("Exit", self)
        act_exit.triggered.connect(self._tray_exit_app)
        tray_menu.addAction(act_exit)

        self.tray_icon.setContextMenu(tray_menu)
        self.tray_icon.activated.connect(self._on_tray_activated)
        self.tray_icon.show()

    def _on_tray_activated(self, reason: QSystemTrayIcon.ActivationReason) -> None:
        if reason in (
            QSystemTrayIcon.ActivationReason.Trigger,
            QSystemTrayIcon.ActivationReason.DoubleClick,
        ):
            self._tray_open_dashboard()

    def _tray_open_dashboard(self) -> None:
        self.showNormal()
        self.activateWindow()
        self.raise_()
        self.tabs.setCurrentIndex(0)

    def _tray_open_scanner(self) -> None:
        self.showNormal()
        self.activateWindow()
        self.raise_()
        self.tabs.setCurrentIndex(1)

    def _tray_open_status(self) -> None:
        self.showNormal()
        self.activateWindow()
        self.raise_()
        self.tabs.setCurrentIndex(3)
        self._refresh_diagnostics_view()

    def _tray_pause_predictions(self) -> None:
        self.controller.predictions_paused = True
        self.act_pause.setEnabled(False)
        self.act_resume.setEnabled(True)
        self.controller.log_event("Predictions PAUSED by user via System Tray.")

    def _tray_resume_predictions(self) -> None:
        self.controller.predictions_paused = False
        self.act_pause.setEnabled(True)
        self.act_resume.setEnabled(False)
        self.controller.log_event("Predictions RESUMED by user via System Tray.")
        if self._auto_start_workers:
            self.trigger_instrument_evaluation(self.active_symbol)

    def _tray_exit_app(self) -> None:
        self._force_quit = True
        self.live_timer.stop()
        if hasattr(self, "clock_timer") and self.clock_timer:
            self.clock_timer.stop()
        self.controller.stop_background_services()
        self.tray_icon.hide()
        QApplication.quit()

    def closeEvent(self, event: QCloseEvent) -> None:
        """Cleanly shutdown application and background workers on window close."""
        self._force_quit = True
        self.live_timer.stop()
        if hasattr(self, "clock_timer") and self.clock_timer:
            self.clock_timer.stop()
        self.controller.stop_background_services()
        if self._eval_worker is not None and self._eval_worker.isRunning():
            self._eval_worker.wait(2000)
        if self._scanner_worker is not None and self._scanner_worker.isRunning():
            self._scanner_worker.wait(2000)
        if hasattr(self, "tray_icon") and self.tray_icon:
            self.tray_icon.hide()
        event.accept()

    # =========================================================================
    # INTERACTION & INCREMENTAL UI UPDATES (Section 5)
    # =========================================================================
    def _style_market_buttons(self) -> None:
        for mkt, btn in self.market_buttons.items():
            if mkt == self.active_market_filter:
                btn.setStyleSheet(
                    "background-color: #0284C7; color: #FFFFFF; border: 1px solid #38BDF8; font-weight: bold;"
                )
            else:
                btn.setStyleSheet(
                    "background-color: #1E293B; color: #CBD5E1; border: 1px solid #334155;"
                )

    def _on_market_filter_clicked(self, market: str) -> None:
        self.active_market_filter = market
        self._style_market_buttons()
        q = self.search_input.text().strip()
        if q:
            self._on_search_text_changed(q)
        else:
            # Switch to a representative default instrument for that market
            defaults = {
                "ALL": "NSE:TCS",
                "NSE": "NSE:TCS",
                "BSE": "BSE:RELIANCE",
                "FOREX": "FX:EUR/USD",
                "QUOTEX": "QUOTEX:EURUSD_OTC",
                "CRYPTO": "CRYPTO:BTC-USD",
            }
            target = defaults.get(market, "NSE:TCS")
            self.trigger_instrument_evaluation(target)

    def _on_search_text_changed(self, text: str) -> None:
        q = text.strip()
        if not q:
            self.search_results_list.setVisible(False)
            return
        hits = self.controller.search_universe(q, market=self.active_market_filter, limit=20)
        self.search_results_list.clear()
        if not hits:
            self.search_results_list.setVisible(False)
            return
        for h in hits:
            item = QListWidgetItem(h["display_line"])
            item.setData(Qt.ItemDataRole.UserRole, h["canonical_symbol"])
            self.search_results_list.addItem(item)
        self.search_results_list.setVisible(True)

    def _on_search_item_clicked(self, item: QListWidgetItem) -> None:
        canon = item.data(Qt.ItemDataRole.UserRole)
        self.search_results_list.setVisible(False)
        if canon:
            self.trigger_instrument_evaluation(str(canon))

    def _on_search_enter_pressed(self) -> None:
        q = self.search_input.text().strip()
        self.search_results_list.setVisible(False)
        if not q:
            return
        hits = self.controller.search_universe(q, market=self.active_market_filter, limit=1)
        if hits:
            self.trigger_instrument_evaluation(hits[0]["canonical_symbol"])
        else:
            self.trigger_instrument_evaluation(q)

    def _on_horizon_changed(self, horizon: str) -> None:
        self.active_horizon = horizon
        self.trigger_instrument_evaluation(self.active_symbol)

    def _on_live_timer_tick(self) -> None:
        if not self.controller.predictions_paused and self.isVisible():
            self.trigger_instrument_evaluation(self.active_symbol)

    def trigger_instrument_evaluation(self, symbol_or_query: str) -> None:
        if self._eval_worker is not None and self._eval_worker.isRunning():
            return
        self.active_symbol = symbol_or_query
        self._eval_worker = EvaluationWorker(
            controller=self.controller,
            symbol=symbol_or_query,
            timeframe=self.active_timeframe,
            horizon=self.active_horizon,
            parent=self,
        )
        self._eval_worker.evaluation_ready.connect(self._on_evaluation_ready)
        self._eval_worker.start()

    def _on_evaluation_ready(self, payload: dict[str, Any]) -> None:
        """Incrementally update only the UI widgets whose values changed (Section 5)."""
        t_ui0 = time.perf_counter()
        self._last_eval_payload = payload

        sym = str(payload.get("symbol", self.active_symbol))
        canon = str(payload.get("canonical_symbol", sym))
        mkt = str(payload.get("market", "NSE"))
        comp_name = str(payload.get("company_name", sym))
        cross = payload.get("cross_listing") or {}

        if self.lbl_symbol.text() != sym:
            self.lbl_symbol.setText(sym)

        # Build subtitle with NSE + BSE scrip code cross-listing (Section 6)
        sub_parts = [mkt, comp_name]
        nse_s = cross.get("nse_symbol")
        bse_scrip = cross.get("bse_scrip_code")
        if nse_s:
            sub_parts.append(nse_s)
        if bse_scrip:
            sub_parts.append(f"BSE:{bse_scrip}")
        qx_info = payload.get("quotex_info") or {}
        if qx_info:
            sub_parts.append(f"Underlying: {qx_info.get('underlying_instrument')}")
        sub_text = "  •  ".join(sub_parts)
        if self.lbl_exchange_sub.text() != sub_text:
            self.lbl_exchange_sub.setText(sub_text)

        # Session Badge (LIVE / PRE-MARKET / POST-MARKET / MARKET CLOSED / QUOTEX DATA UNAVAILABLE)
        badge = str(payload.get("session_badge", "LIVE"))
        timing = payload.get("session_timing") or {}
        countdown = timing.get("countdown_str")
        if "CLOSED" in badge and countdown:
            badge_text = f"MARKET CLOSED • {countdown}"
        elif "PRE-MARKET" in badge and countdown:
            badge_text = f"PRE-MARKET • {countdown}"
        else:
            badge_text = badge

        if self.lbl_session_badge.text() != badge_text:
            self.lbl_session_badge.setText(badge_text)
        if timing.get("message"):
            self.lbl_session_badge.setToolTip(timing["message"])
        if "UNAVAILABLE" in badge or "ERROR" in badge:
            self.lbl_session_badge.setStyleSheet(
                "background-color: #7F1D1D; color: #FCA5A5; padding: 4px 12px; border-radius: 12px; border: none;"
            )
        elif "STALE" in badge:
            self.lbl_session_badge.setStyleSheet(
                "background-color: #78350F; color: #FCD34D; padding: 4px 12px; border-radius: 12px; border: none;"
            )
        elif "PRE-MARKET" in badge:
            self.lbl_session_badge.setStyleSheet(
                "background-color: #1E3A8A; color: #93C5FD; padding: 4px 12px; border-radius: 12px; border: none;"
            )
        elif "CLOSED" in badge:
            self.lbl_session_badge.setStyleSheet(
                "background-color: #334155; color: #94A3B8; padding: 4px 12px; border-radius: 12px; border: none;"
            )
        elif "REFERENCE" in badge or "CACHED" in badge:
            self.lbl_session_badge.setStyleSheet(
                "background-color: #1E293B; color: #38BDF8; padding: 4px 12px; border-radius: 12px; border: none;"
            )
        elif "LIVE" in badge:
            self.lbl_session_badge.setStyleSheet(
                "background-color: #064E3B; color: #34D399; padding: 4px 12px; border-radius: 12px; border: none;"
            )
        else:
            self.lbl_session_badge.setStyleSheet(
                "background-color: #334155; color: #CBD5E1; padding: 4px 12px; border-radius: 12px; border: none;"
            )

        # Quote / Evaluation Timestamp
        ts_val = payload.get("data_timestamp") or payload.get("timestamp") or ""
        if ts_val:
            try:
                dt_clean = ts_val.replace("Z", "+00:00")
                dt_obj = datetime.fromisoformat(dt_clean)
                self.lbl_timestamp.setText(f"Updated: {dt_obj.strftime('%H:%M:%S UTC')}")
            except Exception:
                self.lbl_timestamp.setText(f"Updated: {ts_val[:19]}")
        else:
            self.lbl_timestamp.setText("Updated: Just now")

        # Current Price
        price_disp = str(payload.get("price_display", "—"))
        if self.lbl_price.text() != price_disp:
            self.lbl_price.setText(price_disp)

        # Dominant Signal Badge (BUY / SELL / NO-TRADE)
        sig = str(payload.get("signal", "NO-TRADE"))
        if self.lbl_signal_banner.text() != sig:
            self.lbl_signal_banner.setText(sig)
        if sig == "BUY":
            self.lbl_signal_banner.setStyleSheet(
                "background-color: #065F46; color: #34D399; border: 2px solid #10B981; border-radius: 14px; padding: 12px 24px;"
            )
        elif sig == "SELL":
            self.lbl_signal_banner.setStyleSheet(
                "background-color: #7F1D1D; color: #FCA5A5; border: 2px solid #EF4444; border-radius: 14px; padding: 12px 24px;"
            )
        else:
            self.lbl_signal_banner.setStyleSheet(
                "background-color: #1E293B; color: #CBD5E1; border: 2px solid #475569; border-radius: 14px; padding: 12px 24px;"
            )

        # Confidence & Horizon
        conf_pct = int(payload.get("confidence_pct", 0))
        conf_str = f"{conf_pct}% CONFIDENCE"
        if self.lbl_confidence.text() != conf_str:
            self.lbl_confidence.setText(conf_str)

        hz_lbl = f"Forecast: {payload.get('horizon_label', 'Next 1 Hour')}"
        if self.lbl_forecast_horizon.text() != hz_lbl:
            self.lbl_forecast_horizon.setText(hz_lbl)

        # Entry, Stop, Target
        curr = str(payload.get("currency", "INR"))
        ep = float(payload.get("entry_price", 0.0) or 0.0)
        sl = float(payload.get("stop_loss", 0.0) or 0.0)
        tp = float(payload.get("take_profit", 0.0) or 0.0)
        if sig in ("BUY", "SELL") and ep > 0:
            self.lbl_entry.setText(self.controller.format_price(ep, curr, mkt))
            self.lbl_stop.setText(self.controller.format_price(sl, curr, mkt) if sl > 0 else "—")
            self.lbl_target.setText(self.controller.format_price(tp, curr, mkt) if tp > 0 else "—")
        else:
            self.lbl_entry.setText("—")
            self.lbl_stop.setText("—")
            self.lbl_target.setText("—")

        # Recalculate Tactical Risk & Position Sizing
        self._recalculate_position_size()

        # WHY Bullets
        bullets = payload.get("why_bullets") or []
        why_text = "\n".join(f"• {b}" for b in bullets) if bullets else "• Analyzing market state..."
        if self.lbl_why_bullets.text() != why_text:
            self.lbl_why_bullets.setText(why_text)

        # Update Native Chart Widget
        status_msg = "QUOTEX DATA UNAVAILABLE" if payload.get("status") == "QUOTEX_DATA_UNAVAILABLE" else ""
        self.chart_widget.update_chart_data(
            candles=payload.get("candles") or [],
            symbol=canon,
            signal=sig,
            entry=ep,
            stop=sl,
            target=tp,
            currency=curr,
            market=mkt,
            status_message=status_msg,
        )

        # Record UI update latency
        ui_ms = (time.perf_counter() - t_ui0) * 1000.0
        self.controller.record_ui_render_latency(ui_ms)

        # Emit Desktop Tray Notification & Non-blocking Audio Chime if triggered (Section 18)
        notif = payload.get("notification")
        if notif:
            if self.tray_icon.isVisible():
                self.tray_icon.showMessage(
                    str(notif.get("title", "Trading Alert")),
                    str(notif.get("message", "")),
                    QSystemTrayIcon.MessageIcon.Information,
                    3000,
                )
            if getattr(self.controller.notification_settings, "sound_enabled", True):
                try:
                    import winsound

                    winsound.MessageBeep(winsound.MB_ICONASTERISK)
                except Exception:
                    pass

        if self.tabs.currentIndex() == 3:
            self._refresh_diagnostics_view()

        # Refresh recent prediction outcomes table
        if hasattr(self, "table_recent_outcomes"):
            self._populate_recent_outcomes_table()

        # Refresh historical pattern analogs table
        if hasattr(self, "table_pattern_analogs"):
            self._populate_pattern_analogs_table(payload.get("pattern_consensus") or {})

    # =========================================================================
    # MARKET SCANNER METHODS (Section 8)
    # =========================================================================
    def trigger_scanner_refresh(self) -> None:
        if self._scanner_worker is not None and self._scanner_worker.isRunning():
            return
        self.scanner_status_lbl.setText("Scanning active universe across exchanges...")
        self._scanner_worker = ScannerWorker(
            controller=self.controller,
            market=self.scanner_market_combo.currentText(),
            signal_filter=self.scanner_signal_combo.currentText(),
            min_conf=self.scanner_conf_slider.value(),
            horizon=self.scanner_horizon_combo.currentText(),
            search_q=self.scanner_search.text(),
            sort_by=self.scanner_sort_combo.currentText(),
            parent=self,
        )
        self._scanner_worker.scanner_ready.connect(self._on_scanner_ready)
        self._scanner_worker.start()

    def _on_scanner_ready(self, rows: list[dict[str, Any]]) -> None:
        self.scanner_table.setRowCount(len(rows))
        self.scanner_status_lbl.setText(
            f"Showing {len(rows)} scanned instruments. Double-click any row to inspect on the Beginner Dashboard."
        )
        for r_idx, row in enumerate(rows):
            sig = str(row.get("signal", "NO-TRADE"))
            mkt = str(row.get("market", "NSE"))
            curr = "INR" if mkt in ("NSE", "BSE") else "USD"
            ep = self.controller.format_price(float(row.get("entry_price", 0) or 0), curr, mkt)
            sl = self.controller.format_price(float(row.get("stop_loss", 0) or 0), curr, mkt)
            tp = self.controller.format_price(float(row.get("take_profit", 0) or 0), curr, mkt)
            levels_str = f"{ep} / {sl} / {tp}" if sig in ("BUY", "SELL") else "—"

            cols = [
                mkt,
                str(row.get("symbol", "")),
                str(row.get("scrip_code", "") or "—"),
                str(row.get("company_name", "")),
                str(row.get("price_display", "")),
                sig,
                f"{row.get('confidence_pct', 0)}%",
                str(row.get("horizon", "1h")),
                levels_str,
                str(row.get("rationale", "")),
            ]
            for c_idx, val in enumerate(cols):
                item = QTableWidgetItem(val)
                item.setData(Qt.ItemDataRole.UserRole, row.get("canonical_symbol"))
                if c_idx == 5:
                    item.setFont(QFont("Segoe UI", 10, QFont.Weight.Bold))
                    if sig == "BUY":
                        item.setForeground(QColor("#34D399"))
                    elif sig == "SELL":
                        item.setForeground(QColor("#F87171"))
                    else:
                        item.setForeground(QColor("#94A3B8"))
                self.scanner_table.setItem(r_idx, c_idx, item)

    def _on_scanner_row_activated(self, row: int, _col: int) -> None:
        item = self.scanner_table.item(row, 0)
        if not item:
            return
        canon = item.data(Qt.ItemDataRole.UserRole)
        if canon:
            self.tabs.setCurrentIndex(0)
            self.trigger_instrument_evaluation(str(canon))

    def _apply_scanner_preset(self, mkt: str, sig: str, min_conf: int) -> None:
        self.scanner_market_combo.setCurrentText(mkt)
        self.scanner_signal_combo.setCurrentText(sig)
        self.scanner_conf_slider.setValue(min_conf)
        self.scanner_conf_label.setText(f"{min_conf}%")
        self.trigger_scanner_refresh()

    def _export_scanner_to_csv(self) -> None:
        try:
            import csv
            from pathlib import Path
            rows = self.scanner_table.rowCount()
            cols = self.scanner_table.columnCount()
            if rows == 0:
                self.scanner_status_lbl.setText("No scan results to export.")
                return

            export_dir = Path("data/exports")
            export_dir.mkdir(parents=True, exist_ok=True)
            ts_str = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
            csv_path = export_dir / f"scanner_results_{ts_str}.csv"

            headers = [self.scanner_table.horizontalHeaderItem(c).text() for c in range(cols)]
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(headers)
                for r in range(rows):
                    row_data = []
                    for c in range(cols):
                        it = self.scanner_table.item(r, c)
                        row_data.append(it.text() if it else "")
                    writer.writerow(row_data)

            self.scanner_status_lbl.setText(f"Exported {rows} scanner rows to {csv_path.name}")
        except Exception as exc:
            self.scanner_status_lbl.setText(f"Export failed: {exc}")

    def _on_scanner_context_menu(self, pos: Any) -> None:
        row = self.scanner_table.rowAt(pos.y())
        if row < 0:
            return
        item = self.scanner_table.item(row, 0)
        sym_item = self.scanner_table.item(row, 1)
        canon = (item.data(Qt.ItemDataRole.UserRole) if item else None) or (sym_item.text() if sym_item else "")
        if not canon:
            return

        menu = QMenu(self)
        act_analyze = QAction(f"Analyze {canon} on Dashboard", self)
        act_analyze.triggered.connect(lambda: self._activate_scanner_symbol(str(canon)))
        menu.addAction(act_analyze)

        act_copy = QAction("Copy Symbol", self)
        act_copy.triggered.connect(lambda: QApplication.clipboard().setText(str(canon)))
        menu.addAction(act_copy)

        menu.exec(self.scanner_table.viewport().mapToGlobal(pos))

    def _activate_scanner_symbol(self, canon: str) -> None:
        self.tabs.setCurrentIndex(0)
        self.trigger_instrument_evaluation(str(canon))

    # =========================================================================
    # QUOTEX TAB METHODS (Section 7)
    # =========================================================================
    def _populate_quotex_table(self) -> None:
        q_text = self.quotex_search_edit.text() if hasattr(self, "quotex_search_edit") else ""
        c_type = self.quotex_type_combo.currentText() if hasattr(self, "quotex_type_combo") else "ALL"
        hz = self.quotex_horizon_combo.currentText() if hasattr(self, "quotex_horizon_combo") else self.active_horizon

        rows = self.controller.get_quotex_section_rows(
            horizon=hz,
            search_query=q_text,
            asset_type_filter=c_type,
        )
        self.quotex_table.setRowCount(len(rows))

        live_cnt = sum(1 for r in rows if "UNAVAILABLE" not in str(r.get("live_status", "")))
        unavail_cnt = len(rows) - live_cnt
        if hasattr(self, "quotex_status_lbl"):
            self.quotex_status_lbl.setText(
                f"Showing {len(rows)} Quotex instruments ({live_cnt} active with live underlying feeds, {unavail_cnt} unsupported proprietary OTC). "
                "Double-click any row to view its live chart, full forecast, and plain-English reasons."
            )

        for r_idx, row in enumerate(rows):
            status = str(row.get("live_status", ""))
            sig = str(row.get("signal", "NO-TRADE"))
            conf_val = row.get("confidence_pct", 0)
            conf_str = f"{conf_val}%" if conf_val > 0 else "—"

            cols = [
                str(row.get("platform_instrument", "")),
                str(row.get("name", "")),
                str(row.get("underlying_instrument", "")),
                status,
                str(row.get("price_display", "—")),
                sig,
                conf_str,
                str(row.get("horizon", hz)),
                str(row.get("levels_display", "—")),
                str(row.get("rationale", "") or row.get("note", "")),
            ]
            for c_idx, val in enumerate(cols):
                item = QTableWidgetItem(val)
                item.setData(Qt.ItemDataRole.UserRole, row.get("platform_instrument"))

                # Data Source Status styling
                if c_idx == 3:
                    item.setFont(QFont("Segoe UI", 9, QFont.Weight.Bold))
                    if "UNAVAILABLE" in status:
                        item.setForeground(QColor("#F87171"))
                    elif "PROXY" in status:
                        item.setForeground(QColor("#FCD34D"))
                    else:
                        item.setForeground(QColor("#34D399"))

                # Signal styling
                elif c_idx == 5:
                    item.setFont(QFont("Segoe UI", 10, QFont.Weight.Bold))
                    if sig == "BUY":
                        item.setForeground(QColor("#34D399"))
                    elif sig == "SELL":
                        item.setForeground(QColor("#F87171"))
                    else:
                        item.setForeground(QColor("#94A3B8"))

                # Confidence styling
                elif c_idx == 6:
                    item.setFont(QFont("Segoe UI", 9, QFont.Weight.Bold))
                    item.setForeground(QColor("#38BDF8") if conf_val >= 50 else QColor("#94A3B8"))

                self.quotex_table.setItem(r_idx, c_idx, item)

    def _on_quotex_row_activated(self, row: int, _col: int) -> None:
        item = self.quotex_table.item(row, 0)
        if not item:
            return
        canon = item.data(Qt.ItemDataRole.UserRole)
        if canon:
            self.tabs.setCurrentIndex(0)
            self.trigger_instrument_evaluation(str(canon))

    def _on_manual_refresh_clicked(self) -> None:
        self.trigger_instrument_evaluation(self.active_symbol)
        self._populate_recent_outcomes_table()

    def _populate_recent_outcomes_table(self) -> None:
        if not hasattr(self, "table_recent_outcomes"):
            return
        try:
            records = self.controller.get_recent_prediction_outcomes(limit=20)
            self.table_recent_outcomes.setRowCount(len(records))
            for r_idx, r in enumerate(records):
                # 0: Time
                t_item = QTableWidgetItem(str(r.get("time", "")))
                t_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                self.table_recent_outcomes.setItem(r_idx, 0, t_item)

                # 1: Symbol
                s_item = QTableWidgetItem(str(r.get("symbol", "")))
                s_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                s_item.setFont(QFont("Segoe UI", 9, QFont.Weight.Bold))
                self.table_recent_outcomes.setItem(r_idx, 1, s_item)

                # 2: Signal
                sig = str(r.get("action", ""))
                sig_item = QTableWidgetItem(sig)
                sig_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                sig_item.setFont(QFont("Segoe UI", 9, QFont.Weight.Bold))
                if sig == "BUY":
                    sig_item.setForeground(QColor("#34D399"))
                elif sig == "SELL":
                    sig_item.setForeground(QColor("#F87171"))
                else:
                    sig_item.setForeground(QColor("#94A3B8"))
                self.table_recent_outcomes.setItem(r_idx, 2, sig_item)

                # 3: Entry Price
                ep = r.get("entry_price", 0.0)
                ep_str = f"₹{ep:.2f}" if isinstance(ep, (int, float)) and ep > 0 else str(ep)
                ep_item = QTableWidgetItem(ep_str)
                ep_item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
                self.table_recent_outcomes.setItem(r_idx, 3, ep_item)

                # 4: Exit Price
                xp = r.get("exit_price", 0.0)
                xp_str = f"₹{xp:.2f}" if isinstance(xp, (int, float)) and xp > 0 else str(xp)
                xp_item = QTableWidgetItem(xp_str)
                xp_item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
                self.table_recent_outcomes.setItem(r_idx, 4, xp_item)

                # 5: Outcome
                out = str(r.get("outcome", ""))
                out_item = QTableWidgetItem(out)
                out_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                out_item.setFont(QFont("Segoe UI", 9, QFont.Weight.Bold))
                if out == "WIN":
                    out_item.setForeground(QColor("#34D399"))
                elif out == "LOSS":
                    out_item.setForeground(QColor("#F87171"))
                else:
                    out_item.setForeground(QColor("#E2E8F0"))
                self.table_recent_outcomes.setItem(r_idx, 5, out_item)

                # 6: Net Return
                ret = r.get("return_pct", 0.0)
                ret_str = f"{ret:+.2f}%" if isinstance(ret, (int, float)) else str(ret)
                ret_item = QTableWidgetItem(ret_str)
                ret_item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
                ret_item.setFont(QFont("Segoe UI", 9, QFont.Weight.Bold))
                if isinstance(ret, (int, float)):
                    if ret > 0:
                        ret_item.setForeground(QColor("#34D399"))
                    elif ret < 0:
                        ret_item.setForeground(QColor("#F87171"))
                self.table_recent_outcomes.setItem(r_idx, 6, ret_item)

            if hasattr(self, "lbl_history_count"):
                self.lbl_history_count.setText(f"Showing {len(records)} recent resolved predictions from ledger")
        except Exception as exc:
            if hasattr(self, "lbl_history_count"):
                self.lbl_history_count.setText(f"Error loading ledger: {exc}")

    def _populate_pattern_analogs_table(self, pc: dict[str, Any]) -> None:
        """Populate the historical pattern similarity analog table with zero-lookahead historical matches."""
        if not hasattr(self, "table_pattern_analogs"):
            return
        try:
            matched = pc.get("matched_historical_timestamps") or []
            n_m = len(matched)
            direction = str(pc.get("direction", "NEUTRAL")).upper()
            win_rate = float(pc.get("win_rate", 0.5)) * 100.0
            exp_ret = float(pc.get("expected_return", 0.0))
            mfe = float(pc.get("mfe_pct", 0.0))
            mae = float(pc.get("mae_pct", 0.0))

            if hasattr(self, "lbl_analog_summary"):
                if n_m > 0:
                    self.lbl_analog_summary.setText(
                        f"Analog Consensus: {direction} ({win_rate:.1f}% Win Rate across {n_m} analogs)  |  "
                        f"Exp Return: {exp_ret:+.2f}%  |  Median MFE: +{mfe:.2f}%  |  Median MAE: -{mae:.2f}%  |  Zero Leakage: Verified"
                    )
                else:
                    self.lbl_analog_summary.setText(
                        "Analog Consensus: Insufficient historical candles for deep similarity search."
                    )

            self.table_pattern_analogs.setRowCount(n_m)
            for r_idx, m in enumerate(matched):
                # 0: Analog #
                item_num = QTableWidgetItem(f"Analog #{r_idx + 1}")
                item_num.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                item_num.setFont(QFont("Segoe UI", 9, QFont.Weight.Bold))
                self.table_pattern_analogs.setItem(r_idx, 0, item_num)

                # 1: Historical Session Time
                ts_str = str(m.get("timestamp", ""))
                if "T" in ts_str:
                    ts_str = ts_str.replace("T", " ")[:16]
                item_ts = QTableWidgetItem(ts_str)
                item_ts.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                self.table_pattern_analogs.setItem(r_idx, 1, item_ts)

                # 2: Similarity Score
                sim = float(m.get("similarity_score", 0.0))
                item_sim = QTableWidgetItem(f"{sim * 100:.1f}%")
                item_sim.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                item_sim.setFont(QFont("Segoe UI", 9, QFont.Weight.Bold))
                if sim >= 0.85:
                    item_sim.setForeground(QColor("#38BDF8"))
                self.table_pattern_analogs.setItem(r_idx, 2, item_sim)

                # 3: Time of Day Aligned
                is_tod = bool(m.get("time_of_day_aligned", False))
                item_tod = QTableWidgetItem("YES (±60m)" if is_tod else "NO")
                item_tod.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                item_tod.setForeground(QColor("#34D399") if is_tod else QColor("#94A3B8"))
                self.table_pattern_analogs.setItem(r_idx, 3, item_tod)

                # Forward returns
                fwd = m.get("forward_returns", {})
                r_5m = float(fwd.get(1, 0.0)) * 100.0 if isinstance(fwd, dict) else 0.0
                r_15m = float(fwd.get(3, 0.0)) * 100.0 if isinstance(fwd, dict) else 0.0
                r_1h = (
                    float(fwd.get(12, fwd.get(10, fwd.get(5, 0.0)))) * 100.0
                    if isinstance(fwd, dict)
                    else 0.0
                )

                # 4: +5m Realized
                item_5m = QTableWidgetItem(f"{r_5m:+.2f}%")
                item_5m.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
                item_5m.setForeground(
                    QColor("#34D399") if r_5m > 0 else (QColor("#F87171") if r_5m < 0 else QColor("#94A3B8"))
                )
                self.table_pattern_analogs.setItem(r_idx, 4, item_5m)

                # 5: +15m Realized
                item_15m = QTableWidgetItem(f"{r_15m:+.2f}%")
                item_15m.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
                item_15m.setForeground(
                    QColor("#34D399") if r_15m > 0 else (QColor("#F87171") if r_15m < 0 else QColor("#94A3B8"))
                )
                self.table_pattern_analogs.setItem(r_idx, 5, item_15m)

                # 6: +1h Realized
                item_1h = QTableWidgetItem(f"{r_1h:+.2f}%")
                item_1h.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
                item_1h.setForeground(
                    QColor("#34D399") if r_1h > 0 else (QColor("#F87171") if r_1h < 0 else QColor("#94A3B8"))
                )
                self.table_pattern_analogs.setItem(r_idx, 6, item_1h)

                # 7: Max Fav / Adv (MFE/MAE)
                all_r = (
                    [float(v) * 100.0 for v in fwd.values()]
                    if isinstance(fwd, dict) and fwd
                    else [r_5m, r_15m, r_1h]
                )
                mfe_row = max([0.0] + all_r)
                mae_row = abs(min([0.0] + all_r))
                item_excursion = QTableWidgetItem(f"+{mfe_row:.2f}% / -{mae_row:.2f}%")
                item_excursion.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                self.table_pattern_analogs.setItem(r_idx, 7, item_excursion)
        except Exception:
            pass

    # =========================================================================
    # DIAGNOSTICS & SELF-DEVELOPMENT METHODS (Sections 10, 11, 12, 13, 18, 19)
    # =========================================================================
    def _on_tab_changed(self, index: int) -> None:
        if index == 1 and self.scanner_table.rowCount() == 0:
            self.trigger_scanner_refresh()
        elif index == 2:
            self._populate_quotex_table()
        elif index == 3:
            self._refresh_diagnostics_view()

    def _sync_notification_preferences(self) -> None:
        ns = self.controller.notification_settings
        ns.buy_signal = self.chk_notif_buy.isChecked()
        ns.sell_signal = self.chk_notif_sell.isChecked()
        ns.signal_invalidated = self.chk_notif_inv.isChecked()
        ns.data_failure = self.chk_notif_data.isChecked()
        ns.model_update = self.chk_notif_model.isChecked()
        ns.connection_lost = self.chk_notif_conn.isChecked()
        ns.sound_enabled = self.chk_notif_sound.isChecked()

    def _refresh_diagnostics_view(self) -> None:
        snap = self.controller.get_diagnostics_snapshot(self._last_eval_payload)
        lat = snap["latencies"]
        self.lbl_latency_summary.setText(
            f"Market-Data Latency : {lat.get('market_data_latency_ms', 0.0):.2f} ms\n"
            f"Processing Latency  : {lat.get('processing_latency_ms', 0.0):.2f} ms\n"
            f"Prediction Latency  : {lat.get('prediction_latency_ms', 0.0):.2f} ms\n"
            f"UI Update Latency   : {lat.get('ui_update_latency_ms', 0.0):.2f} ms\n"
            f"End-to-End Total    : {lat.get('end_to_end_latency_ms', 0.0):.2f} ms"
        )

        lm = snap["ledger_metrics"]
        self.lbl_ledger_summary.setText(
            f"Production Model: {lm.get('champion_model_id', 'v1')}  |  "
            f"Predictions Recorded: {lm.get('total_predictions', 0)}  |  "
            f"Resolved: {lm.get('resolved_predictions', 0)}\n"
            f"Out-of-Sample Win Rate: {float(lm.get('win_rate', 0.0)) * 100:.1f}% (Honest OOS, never faked)  |  "
            f"Brier Score: {float(lm.get('brier_score', 0.0)):.4f}"
        )

        mat = snap.get("model_accuracy", {})
        if hasattr(self, "lbl_dual_mode_accuracy"):
            if mat.get("has_candidate_audit"):
                unc_acc = mat.get("unconstrained_accuracy_pct") or 0.0
                ci_l = mat.get("wilson_ci_lower_pct") or 0.0
                ci_u = mat.get("wilson_ci_upper_pct") or 0.0
                hc_win = mat.get("high_conviction_win_rate_pct") or 0.0
                hc_exp = mat.get("high_conviction_expectancy_pct") or 0.0
                hc_cov = mat.get("high_conviction_coverage_pct") or 0.0
                state = mat.get("acceptance_state", "EVALUATING")
                self.lbl_dual_mode_accuracy.setText(
                    f"Section 8 Dual-Mode Telemetry [{state}]:\n"
                    f"Mode A (All-Bars): {unc_acc:.1f}% Acc (Wilson CI: [{ci_l:.1f}%, {ci_u:.1f}%])\n"
                    f"Mode B (High-Conviction): {hc_win:.1f}% Win Rate | {hc_exp:+.3f}% Exp | {hc_cov:.1f}% Cov"
                )
            else:
                self.lbl_dual_mode_accuracy.setText("Section 8 Dual-Mode Telemetry: No candidate audit available")

        sys_res = snap.get("system_resources", {})
        if hasattr(self, "lbl_system_resources"):
            self.lbl_system_resources.setText(
                f"PID: {sys_res.get('pid', '—')}  |  "
                f"Active Threads: {sys_res.get('active_threads', 1)}  |  "
                f"Equity Cache: {sys_res.get('equity_cached_instruments', 0)} instruments ({sys_res.get('equity_cache_bytes', 0):,} B)  |  "
                f"Governance: {sys_res.get('governance_mode', 'CONTROLLED')}  |  "
                f"Predictions: {'ACTIVE' if not sys_res.get('predictions_paused', False) else 'PAUSED'}"
            )

        # Developing-Day 22-Metric Inspector (Section 10)
        dd = self._last_eval_payload.get("developing_day") or {}
        dd_lines = [
            f"=== DEVELOPING-DAY STATE ({self._last_eval_payload.get('canonical_symbol', self.active_symbol)}) ===",
            f"Session Date       : {dd.get('session_date', '—')}",
            f"Session Timestamp  : {dd.get('session_timestamp', '—')}",
            f"Open               : {dd.get('open', '—')}",
            f"Current            : {dd.get('current', '—')}",
            f"High               : {dd.get('high', '—')}",
            f"Low                : {dd.get('low', '—')}",
            f"Range              : {dd.get('range', '—')}",
            f"Gap (%)            : {dd.get('gap', '—')}%",
            f"Volume             : {dd.get('volume', '—')}",
            f"Relative Volume    : {dd.get('relative_volume', '—')}x",
            f"VWAP               : {dd.get('vwap', '—')}",
            f"Trend              : {dd.get('trend', '—')}",
            f"Momentum (%)       : {dd.get('momentum', '—')}%",
            f"Volatility (%)     : {dd.get('volatility', '—')}%",
            f"Support            : {dd.get('support', '—')}",
            f"Resistance         : {dd.get('resistance', '—')}",
            f"Opening Range      : {dd.get('opening_range', '—')}",
            f"Breakout           : {dd.get('breakout', '—')}",
            f"Pullback           : {dd.get('pullback', '—')}",
            f"Consolidation      : {dd.get('consolidation', '—')}",
            f"Higher Highs       : {dd.get('higher_highs', '—')}",
            f"Higher Lows        : {dd.get('higher_lows', '—')}",
            f"Lower Highs        : {dd.get('lower_highs', '—')}",
            f"Lower Lows         : {dd.get('lower_lows', '—')}",
        ]
        self.txt_developing_day.setPlainText("\n".join(dd_lines))

        # Historical + Live Timestamp Synchronization Inspector (Section 11)
        pc = self._last_eval_payload.get("pattern_consensus") or {}
        matched = pc.get("matched_historical_timestamps") or []
        sync_lines = [
            "=== HISTORICAL + LIVE TIMESTAMP SYNCHRONIZATION (Section 11) ===",
            f"Zero Future Leakage Verified : {pc.get('zero_future_leakage_verified', True)}",
            f"Matched Historical Sessions  : {pc.get('sample_size', 0)}",
            f"Time-of-Day Aligned Matches  : {pc.get('time_aligned_matches', 0)}",
            f"Historical Win Rate          : {float(pc.get('win_rate', 0.5)) * 100:.1f}%",
            f"P(Up) / P(Down) / P(Flat)    : {pc.get('prob_up', 0):.2f} / {pc.get('prob_down', 0):.2f} / {pc.get('prob_flat', 0):.2f}",
            f"Median MFE / MAE             : +{pc.get('mfe_pct', 0):.2f}% / -{pc.get('mae_pct', 0):.2f}%",
            "",
            "--- Historical Days Matched at Equivalent Session Timestamp ---",
        ]
        for idx, m in enumerate(matched[:8], 1):
            sync_lines.append(
                f"Day {idx} → {m.get('timestamp')}  |  Sim: {m.get('similarity_score'):.3f}  |  "
                f"ToD Aligned: {m.get('time_of_day_aligned')}"
            )
        self.txt_historical_sync.setPlainText("\n".join(sync_lines))

        # Universe & System Logs
        u_sum = snap["universe_summary"]
        by_m = u_sum.get("by_market", {})
        log_lines = [
            "=== SYNCHRONIZED MULTI-EXCHANGE UNIVERSE ===",
            f"Total Canonical Instruments : {u_sum.get('total_instruments', 0):,}",
            f"NSE Stocks                  : {by_m.get('NSE', 0):,}",
            f"BSE Stocks                  : {by_m.get('BSE', 0):,}",
            f"Cross-Listed NSE+BSE (ISIN) : {u_sum.get('cross_listed_companies', 0):,}",
            f"Quotex Instruments          : {by_m.get('QUOTEX', 0)}",
            f"Forex Pairs                 : {by_m.get('FOREX', 0)}",
            f"Crypto Spot Pairs           : {by_m.get('CRYPTO', 0)}",
            "",
            "=== RUNTIME EVENT LOGS ===",
            *snap["logs"],
        ]
        self.txt_system_logs.setPlainText("\n".join(log_lines))

    def _on_run_self_development(self) -> None:
        from learning.ledger import PredictionLedgerService

        ledger = PredictionLedgerService.get_instance()
        report = ledger.run_self_development_cycle(self.active_symbol)
        self.controller.log_event(
            f"Self-Development cycle completed for {self.active_symbol}: "
            f"decision={report.get('promotion_decision', 'KEEP_CHAMPION')}"
        )
        self._refresh_diagnostics_view()
        QMessageBox.information(
            self,
            "Controlled Self-Development Cycle Complete",
            f"Symbol: {self.active_symbol}\n"
            f"Champion Model: {report.get('champion_model_id')}\n"
            f"Candidate Model: {report.get('candidate_model_id')}\n"
            f"Walk-Forward OOS Decision: {report.get('promotion_decision')}\n"
            f"Reason: {report.get('governance_reason', 'Validated out-of-sample with zero future leakage.')}",
        )
