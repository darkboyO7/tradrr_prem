"""Trading Intelligence Platform — Windows Desktop Application Entrypoint.

Launch directly via:
    python -m desktop.app
or via the packaged Windows executable:
    TradingIntelligencePlatform.exe
"""

from __future__ import annotations

import argparse
import ctypes
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
import sys

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication

from core.system_lock import ProcessFileLock
from desktop.main_window import StartupStatusDialog, TradingIntelligenceDesktopWindow, create_app_icon
from desktop.service_controller import DesktopServiceController

DESKTOP_LOCK_PATH = Path("data/.desktop_app.lock")
DESKTOP_LOG_PATH = Path("logs/desktop_app.log")


def setup_desktop_file_logging() -> None:
    """Ensure logs directory exists and configure file handler to logs/desktop_app.log."""
    try:
        DESKTOP_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.INFO)
        for h in root_logger.handlers:
            if isinstance(h, RotatingFileHandler) and Path(getattr(h, "baseFilename", "")).name == "desktop_app.log":
                return
        file_handler = RotatingFileHandler(
            DESKTOP_LOG_PATH,
            maxBytes=10 * 1024 * 1024,
            backupCount=5,
            encoding="utf-8",
        )
        file_formatter = logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s (PID:%(process)d): %(message)s"
        )
        file_handler.setFormatter(file_formatter)
        file_handler.setLevel(logging.INFO)
        root_logger.addHandler(file_handler)
    except Exception as exc:
        sys.stderr.write(f"Warning: could not initialize desktop file logging: {exc}\n")


def _set_windows_app_id() -> None:
    """Register explicit Windows 10/11 AppUserModelID for taskbar & system tray identity."""
    if sys.platform == "win32":
        try:
            ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(
                "Antigravity.TradingIntelligencePlatform.Desktop.2"
            )
        except Exception:
            pass


def launch_desktop_app(verify_and_exit: bool = False) -> int:
    """Start the Windows Desktop Application with Startup Checklist -> Main Window."""
    setup_desktop_file_logging()
    logger = logging.getLogger("desktop.app")
    logger.info("Initializing Trading Intelligence Platform Desktop Application...")
    _set_windows_app_id()

    # Enforce Windows single-instance execution
    app_lock = ProcessFileLock(DESKTOP_LOCK_PATH)
    if not app_lock.acquire():
        if verify_and_exit:
            # In headless verification mode, test startup sequence directly without launching duplicate GUI
            controller = DesktopServiceController()
            res = controller.run_startup_sequence()
            controller.stop_background_services()
            print(f"[DesktopApp] Primary instance already running. Headless verification: {'PASSED' if res.all_critical_ok else 'FAILED'}")
            return 0 if res.all_critical_ok else 1
        else:
            print("[DesktopApp] Another instance of Trading Intelligence Desktop is already active. Secondary instance exiting safely.")
            return 0

    app = QApplication.instance() or QApplication(sys.argv)
    app.setApplicationName("Trading Intelligence Platform")
    app.setOrganizationName("Antigravity Trading Systems")
    app.setWindowIcon(create_app_icon())

    controller = DesktopServiceController()

    try:
        # Step 1-10: Show Startup Status Window (Section 15)
        splash = StartupStatusDialog(controller)
        splash.show()
        app.processEvents()
        startup_res = splash.run_checks()
        app.processEvents()

        # Create and show the main Windows Desktop Application Window
        main_win = TradingIntelligenceDesktopWindow(controller)
        QTimer.singleShot(450, splash.close)
        main_win.show()
        main_win.activateWindow()
        main_win.raise_()

        if verify_and_exit:
            # Allow background evaluation & scanner threads to populate the UI, then exit cleanly
            def _finish_verification() -> None:
                controller.stop_background_services()
                main_win._force_quit = True
                main_win.tray_icon.hide()
                main_win.close()
                app.quit()

            QTimer.singleShot(3500, _finish_verification)

        ret_code = app.exec()
        return ret_code
    finally:
        controller.stop_background_services()
        app_lock.release()


def main() -> None:
    parser = argparse.ArgumentParser(description="Trading Intelligence Platform — Windows Desktop Application")
    parser.add_argument(
        "--verify-and-exit",
        action="store_true",
        help="Launch desktop UI, run full startup + live evaluation verification, and exit cleanly.",
    )
    args, _ = parser.parse_known_args()
    sys.exit(launch_desktop_app(verify_and_exit=args.verify_and_exit))


if __name__ == "__main__":
    main()
