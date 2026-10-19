"""Trading Intelligence Platform — Windows Desktop Application Entrypoint.

Launch directly via:
    python -m desktop.app
or via the packaged Windows executable:
    TradingIntelligencePlatform.exe
"""

from __future__ import annotations

import argparse
import ctypes
import sys

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication

from desktop.main_window import StartupStatusDialog, TradingIntelligenceDesktopWindow, create_app_icon
from desktop.service_controller import DesktopServiceController


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
    _set_windows_app_id()

    app = QApplication.instance() or QApplication(sys.argv)
    app.setApplicationName("Trading Intelligence Platform")
    app.setOrganizationName("Antigravity Trading Systems")
    app.setWindowIcon(create_app_icon())

    controller = DesktopServiceController()

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
            main_win._force_quit = True
            main_win.tray_icon.hide()
            main_win.close()
            app.quit()

        QTimer.singleShot(3500, _finish_verification)

    return app.exec()


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
