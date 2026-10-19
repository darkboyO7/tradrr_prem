"""Trading Intelligence Platform — Standalone Windows Executable Entrypoint (TradingIntelligencePlatform.exe).

Allows double-clicking TradingIntelligencePlatform.exe from Windows Explorer, Desktop shortcut,
or Start Menu shortcut to start the complete Trading Intelligence Platform Desktop Application
with zero browser and zero manual Python configuration.
"""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys


def _find_project_and_python() -> tuple[Path, Path | None]:
    """Locate the application root and configured Python runtime."""
    candidates: list[Path] = []
    if getattr(sys, "frozen", False):
        exe_dir = Path(sys.executable).resolve().parent
        candidates.extend([exe_dir, exe_dir.parent, exe_dir.parent.parent, Path(r"d:\ant")])
        cfg_file = exe_dir / "tip_runtime_path.txt"
        if cfg_file.exists():
            try:
                saved = Path(cfg_file.read_text(encoding="utf-8").strip())
                candidates.insert(0, saved)
            except Exception:
                pass
    else:
        candidates.append(Path(__file__).resolve().parent.parent)

    for root in candidates:
        venv_pyw = root / ".venv" / "Scripts" / "pythonw.exe"
        venv_py = root / ".venv" / "Scripts" / "python.exe"
        if (root / "desktop" / "app.py").exists():
            if "--verify-and-exit" in sys.argv and venv_py.exists():
                return root, venv_py
            if venv_pyw.exists():
                return root, venv_pyw
            if venv_py.exists():
                return root, venv_py

    return Path(r"d:\ant"), None


def main() -> int:
    root, py_exe = _find_project_and_python()

    # If running directly inside the full environment (not frozen), invoke launch_desktop_app in-process
    if not getattr(sys, "frozen", False):
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))
        from desktop.app import launch_desktop_app

        verify = "--verify-and-exit" in sys.argv
        return launch_desktop_app(verify_and_exit=verify)

    # When running from compiled TradingIntelligencePlatform.exe, launch the desktop app seamlessly
    if py_exe and py_exe.exists():
        args = [str(py_exe), "-m", "desktop.app", *sys.argv[1:]]
        creation_flags = 0x08000000 if sys.platform == "win32" else 0  # CREATE_NO_WINDOW
        if "--verify-and-exit" in sys.argv:
            proc = subprocess.run(args, cwd=str(root), creationflags=creation_flags, check=False)
            return proc.returncode
        subprocess.Popen(args, cwd=str(root), creationflags=creation_flags)
        return 0

    # Fallback in-process if environment is self-contained
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    os.chdir(str(root))
    from desktop.app import launch_desktop_app

    return launch_desktop_app(verify_and_exit=("--verify-and-exit" in sys.argv))


if __name__ == "__main__":
    sys.exit(main())
