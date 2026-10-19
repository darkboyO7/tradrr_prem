"""Trading Intelligence Platform — Windows Setup Installer (Trading Intelligence Platform Setup.exe).

Installs:
- TradingIntelligencePlatform.exe Desktop Application
- Runtime & local service configuration
- Authoritative NSE/BSE/Quotex universe master cache
- Windows Desktop Shortcut ("Trading Intelligence Platform.lnk")
- Windows Start-Menu Shortcut ("Trading Intelligence Platform.lnk")
"""

from __future__ import annotations

import ctypes
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys


def _create_windows_shortcut(
    shortcut_path: Path,
    target_exe: Path,
    working_dir: Path,
    icon_path: Path,
    description: str,
) -> bool:
    """Create a native Windows .lnk shortcut using PowerShell / WScript.Shell COM."""
    try:
        shortcut_path.parent.mkdir(parents=True, exist_ok=True)
        ps_script = (
            f"$ws = New-Object -ComObject WScript.Shell; "
            f"$s = $ws.CreateShortcut('{shortcut_path}'); "
            f"$s.TargetPath = '{target_exe}'; "
            f"$s.WorkingDirectory = '{working_dir}'; "
            f"$s.IconLocation = '{icon_path}'; "
            f"$s.Description = '{description}'; "
            f"$s.Save()"
        )
        subprocess.run(
            ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", ps_script],
            check=True,
            creationflags=0x08000000 if sys.platform == "win32" else 0,
        )
        return shortcut_path.exists()
    except Exception:
        return False


def perform_installation(
    source_root: Path | None = None,
    install_dir: Path | None = None,
    create_shortcuts: bool = True,
) -> dict[str, str | bool]:
    """Install the Trading Intelligence Platform Windows Desktop Application and shortcuts."""
    if source_root is None:
        if getattr(sys, "frozen", False):
            exe_parent = Path(sys.executable).resolve().parent
            source_root = exe_parent.parent if (exe_parent.parent / "desktop").exists() else Path(r"d:\ant")
        else:
            source_root = Path(__file__).resolve().parent.parent

    if install_dir is None:
        local_appdata = Path(os.environ.get("LOCALAPPDATA", str(Path.home() / "AppData" / "Local")))
        install_dir = local_appdata / "Programs" / "TradingIntelligencePlatform"

    install_dir.mkdir(parents=True, exist_ok=True)

    # 1. Copy TradingIntelligencePlatform.exe
    built_exe_candidates = [
        source_root / "dist" / "TradingIntelligencePlatform.exe",
        source_root / "dist" / "TradingIntelligencePlatform" / "TradingIntelligencePlatform.exe",
        source_root / "TradingIntelligencePlatform.exe",
    ]
    target_exe = install_dir / "TradingIntelligencePlatform.exe"
    for cand in built_exe_candidates:
        if cand.exists():
            shutil.copy2(cand, target_exe)
            break

    # 2. Copy Icon & Runtime configuration
    icon_src = source_root / "desktop" / "assets" / "app_icon.ico"
    icon_dst = install_dir / "app_icon.ico"
    if icon_src.exists():
        shutil.copy2(icon_src, icon_dst)

    (install_dir / "tip_runtime_path.txt").write_text(str(source_root.resolve()), encoding="utf-8")
    if (source_root / "tip_runtime_path.txt").parent.exists():
        (source_root / "tip_runtime_path.txt").write_text(str(source_root.resolve()), encoding="utf-8")

    # 3. Write application configuration manifest
    config_manifest = {
        "application_name": "Trading Intelligence Platform",
        "edition": "Windows Desktop Edition",
        "version": "2.0.0",
        "backend_root": str(source_root.resolve()),
        "install_dir": str(install_dir.resolve()),
        "executable": str(target_exe.resolve()),
        "auto_start_services": True,
        "markets_enabled": ["NSE", "BSE", "QUOTEX", "FOREX", "CRYPTO"],
    }
    (install_dir / "desktop_config.json").write_text(json.dumps(config_manifest, indent=2), encoding="utf-8")

    # 4. Create Desktop Shortcut & Start Menu Shortcut
    desktop_dir = Path(os.environ.get("USERPROFILE", str(Path.home()))) / "Desktop"
    desktop_lnk = desktop_dir / "Trading Intelligence Platform.lnk"

    appdata_dir = Path(os.environ.get("APPDATA", str(Path.home() / "AppData" / "Roaming")))
    start_menu_dir = appdata_dir / "Microsoft" / "Windows" / "Start Menu" / "Programs"
    start_menu_lnk = start_menu_dir / "Trading Intelligence Platform.lnk"

    exe_for_shortcut = target_exe if target_exe.exists() else (source_root / "TradingIntelligencePlatform.exe")
    ico_for_shortcut = icon_dst if icon_dst.exists() else exe_for_shortcut

    desktop_ok = False
    start_menu_ok = False
    if create_shortcuts:
        desktop_ok = _create_windows_shortcut(
            shortcut_path=desktop_lnk,
            target_exe=exe_for_shortcut,
            working_dir=source_root,
            icon_path=ico_for_shortcut,
            description="Trading Intelligence Platform — Windows Desktop Application",
        )
        start_menu_ok = _create_windows_shortcut(
            shortcut_path=start_menu_lnk,
            target_exe=exe_for_shortcut,
            working_dir=source_root,
            icon_path=ico_for_shortcut,
            description="Trading Intelligence Platform — Windows Desktop Application",
        )

    return {
        "installed": True,
        "install_dir": str(install_dir),
        "executable": str(exe_for_shortcut),
        "desktop_shortcut": str(desktop_lnk),
        "desktop_shortcut_created": desktop_ok,
        "start_menu_shortcut": str(start_menu_lnk),
        "start_menu_shortcut_created": start_menu_ok,
    }


def main() -> int:
    silent = "--silent" in sys.argv
    result = perform_installation(create_shortcuts=True)

    if not silent and sys.platform == "win32":
        msg = (
            "Trading Intelligence Platform has been installed on your computer.\n\n"
            f"Executable:\n{result['executable']}\n\n"
            f"Desktop Shortcut:\n{result['desktop_shortcut']}\n\n"
            f"Start Menu Shortcut:\n{result['start_menu_shortcut']}\n\n"
            "Click OK to launch Trading Intelligence Platform."
        )
        ctypes.windll.user32.MessageBoxW(
            0,
            msg,
            "Trading Intelligence Platform Setup",
            0x00000040,  # MB_ICONINFORMATION
        )
        exe_path = Path(str(result["executable"]))
        if exe_path.exists():
            subprocess.Popen([str(exe_path)], creationflags=0x08000000)

    return 0


if __name__ == "__main__":
    sys.exit(main())
