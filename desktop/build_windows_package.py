"""Build script to generate Windows .ico asset, TradingIntelligencePlatform.exe, and Trading Intelligence Platform Setup.exe."""

from __future__ import annotations

from pathlib import Path
import shutil
import subprocess
import sys

from PIL import Image, ImageDraw


def generate_windows_ico(target_ico: Path) -> None:
    """Render a multi-size Windows .ico icon for the desktop executable and installer."""
    target_ico.parent.mkdir(parents=True, exist_ok=True)
    img = Image.new("RGBA", (256, 256), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    # Dark rounded background with emerald border
    draw.rounded_rectangle((12, 12, 244, 244), radius=48, fill=(15, 23, 42, 255), outline=(16, 185, 129, 255), width=6)

    # Three ascending candlestick bars
    draw.rounded_rectangle((56, 136, 88, 204), radius=8, fill=(56, 189, 248, 255))
    draw.line((72, 114, 72, 218), fill=(56, 189, 248, 255), width=6)

    draw.rounded_rectangle((112, 96, 144, 204), radius=8, fill=(16, 185, 129, 255))
    draw.line((128, 74, 128, 218), fill=(16, 185, 129, 255), width=6)

    draw.rounded_rectangle((168, 56, 200, 204), radius=8, fill=(52, 211, 153, 255))
    draw.line((184, 38, 184, 218), fill=(52, 211, 153, 255), width=6)

    img.save(
        target_ico,
        format="ICO",
        sizes=[(16, 16), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)],
    )


def build_executables() -> None:
    root = Path(__file__).resolve().parent.parent
    ico_path = root / "desktop" / "assets" / "app_icon.ico"
    generate_windows_ico(ico_path)
    print(f"[1/4] Generated Windows Icon: {ico_path}")

    pyinstaller_exe = root / ".venv" / "Scripts" / "pyinstaller.exe"
    dist_dir = root / "dist"
    build_dir = root / "build"

    # 1. Build TradingIntelligencePlatform.exe
    print("[2/4] Building TradingIntelligencePlatform.exe...")
    subprocess.run(
        [
            str(pyinstaller_exe),
            "--noconfirm",
            "--clean",
            "--onefile",
            "--windowed",
            "--name",
            "TradingIntelligencePlatform",
            "--icon",
            str(ico_path),
            "--distpath",
            str(dist_dir),
            "--workpath",
            str(build_dir / "app"),
            "--specpath",
            str(build_dir),
            str(root / "desktop" / "launcher_entry.py"),
        ],
        cwd=str(root),
        check=True,
    )

    app_exe = dist_dir / "TradingIntelligencePlatform.exe"
    if app_exe.exists():
        shutil.copy2(app_exe, root / "TradingIntelligencePlatform.exe")
        print(f"      -> Built: {app_exe} and {root / 'TradingIntelligencePlatform.exe'}")

    # 2. Build Trading Intelligence Platform Setup.exe
    print("[3/4] Building Trading Intelligence Platform Setup.exe...")
    subprocess.run(
        [
            str(pyinstaller_exe),
            "--noconfirm",
            "--clean",
            "--onefile",
            "--windowed",
            "--name",
            "Trading Intelligence Platform Setup",
            "--icon",
            str(ico_path),
            "--distpath",
            str(dist_dir),
            "--workpath",
            str(build_dir / "setup"),
            "--specpath",
            str(build_dir),
            str(root / "desktop" / "setup_installer.py"),
        ],
        cwd=str(root),
        check=True,
    )

    setup_exe = dist_dir / "Trading Intelligence Platform Setup.exe"
    if setup_exe.exists():
        shutil.copy2(setup_exe, root / "Trading Intelligence Platform Setup.exe")
        print(f"      -> Built: {setup_exe} and {root / 'Trading Intelligence Platform Setup.exe'}")

    # 3. Execute silent installation to register Desktop & Start-Menu shortcuts
    print("[4/4] Installing Desktop Application and creating Windows Shortcuts...")
    subprocess.run([str(setup_exe), "--silent"], cwd=str(root), check=True)
    print("      -> Installation complete!")


if __name__ == "__main__":
    build_executables()
