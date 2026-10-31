"""Trading Intelligence Platform — Cross-Platform Process and Mutex Locking Architecture.

Provides robust, cross-platform file-descriptor locking preventing duplicate process execution
and ensuring single-owner lifecycles for background services (e.g. Desktop application,
PredictionOutcomeResolver, and PaperTradingWorker).
"""

from __future__ import annotations

from datetime import UTC, datetime
import logging
import os
from pathlib import Path
import sys
from typing import Any

logger = logging.getLogger("core.system_lock")


class ProcessFileLock:
    """Cross-platform kernel/filesystem non-blocking lock.
    
    Guarantees:
    1. Single process ownership via OS kernel file descriptor locking.
    2. Automatic kernel lock release on process crash or termination.
    3. Safe PID and timestamp tracking in lockfile.
    4. Context manager interface (with ProcessFileLock(...):).
    """

    def __init__(self, lock_path: Path | str) -> None:
        self.lock_path = Path(lock_path)
        self.lock_path.parent.mkdir(parents=True, exist_ok=True)
        self._file: Any = None
        self._acquired = False

    def __enter__(self) -> bool:
        return self.acquire()

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.release()

    @property
    def is_acquired(self) -> bool:
        return self._acquired

    def acquire(self) -> bool:
        """Attempt to acquire non-blocking exclusive lock."""
        if self._acquired:
            return True
        try:
            self._file = open(self.lock_path, "a+b")
            self._file.seek(0)
            if sys.platform == "win32":
                import msvcrt
                # LK_NBLCK: Non-blocking 1-byte lock
                msvcrt.locking(self._file.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(self._file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)

            # Record PID and ISO timestamp
            self._file.seek(0)
            self._file.truncate(0)
            payload = f"pid={os.getpid()}\nstarted_at={datetime.now(UTC).isoformat()}\n".encode("utf-8")
            self._file.write(payload)
            self._file.flush()
            self._acquired = True
            return True
        except (OSError, PermissionError, IOError) as exc:
            logger.debug(f"Process lock at {self.lock_path} already held or unavailable: {exc}")
            if self._file:
                try:
                    self._file.close()
                except Exception:
                    pass
                self._file = None
            self._acquired = False
            return False

    def release(self) -> None:
        """Release the acquired lock cleanly."""
        if not self._acquired or not self._file:
            return
        try:
            self._file.seek(0)
            if sys.platform == "win32":
                import msvcrt
                try:
                    msvcrt.locking(self._file.fileno(), msvcrt.LK_UNLCK, 1)
                except Exception:
                    pass
            else:
                import fcntl
                try:
                    fcntl.flock(self._file.fileno(), fcntl.LOCK_UN)
                except Exception:
                    pass
            self._file.close()
        except Exception as e:
            logger.debug(f"Error releasing lock file {self.lock_path}: {e}")
        finally:
            self._file = None
            self._acquired = False
            try:
                if self.lock_path.exists():
                    self.lock_path.unlink()
            except Exception:
                pass

    @staticmethod
    def check_lock_status(lock_path: Path | str) -> dict[str, Any]:
        """Inspects lock file and tests whether an active process holds the kernel lock."""
        lock_path = Path(lock_path)
        if not lock_path.exists():
            return {"locked": False, "pid": None, "started_at": None, "status": "NO_LOCK_FILE"}

        pid = None
        started_at = None
        try:
            content = lock_path.read_text(encoding="utf-8")
            for line in content.splitlines():
                if line.startswith("pid="):
                    pid = int(line.split("=", 1)[1].strip())
                elif line.startswith("started_at="):
                    started_at = line.split("=", 1)[1].strip()
        except Exception:
            pass

        is_held = False
        try:
            f = open(lock_path, "r+b")
            try:
                f.seek(0)
                if sys.platform == "win32":
                    import msvcrt
                    try:
                        msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)
                        # Succeeded in acquiring, so NOT held by any other process!
                        f.seek(0)
                        msvcrt.locking(f.fileno(), msvcrt.LK_UNLCK, 1)
                        is_held = False
                    except (OSError, PermissionError, IOError):
                        is_held = True
                else:
                    import fcntl
                    try:
                        fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                        fcntl.flock(f.fileno(), fcntl.LOCK_UN)
                        is_held = False
                    except (OSError, IOError):
                        is_held = True
            finally:
                f.close()
        except (OSError, PermissionError, IOError):
            is_held = True

        return {
            "locked": is_held,
            "pid": pid if is_held else None,
            "started_at": started_at if is_held else None,
            "status": "LOCKED" if is_held else "UNLOCKED_OR_STALE",
        }
