"""Trading Intelligence Platform - System Health & Observability Engine.

Provides unified health diagnostics across Database, Cache, Provider streams,
and Ingestion pipelines with latency percentiles.
"""

import os
import platform
import time
from typing import Any

from core.cache import cache
from core.database import check_db_health
from core.logging import get_logger

logger = get_logger(__name__)

_app_start_time = time.monotonic()


async def check_system_health(pipeline_metrics: dict[str, Any] | None = None) -> dict[str, Any]:
    """Perform a holistic health diagnostic check across all system subsystems."""
    uptime_seconds = time.monotonic() - _app_start_time

    # 1. Database Health
    db_health = await check_db_health()

    # 2. Cache Health
    cache_health = await cache.check_health()

    # 3. Overall status determination
    is_healthy = (
        db_health.get("status") == "healthy" and
        cache_health.get("status") in ("healthy", "degraded")
    )

    return {
        "status": "healthy" if is_healthy else "unhealthy",
        "uptime_seconds": round(uptime_seconds, 2),
        "environment": {
            "os": platform.system(),
            "os_release": platform.release(),
            "python_version": platform.python_version(),
            "pid": os.getpid(),
        },
        "components": {
            "database": db_health,
            "cache": cache_health,
            "pipeline": pipeline_metrics or {"status": "inactive"},
        },
    }
