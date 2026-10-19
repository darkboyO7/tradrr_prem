"""Trading Intelligence Platform - Redis & In-Memory Async Cache/Message Bus.

Provides a unified caching and pub/sub abstraction. Connects to Redis when available,
and automatically falls back to a high-speed async in-memory store when Redis is offline.
"""

import asyncio
from collections.abc import AsyncGenerator
from typing import Any

import redis.asyncio as aioredis

from core.config import settings
from core.logging import get_logger

logger = get_logger(__name__)


class InMemoryCache:
    """Async thread-safe in-memory cache and message pub/sub fallback."""

    def __init__(self) -> None:
        self._store: dict[str, str] = {}
        self._channels: dict[str, list[asyncio.Queue[str]]] = {}
        self._lock = asyncio.Lock()

    async def get(self, key: str) -> str | None:
        async with self._lock:
            return self._store.get(key)

    async def set(self, key: str, value: str, ex: int | None = None) -> bool:
        async with self._lock:
            self._store[key] = value
            # In-memory expiry can be handled or kept simple for dev
            return True

    async def delete(self, key: str) -> bool:
        async with self._lock:
            return self._store.pop(key, None) is not None

    async def publish(self, channel: str, message: str) -> int:
        async with self._lock:
            subscribers = self._channels.get(channel, [])
            count = 0
            for q in subscribers:
                await q.put(message)
                count += 1
            return count

    async def subscribe(self, channel: str) -> AsyncGenerator[str, None]:
        q: asyncio.Queue[str] = asyncio.Queue()
        async with self._lock:
            if channel not in self._channels:
                self._channels[channel] = []
            self._channels[channel].append(q)

        try:
            while True:
                msg = await q.get()
                yield msg
        finally:
            async with self._lock:
                if channel in self._channels and q in self._channels[channel]:
                    self._channels[channel].remove(q)

    async def ping(self) -> bool:
        return True


class CacheManager:
    """Unified cache coordinator supporting Redis and In-Memory fallback."""

    def __init__(self) -> None:
        self._redis_client: aioredis.Redis | None = None
        self._in_memory = InMemoryCache()
        self._is_redis_active: bool = False

    async def initialize(self) -> None:
        """Attempt to connect to Redis; fall back to in-memory if unreachable."""
        try:
            client = aioredis.Redis(
                host=settings.REDIS_HOST,
                port=settings.REDIS_PORT,
                db=settings.REDIS_DB,
                password=settings.REDIS_PASSWORD or None,
                decode_responses=True,
                socket_connect_timeout=2.0,
            )
            await client.ping()
            self._redis_client = client
            self._is_redis_active = True
            logger.info("Connected to Redis instance successfully", host=settings.REDIS_HOST, port=settings.REDIS_PORT)
        except Exception as exc:
            self._is_redis_active = False
            self._redis_client = None
            logger.info(
                "Redis unavailable, operating with in-memory async cache engine",
                error=str(exc)
            )

    async def get(self, key: str) -> str | None:
        if self._is_redis_active and self._redis_client:
            try:
                return await self._redis_client.get(key)
            except Exception:
                return await self._in_memory.get(key)
        return await self._in_memory.get(key)

    async def set(self, key: str, value: str, ex: int | None = None) -> bool:
        if self._is_redis_active and self._redis_client:
            try:
                return bool(await self._redis_client.set(key, value, ex=ex))
            except Exception:
                return await self._in_memory.set(key, value, ex=ex)
        return await self._in_memory.set(key, value, ex=ex)

    async def delete(self, key: str) -> bool:
        if self._is_redis_active and self._redis_client:
            try:
                return bool(await self._redis_client.delete(key))
            except Exception:
                return await self._in_memory.delete(key)
        return await self._in_memory.delete(key)

    async def publish(self, channel: str, message: str) -> int:
        if self._is_redis_active and self._redis_client:
            try:
                return int(await self._redis_client.publish(channel, message))
            except Exception:
                return await self._in_memory.publish(channel, message)
        return await self._in_memory.publish(channel, message)

    async def check_health(self) -> dict[str, Any]:
        """Check cache status and engine."""
        if self._is_redis_active and self._redis_client:
            try:
                alive = await self._redis_client.ping()
                return {"status": "healthy" if alive else "unhealthy", "engine": "redis"}
            except Exception as exc:
                return {"status": "degraded", "engine": "in_memory_fallback", "redis_error": str(exc)}
        return {"status": "healthy", "engine": "in_memory"}

    async def close(self) -> None:
        if self._redis_client:
            await self._redis_client.aclose()
            self._redis_client = None
            self._is_redis_active = False


# Cache singleton
cache = CacheManager()
