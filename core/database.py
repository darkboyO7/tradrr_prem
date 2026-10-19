"""Trading Intelligence Platform - Async Database Connection & Session Management.

Supports PostgreSQL/TimescaleDB connection pooling via asyncpg,
with seamless zero-configuration fallback to SQLite for local Windows development and tests.
"""

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Any

from sqlalchemy import text
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from core.config import settings
from core.logging import get_logger
from data.storage.models import Base

logger = get_logger(__name__)

# Engine and SessionMaker singletons
_engine: AsyncEngine | None = None
_session_maker: async_sessionmaker[AsyncSession] | None = None


def get_engine() -> AsyncEngine:
    """Retrieve or initialize the SQLAlchemy async database engine."""
    global _engine
    if _engine is None:
        db_url = settings.DATABASE_URL
        is_sqlite = db_url.startswith("sqlite")

        if is_sqlite:
            # SQLite does not support connection pooling arguments like pool_size
            _engine = create_async_engine(
                db_url,
                echo=False,
                future=True,
            )
        else:
            _engine = create_async_engine(
                db_url,
                echo=False,
                future=True,
                pool_size=settings.POSTGRES_POOL_SIZE,
                max_overflow=settings.POSTGRES_MAX_OVERFLOW,
                pool_pre_ping=True,
            )
    return _engine


def get_session_maker() -> async_sessionmaker[AsyncSession]:
    """Retrieve or initialize the SQLAlchemy async session maker."""
    global _session_maker
    if _session_maker is None:
        engine = get_engine()
        _session_maker = async_sessionmaker(
            bind=engine,
            class_=AsyncSession,
            expire_on_commit=False,
            autoflush=False,
        )
    return _session_maker


@asynccontextmanager
async def get_db_session() -> AsyncGenerator[AsyncSession, None]:
    """Provide a transactional asynchronous database session context."""
    session_factory = get_session_maker()
    async with session_factory() as session:
        try:
            yield session
            await session.commit()
        except Exception as exc:
            await session.rollback()
            logger.error("Database session transaction rolled back due to error", error=str(exc))
            raise


async def init_db() -> None:
    """Initialize database tables and TimescaleDB extensions if available."""
    engine = get_engine()
    async with engine.begin() as conn:
        # Check if TimescaleDB extension can be enabled (Postgres only)
        if not str(engine.url).startswith("sqlite"):
            try:
                await conn.execute(text("CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE;"))
                logger.info("TimescaleDB extension enabled or already present")
            except Exception as exc:
                logger.warning("Could not enable TimescaleDB extension (standard Postgres mode)", error=str(exc))

        # Create all tables
        await conn.run_sync(Base.metadata.create_all)
        logger.info("Database tables initialized successfully", tables=list(Base.metadata.tables.keys()))


async def check_db_health() -> dict[str, Any]:
    """Perform a ping check against the database."""
    engine = get_engine()
    try:
        async with engine.connect() as conn:
            result = await conn.execute(text("SELECT 1"))
            val = result.scalar()
            return {
                "status": "healthy" if val == 1 else "unhealthy",
                "dialect": engine.dialect.name,
                "url": str(engine.url).split("@")[-1],  # Sanitized host/db
            }
    except Exception as exc:
        return {
            "status": "unhealthy",
            "error": str(exc),
            "dialect": engine.dialect.name if engine else "unknown",
        }


async def close_db() -> None:
    """Dispose of the database engine and clean up connection pools."""
    global _engine, _session_maker
    if _engine is not None:
        await _engine.dispose()
        _engine = None
        _session_maker = None
        logger.info("Database connection pool disposed")
