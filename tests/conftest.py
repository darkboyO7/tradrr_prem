"""Trading Intelligence Platform - Pytest Configuration & Test Fixtures."""

import asyncio
from collections.abc import AsyncGenerator
from pathlib import Path

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from data.storage.models import Base

TEST_DB_PATH = Path(__file__).parent / "test_scratch.db"
TEST_DB_URL = f"sqlite+aiosqlite:///{TEST_DB_PATH.as_posix()}"


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for each test case."""
    policy = asyncio.get_event_loop_policy()
    loop = policy.new_event_loop()
    yield loop
    loop.close()


from sqlalchemy.pool import NullPool


@pytest_asyncio.fixture(scope="function")
async def test_engine():
    """Create isolated SQLite database engine for test with shared on-disk scratch file."""
    if TEST_DB_PATH.exists():
        try:
            TEST_DB_PATH.unlink()
        except Exception:
            pass

    engine = create_async_engine(
        TEST_DB_URL,
        echo=False,
        poolclass=NullPool,
    )
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield engine
    await engine.dispose()
    if TEST_DB_PATH.exists():
        try:
            TEST_DB_PATH.unlink()
        except Exception:
            pass


@pytest_asyncio.fixture(scope="function")
async def db_session(test_engine) -> AsyncGenerator[AsyncSession, None]:
    """Provide an isolated async session for database operations."""
    session_factory = async_sessionmaker(
        bind=test_engine,
        class_=AsyncSession,
        expire_on_commit=False,
    )
    async with session_factory() as session:
        yield session
        await session.rollback()


@pytest.fixture
def sample_coinbase_match_payload():
    """Sample raw match JSON payload from Coinbase Exchange."""
    return {
        "type": "match",
        "trade_id": 987654321,
        "maker_order_id": "d0e12345-6789-abcd-ef01-234567890abc",
        "taker_order_id": "e1f23456-789a-bcde-f012-3456789abcde",
        "side": "buy",
        "size": "0.15000000",
        "price": "65432.10",
        "product_id": "BTC-USD",
        "sequence": 100200300,
        "time": "2026-10-04T08:00:00.123456Z",
    }


@pytest.fixture
def sample_coinbase_ticker_payload():
    """Sample raw ticker JSON payload from Coinbase Exchange."""
    return {
        "type": "ticker",
        "sequence": 100200301,
        "product_id": "ETH-USD",
        "price": "3450.50",
        "open_24h": "3400.00",
        "volume_24h": "50000.0",
        "low_24h": "3350.00",
        "high_24h": "3500.00",
        "best_bid": "3450.40",
        "best_bid_size": "5.5",
        "best_ask": "3450.60",
        "best_ask_size": "8.2",
        "side": "sell",
        "time": "2026-10-04T08:00:01.000000Z",
        "trade_id": 87654321,
        "last_size": "1.2",
    }
