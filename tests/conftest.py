"""Trading Intelligence Platform - Pytest Configuration & Test Fixtures."""

import asyncio
from collections.abc import AsyncGenerator
import os
from pathlib import Path
import shutil

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import NullPool

from data.storage.models import Base

TEST_DB_PATH = Path(__file__).parent / "test_scratch.db"
TEST_DB_URL = f"sqlite+aiosqlite:///{TEST_DB_PATH.as_posix()}"


@pytest.fixture(autouse=True)
def isolate_test_environment(monkeypatch, tmp_path):
    """Autouse fixture ensuring EVERY test executes in a hermetically isolated sandbox.

    Guarantees:
    1. Zero writes to production prediction outcomes or pending ledgers.
    2. Zero writes to production continuous learning state, audit logs, or models store.
    3. Zero writes to universe sync log, forex cache, or snapshots.
    4. Isolated SQLite database paths.
    5. Automatic teardown: background worker threads are stopped and singletons are cleanly reset.
    """
    import core.config as c_config
    import core.recovery.snapshot as c_snapshot
    import core.universe as c_universe
    import data.ingestion.forex as d_forex
    import learning.controller as l_controller
    import learning.ledger as l_ledger
    import models.registry as m_registry

    # 1. Build isolated sandbox directory structure
    iso_root = tmp_path / "test_sandbox"
    iso_root.mkdir(parents=True, exist_ok=True)

    iso_ledger_dir = iso_root / "ledger"
    iso_ledger_dir.mkdir(parents=True, exist_ok=True)
    iso_ledger_file = iso_ledger_dir / "prediction_outcomes.jsonl"
    iso_pending_file = iso_ledger_dir / "pending_predictions.jsonl"

    iso_learning_dir = iso_root / "learning"
    iso_learning_dir.mkdir(parents=True, exist_ok=True)
    iso_state_file = iso_learning_dir / "learning_controller_state.json"
    iso_audit_file = iso_learning_dir / "learning_audit_ledger.jsonl"

    iso_models_dir = iso_root / "models_store"
    iso_models_dir.mkdir(parents=True, exist_ok=True)

    iso_universe_dir = iso_root / "universe"
    iso_universe_dir.mkdir(parents=True, exist_ok=True)
    iso_sync_log = iso_universe_dir / "universe_sync_log.jsonl"
    iso_fx_cache = iso_universe_dir / "fx_candles_cache.json"

    iso_snapshots_dir = iso_root / "snapshots"
    iso_snapshots_dir.mkdir(parents=True, exist_ok=True)

    # 2. Copy read-only champion model if present so model evaluation tests work seamlessly
    prod_root = Path(__file__).resolve().parent.parent
    prod_champ = prod_root / "models_store" / "lgbm_champion_v1.0"
    if prod_champ.exists():
        try:
            shutil.copytree(prod_champ, iso_models_dir / "lgbm_champion_v1.0", dirs_exist_ok=True)
        except Exception:
            pass

    # 3. Monkeypatch module paths
    monkeypatch.setattr(l_ledger, "LEDGER_FILE_PATH", iso_ledger_file)
    monkeypatch.setattr(l_ledger, "PENDING_LEDGER_FILE_PATH", iso_pending_file)
    monkeypatch.setattr(l_controller, "DEFAULT_STATE_FILE", iso_state_file)
    monkeypatch.setattr(l_controller, "DEFAULT_AUDIT_LOG_FILE", iso_audit_file)
    monkeypatch.setattr(c_universe, "SYNC_LOG_PATH", iso_sync_log)
    monkeypatch.setattr(d_forex, "DISK_CACHE_PATH", iso_fx_cache)
    monkeypatch.setattr(c_config.settings, "MODEL_STORAGE_DIR", iso_models_dir)
    monkeypatch.setenv("MODEL_STORAGE_DIR", str(iso_models_dir))
    monkeypatch.setenv("SNAPSHOT_DIR", str(iso_snapshots_dir))

    # 4. Patch ModelRegistry and SnapshotManager defaults
    orig_registry_init = m_registry.ModelRegistry.__init__

    def patched_registry_init(self, base_dir=None, session_factory=None):
        target_dir = base_dir if base_dir is not None else iso_models_dir
        orig_registry_init(self, base_dir=target_dir, session_factory=session_factory)

    monkeypatch.setattr(m_registry.ModelRegistry, "__init__", patched_registry_init)

    orig_snapshot_init = c_snapshot.SnapshotManager.__init__

    def patched_snapshot_init(self, snapshot_dir=None):
        target_dir = snapshot_dir if snapshot_dir is not None else iso_snapshots_dir
        orig_snapshot_init(self, snapshot_dir=target_dir)

    monkeypatch.setattr(c_snapshot.SnapshotManager, "__init__", patched_snapshot_init)

    # 5. Clean reset before test execution
    l_ledger.PredictionLedgerService.reset_instance()
    l_controller.ContinuousLearningController.reset_instance()

    yield {
        "sandbox": iso_root,
        "ledger_file": iso_ledger_file,
        "pending_file": iso_pending_file,
        "state_file": iso_state_file,
        "audit_file": iso_audit_file,
        "models_dir": iso_models_dir,
        "universe_dir": iso_universe_dir,
        "snapshots_dir": iso_snapshots_dir,
    }

    # 6. Teardown: Stop any background workers cleanly and reset singletons
    if l_ledger.PredictionLedgerService._instance is not None:
        try:
            l_ledger.PredictionLedgerService._instance.stop_background_resolver()
        except Exception:
            pass
    if l_controller.ContinuousLearningController._instance is not None:
        try:
            l_controller.ContinuousLearningController._instance.stop_background_loop()
        except Exception:
            pass

    l_ledger.PredictionLedgerService.reset_instance()
    l_controller.ContinuousLearningController.reset_instance()


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for each test case."""
    policy = asyncio.get_event_loop_policy()
    loop = policy.new_event_loop()
    yield loop
    loop.close()


@pytest_asyncio.fixture(scope="function")
async def test_engine(tmp_path):
    """Create isolated SQLite database engine for test with per-test temporary scratch file."""
    db_path = tmp_path / "test_scratch.db"
    db_url = f"sqlite+aiosqlite:///{db_path.as_posix()}"
    engine = create_async_engine(
        db_url,
        echo=False,
        poolclass=NullPool,
    )
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield engine
    await engine.dispose()


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
