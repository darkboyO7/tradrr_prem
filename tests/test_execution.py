"""Tests for Execution Clients and Position Reconciliation."""

import pytest

from execution.reconciliation import PositionReconciler
from execution.simulated import SimulatedExecutionClient
from paper_trading.engine import PaperTradingEngine
from paper_trading.models import OrderSide, PaperPosition


@pytest.mark.asyncio
async def test_simulated_execution_client():
    engine = PaperTradingEngine(initial_cash=100000.0)
    client = SimulatedExecutionClient(paper_engine=engine, simulated_latency_ms=1.0)

    # 1. Connect
    await client.connect()
    assert client.is_connected

    # 2. Balance
    balance = await client.get_balance("USD")
    assert balance == 100000.0

    # 3. Set reference price and place market buy
    client.set_market_price("BTC-USD", 60000.0)
    order_res = await client.place_order(
        symbol="BTC-USD",
        side="BUY",
        order_type="MARKET",
        quantity=1.0,
        stop_loss=58000.0,
        take_profit=64000.0,
    )

    assert order_res["status"] == "FILLED"
    assert order_res["filled_quantity"] == 1.0
    assert order_res["filled_price"] > 60000.0  # Slippage + spread

    # 4. Check positions
    positions = await client.get_positions()
    assert len(positions) == 1
    assert positions[0]["symbol"] == "BTC-USD"

    # 5. Order status check
    status = await client.get_order_status(order_res["order_id"])
    assert status["status"] == "FILLED"

    # 6. Disconnect
    await client.disconnect()
    assert not client.is_connected


def test_position_reconciler_in_sync():
    local_pos = {
        "BTC-USD": PaperPosition(
            position_id="p1",
            symbol="BTC-USD",
            side=OrderSide.BUY,
            quantity=1.5,
            entry_price=60000.0,
            current_price=61000.0,
            stop_loss=58000.0,
            take_profit=64000.0,
        )
    }

    remote_pos = [
        {"symbol": "BTC-USD", "side": "BUY", "quantity": 1.5, "price": 60000.0}
    ]

    report = PositionReconciler.reconcile(local_pos, remote_pos)
    assert report.is_synchronized
    assert len(report.discrepancies) == 0


def test_position_reconciler_discrepancies():
    # Local has 2.0 BTC Buy, remote has 1.0 BTC Buy + Ghost ETH position
    local_pos = {
        "BTC-USD": PaperPosition(
            position_id="p1",
            symbol="BTC-USD",
            side=OrderSide.BUY,
            quantity=2.0,
            entry_price=60000.0,
            current_price=61000.0,
            stop_loss=58000.0,
            take_profit=64000.0,
        )
    }

    remote_pos = [
        {"symbol": "BTC-USD", "side": "BUY", "quantity": 1.0, "price": 60000.0},
        {"symbol": "ETH-USD", "side": "BUY", "quantity": 5.0, "price": 3000.0},
    ]

    report = PositionReconciler.reconcile(local_pos, remote_pos)
    assert not report.is_synchronized
    assert len(report.discrepancies) == 2

    types = [d.issue_type for d in report.discrepancies]
    assert "QUANTITY_MISMATCH" in types
    assert "MISSING_LOCAL" in types
