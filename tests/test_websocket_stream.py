"""Tests for WebSocket Gateway Streaming and Connection Manager."""

import json
from unittest.mock import AsyncMock

from fastapi.testclient import TestClient
import pytest

from api.main import app
from api.streaming import WebSocketConnectionManager


@pytest.mark.asyncio
async def test_connection_manager_subscribe_and_broadcast():
    """Verify WebSocketConnectionManager dispatches messages only to relevant topics."""
    manager = WebSocketConnectionManager()

    ws_all = AsyncMock()
    ws_btc = AsyncMock()
    ws_eth = AsyncMock()

    await manager.connect(ws_all, default_topics={"all"})
    await manager.connect(ws_btc)
    await manager.connect(ws_eth)

    # ws_btc subscribes to trades:BTC-USD
    await manager.subscribe(ws_btc, "trades:BTC-USD")
    # ws_eth subscribes to trades:ETH-USD
    await manager.subscribe(ws_eth, "trades:ETH-USD")

    # Broadcast to trades:BTC-USD -> should reach ws_all (has "all") and ws_btc
    sent_count = await manager.broadcast("trades:BTC-USD", {"price": 68000.0})
    assert sent_count == 2
    assert ws_all.send_text.called
    assert ws_btc.send_text.called
    assert not ws_eth.send_text.called

    # Unsubscribe ws_btc from trades:BTC-USD
    await manager.unsubscribe(ws_btc, "trades:BTC-USD")
    ws_all.reset_mock()
    ws_btc.reset_mock()

    sent_count2 = await manager.broadcast("trades:BTC-USD", {"price": 68100.0})
    # Now only ws_all receives it
    assert sent_count2 == 1
    assert ws_all.send_text.called
    assert not ws_btc.send_text.called

    # Disconnect
    await manager.disconnect(ws_all)
    await manager.disconnect(ws_btc)
    await manager.disconnect(ws_eth)
    assert len(manager._connections) == 0


@pytest.mark.asyncio
async def test_connection_manager_dead_socket_pruning():
    """Verify broken sockets encountering exceptions during broadcast are automatically pruned."""
    manager = WebSocketConnectionManager()

    broken_ws = AsyncMock()
    broken_ws.send_text.side_effect = RuntimeError("Socket disconnected unexpectedly")

    await manager.connect(broken_ws, default_topics={"alerts"})
    assert len(manager._connections) == 1

    sent = await manager.broadcast("alerts", {"msg": "warning"})
    assert sent == 0
    # Broken socket must be removed from connections
    assert len(manager._connections) == 0


def test_fastapi_websocket_stream_lifecycle():
    """Verify client interaction with /ws/stream endpoint over TestClient."""
    client = TestClient(app)
    with client.websocket_connect("/ws/stream") as websocket:
        # 1. Ping / Pong
        websocket.send_text(json.dumps({"action": "ping"}))
        resp1 = json.loads(websocket.receive_text())
        assert resp1.get("status") == "pong"
        assert "time" in resp1

        # 2. Subscribe to custom topic
        websocket.send_text(json.dumps({"action": "subscribe", "topic": "signals:BTC-USD"}))
        resp2 = json.loads(websocket.receive_text())
        assert resp2.get("status") == "subscribed"
        assert resp2.get("topic") == "signals:BTC-USD"

        # 3. Unsubscribe from custom topic
        websocket.send_text(json.dumps({"action": "unsubscribe", "topic": "signals:BTC-USD"}))
        resp3 = json.loads(websocket.receive_text())
        assert resp3.get("status") == "unsubscribed"
        assert resp3.get("topic") == "signals:BTC-USD"

        # 4. Invalid JSON handling
        websocket.send_text("INVALID_NON_JSON")
        resp4 = json.loads(websocket.receive_text())
        assert "error" in resp4
