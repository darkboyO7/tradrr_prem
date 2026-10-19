"""Trading Intelligence Platform - Real-Time WebSocket Streaming Gateway.

Manages active client WebSocket subscriptions for live market state,
fused signals, L2 order book metrics, and portfolio performance telemetry.
"""

import asyncio
from datetime import datetime, timezone
import json
from typing import Any, Dict, List, Optional, Set
from fastapi import WebSocket, WebSocketDisconnect
from core.logging import get_logger

logger = get_logger("ws_gateway")


class WebSocketConnectionManager:
    """Coordinates multi-topic WebSocket client subscriptions and broadcasting."""

    def __init__(self) -> None:
        # websocket -> set of subscribed topics (e.g. {"trades:BTC-USD", "signals:BTC-USD", "all"})
        self._connections: Dict[WebSocket, Set[str]] = {}
        self._lock = asyncio.Lock()

    async def connect(self, websocket: WebSocket, default_topics: Optional[Set[str]] = None) -> None:
        """Accept new WebSocket connection with optional initial topics (default: empty)."""
        await websocket.accept()
        async with self._lock:
            self._connections[websocket] = set(default_topics) if default_topics is not None else set()
        logger.info("Client connected to WebSocket Gateway", total_clients=len(self._connections))

    async def disconnect(self, websocket: WebSocket) -> None:
        """Remove WebSocket connection."""
        async with self._lock:
            self._connections.pop(websocket, None)
        logger.info("Client disconnected from WebSocket Gateway", remaining_clients=len(self._connections))

    async def subscribe(self, websocket: WebSocket, topic: str) -> None:
        """Add a specific topic subscription for a connected client."""
        async with self._lock:
            if websocket in self._connections:
                self._connections[websocket].add(topic)
                logger.debug("Client subscribed to topic", topic=topic)

    async def unsubscribe(self, websocket: WebSocket, topic: str) -> None:
        """Remove a topic subscription for a connected client."""
        async with self._lock:
            if websocket in self._connections:
                self._connections[websocket].discard(topic)
                logger.debug("Client unsubscribed from topic", topic=topic)

    async def broadcast(self, topic: str, data: Dict[str, Any]) -> int:
        """Broadcast payload to all clients subscribed to the topic or 'all'."""
        message_text = json.dumps({
            "topic": topic,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "data": data,
        })

        recipients: List[WebSocket] = []
        async with self._lock:
            for ws, topics in self._connections.items():
                if "all" in topics or topic in topics:
                    recipients.append(ws)

        sent_count = 0
        dead_sockets: List[WebSocket] = []

        for ws in recipients:
            try:
                await ws.send_text(message_text)
                sent_count += 1
            except Exception:
                dead_sockets.append(ws)

        if dead_sockets:
            async with self._lock:
                for ws in dead_sockets:
                    self._connections.pop(ws, None)

        return sent_count


# Global singleton instance
ws_manager = WebSocketConnectionManager()
