"""Trading Intelligence Platform - Base Execution Client.

Defines the broker/exchange-agnostic interface for trade execution.
Ensures that prediction and signal logic never directly interact with execution details.
"""

from abc import ABC, abstractmethod
from typing import Any


class BaseExecutionClient(ABC):
    """Abstract interface for external broker, exchange, or testnet connectivity."""

    @abstractmethod
    async def connect(self) -> None:
        """Establish authenticated session with execution venue."""
        pass

    @abstractmethod
    async def disconnect(self) -> None:
        """Terminate connection with execution venue."""
        pass

    @abstractmethod
    async def get_balance(self, asset: str = "USD") -> float:
        """Retrieve available cash or asset balance."""
        pass

    @abstractmethod
    async def get_positions(self) -> list[dict[str, Any]]:
        """Retrieve open positions reported by execution venue."""
        pass

    @abstractmethod
    async def place_order(
        self,
        symbol: str,
        side: str,  # BUY or SELL
        order_type: str,  # MARKET, LIMIT, STOP
        quantity: float,
        price: float | None = None,
        stop_loss: float | None = None,
        take_profit: float | None = None,
        client_order_id: str | None = None,
    ) -> dict[str, Any]:
        """Submit order to execution venue."""
        pass

    @abstractmethod
    async def cancel_order(self, order_id: str) -> bool:
        """Cancel an open order."""
        pass

    @abstractmethod
    async def get_order_status(self, order_id: str) -> dict[str, Any]:
        """Query current execution state of an order."""
        pass
