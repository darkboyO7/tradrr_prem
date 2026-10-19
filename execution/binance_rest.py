"""Trading Intelligence Platform - Authenticated Binance REST Execution Client.

Provides signed order routing, cancellations, and account balance reconciliation
supporting both Binance Testnet and Production Spot markets.
"""

from datetime import datetime, timezone
import hashlib
import hmac
import time
from typing import Any, Dict, List, Optional
import urllib.parse
import httpx

from core.logging import get_logger
from execution.base import BaseExecutionClient
from paper_trading.models import OrderSide, OrderStatus, OrderType, PaperOrder

logger = get_logger("binance_exec")


class BinanceRESTExecutionClient(BaseExecutionClient):
    """Production-grade signed REST execution client for Binance Spot exchange."""

    PROD_URL = "https://api.binance.com"
    TESTNET_URL = "https://testnet.binance.vision"

    def __init__(
        self,
        api_key: str = "",
        api_secret: str = "",
        testnet: bool = True,
        timeout_seconds: float = 5.0,
    ) -> None:
        self.api_key = api_key
        self.api_secret = api_secret
        self.testnet = testnet
        self.base_url = self.TESTNET_URL if testnet else self.PROD_URL
        self.timeout = timeout_seconds
        self._client: Optional[httpx.AsyncClient] = None
        self._is_connected = False
        self._orders: Dict[str, Dict[str, Any]] = {}

    def _sign_params(self, params: Dict[str, Any]) -> str:
        """Create HMAC-SHA256 signature for Binance API security."""
        params["timestamp"] = int(time.time() * 1000)
        query_string = urllib.parse.urlencode(params)
        signature = hmac.new(
            self.api_secret.encode("utf-8"),
            query_string.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()
        return f"{query_string}&signature={signature}"

    def _get_headers(self) -> Dict[str, str]:
        """Generate API key headers."""
        headers = {"Accept": "application/json"}
        if self.api_key:
            headers["X-MBX-APIKEY"] = self.api_key
        return headers

    async def connect(self) -> None:
        """Establish HTTP connection and ping Binance endpoint."""
        self._client = httpx.AsyncClient(base_url=self.base_url, timeout=self.timeout)
        try:
            resp = await self._client.get("/api/v3/ping")
            resp.raise_for_status()
            self._is_connected = True
            logger.info("Connected to Binance REST API", testnet=self.testnet, base_url=self.base_url)
        except Exception as exc:
            self._is_connected = False
            logger.warning("Binance REST connectivity check failed (will operate in mock mode if unauthenticated)", error=str(exc))

    async def disconnect(self) -> None:
        """Close asynchronous HTTP connection."""
        if self._client:
            await self._client.aclose()
            self._client = None
        self._is_connected = False
        logger.info("Disconnected from Binance REST API")

    async def get_balance(self, asset: str = "USDT") -> float:
        """Query free account balance for the specified asset."""
        if not self._client or not self.api_key:
            # Fallback for unconfigured/test environments
            return 100000.0

        signed_query = self._sign_params({})
        resp = await self._client.get(f"/api/v3/account?{signed_query}", headers=self._get_headers())
        resp.raise_for_status()
        data = resp.json()

        for b in data.get("balances", []):
            if b.get("asset") == asset.upper():
                return float(b.get("free", 0.0))
        return 0.0

    async def get_positions(self) -> List[Dict[str, Any]]:
        """Retrieve active non-zero asset holdings as spot portfolio positions."""
        if not self._client or not self.api_key:
            return []

        signed_query = self._sign_params({})
        resp = await self._client.get(f"/api/v3/account?{signed_query}", headers=self._get_headers())
        resp.raise_for_status()
        data = resp.json()

        positions = []
        for b in data.get("balances", []):
            qty = float(b.get("free", 0.0)) + float(b.get("locked", 0.0))
            asset = b.get("asset", "")
            if qty > 0.0001 and asset not in ("USDT", "USD", "BUSD", "USDC"):
                positions.append({
                    "symbol": f"{asset}-USDT",
                    "side": "BUY",
                    "quantity": qty,
                    "entry_price": 0.0,
                    "current_price": 0.0,
                    "unrealized_pnl": 0.0,
                })
        return positions

    async def place_order(
        self,
        symbol: str,
        side: str,
        order_type: str,
        quantity: float,
        price: Optional[float] = None,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
        client_order_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Submit an order implementing BaseExecutionClient protocol."""
        clean_symbol = symbol.replace("-", "").replace("/", "").upper()
        side_str = side.upper()
        type_str = order_type.upper()

        if not self._client or not self.api_key:
            ord_id = client_order_id or f"mock_{int(time.time()*1000)}"
            exec_price = price or 65000.0
            order_dict = {
                "order_id": ord_id,
                "symbol": symbol,
                "side": side_str,
                "status": "FILLED",
                "filled_price": exec_price,
                "filled_quantity": quantity,
                "fee": round(quantity * exec_price * 0.001, 4),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
            self._orders[ord_id] = order_dict
            return order_dict

        params: Dict[str, Any] = {
            "symbol": clean_symbol,
            "side": side_str,
            "type": type_str,
            "quantity": f"{quantity:.6f}",
        }
        if type_str == "LIMIT" and price:
            params["price"] = f"{price:.2f}"
            params["timeInForce"] = "GTC"
        if client_order_id:
            params["newClientOrderId"] = client_order_id

        signed_query = self._sign_params(params)
        resp = await self._client.post(f"/api/v3/order?{signed_query}", headers=self._get_headers())
        resp.raise_for_status()
        res_data = resp.json()

        ord_id = str(res_data.get("orderId"))
        executed_qty = float(res_data.get("executedQty", 0.0))
        quote_qty = float(res_data.get("cummulativeQuoteQty", 0.0))
        avg_price = quote_qty / max(0.000001, executed_qty) if executed_qty > 0 else (price or 0.0)

        result = {
            "order_id": ord_id,
            "symbol": symbol,
            "side": side_str,
            "status": res_data.get("status", "OPEN"),
            "filled_price": avg_price,
            "filled_quantity": executed_qty,
            "fee": 0.0,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        self._orders[ord_id] = result
        return result

    async def cancel_order(self, order_id: str) -> bool:
        """Cancel an open order on Binance."""
        if not self._client or not self.api_key:
            if order_id in self._orders:
                self._orders[order_id]["status"] = "CANCELED"
            return True

        params = {"orderId": order_id}
        signed_query = self._sign_params(params)
        resp = await self._client.delete(f"/api/v3/order?{signed_query}", headers=self._get_headers())
        return resp.status_code == 200

    async def get_order_status(self, order_id: str) -> Dict[str, Any]:
        """Query current execution state of an order."""
        if order_id in self._orders:
            return self._orders[order_id]

        if not self._client or not self.api_key:
            return {"order_id": order_id, "status": "UNKNOWN"}

        params = {"orderId": order_id}
        signed_query = self._sign_params(params)
        resp = await self._client.get(f"/api/v3/order?{signed_query}", headers=self._get_headers())
        resp.raise_for_status()
        return resp.json()

    async def submit_order(self, order: PaperOrder) -> PaperOrder:
        """High-level PaperOrder bridge."""
        side_str = "BUY" if order.side == OrderSide.BUY else "SELL"
        type_str = "MARKET" if order.order_type == OrderType.MARKET else "LIMIT"

        res = await self.place_order(
            symbol=order.symbol,
            side=side_str,
            order_type=type_str,
            quantity=order.quantity,
            price=getattr(order, "limit_price", getattr(order, "requested_price", None)),
            stop_loss=order.stop_loss,
            take_profit=order.take_profit,
        )

        order.order_id = res["order_id"]
        order.status = OrderStatus.FILLED if res["status"] == "FILLED" else OrderStatus.OPEN
        order.filled_price = res["filled_price"]
        order.filled_quantity = res["filled_quantity"]
        order.fee = res.get("fee", 0.0)
        order.filled_at = datetime.now(timezone.utc)
        return order
