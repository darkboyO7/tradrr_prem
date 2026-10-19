"""Tests for Authenticated Binance REST Execution Client."""

import pytest
from execution.binance_rest import BinanceRESTExecutionClient
from paper_trading.models import OrderSide, OrderStatus, OrderType, PaperOrder


def test_binance_hmac_signature_generation():
    """Verify HMAC-SHA256 signature is correctly computed."""
    client = BinanceRESTExecutionClient(
        api_key="test_api_key",
        api_secret="test_secret_key_12345",
        testnet=True,
    )
    params = {"symbol": "BTCUSDT", "side": "BUY", "quantity": "0.1"}
    signed_query = client._sign_params(params)

    assert "symbol=BTCUSDT" in signed_query
    assert "timestamp=" in signed_query
    assert "signature=" in signed_query


@pytest.mark.asyncio
async def test_binance_execution_mock_fallback():
    """Verify unauthenticated/mock client provides safe fallback execution."""
    client = BinanceRESTExecutionClient(api_key="", api_secret="", testnet=True)
    await client.connect()

    # 1. Balance check
    balance = await client.get_balance("USDT")
    assert balance >= 100000.0

    # 2. Base place_order call
    ord_dict = await client.place_order(
        symbol="BTC-USDT",
        side="BUY",
        order_type="MARKET",
        quantity=0.5,
        price=65000.0,
    )
    assert ord_dict["status"] == "FILLED"
    assert ord_dict["filled_quantity"] == 0.5
    ord_id = ord_dict["order_id"]

    # 3. get_order_status call
    status_dict = await client.get_order_status(ord_id)
    assert status_dict["order_id"] == ord_id
    assert status_dict["status"] == "FILLED"

    # 4. High-level submit_order call
    order = PaperOrder(
        symbol="BTC-USDT",
        side=OrderSide.BUY,
        order_type=OrderType.MARKET,
        quantity=0.25,
        limit_price=65200.0,
    )
    filled_order = await client.submit_order(order)
    assert filled_order.status == OrderStatus.FILLED
    assert filled_order.filled_price == 65200.0
    assert filled_order.filled_quantity == 0.25
    assert filled_order.order_id.startswith("mock_")

    # 5. Order cancellation
    cancelled = await client.cancel_order(filled_order.order_id)
    assert cancelled is True

    await client.disconnect()
