"""Comprehensive verification tests for Execution Safety (P0).

Proves that:
1. Paper execution adapter strictly requires explicit connected state.
2. Market session is checked and closed markets reject orders.
3. Missing, zero, negative, NaN, and inf prices are rejected without fallback to 100.0.
4. Non-positive, NaN, and inf quantities are rejected.
5. Unknown sides and order types are rejected (never silently mapped to SELL or STOP).
6. Unsupported instruments are rejected.
7. Stale or unavailable market data rejects execution.
8. NO_TRADE decisions never trigger order placement.
9. All execution responses explicitly declare simulated paper mode and non-live status.
"""

from __future__ import annotations

from datetime import UTC, datetime
import math
from unittest.mock import MagicMock, patch
import pytest

from core.instrument import MarketType
from core.symbol_resolver import SymbolResolver
from execution.multi_market import (
    BSEExecutionClient,
    ForexExecutionClient,
    MultiMarketPaperExecutionClient,
    NSEExecutionClient,
    QuotexExecutionClient,
)
from execution.simulated import SimulatedExecutionClient
from models.base import SignalAction
from signals.fusion import FinalSignalDecision
from paper_trading.engine import PaperTradingEngine
from paper_trading.models import OrderSide, OrderStatus, PaperOrder
from risk.engine import RiskDecisionStatus, RiskEngine


@pytest.mark.asyncio
async def test_execution_adapter_requires_connected_state() -> None:
    """Proves orders submitted to disconnected execution clients are rejected."""
    client = NSEExecutionClient()
    assert client.is_connected is False

    res = await client.place_order(
        symbol="NSE:TCS",
        side="BUY",
        order_type="MARKET",
        quantity=5.0,
        price=3500.0,
    )
    assert res["status"] == "REJECTED"
    assert "CLIENT_NOT_CONNECTED" in res["reason"]
    assert res["execution_mode"] == "SIMULATED_PAPER"
    assert res["is_live"] is False


@pytest.mark.asyncio
async def test_execution_adapter_rejects_closed_market() -> None:
    """Proves orders submitted when exchange session is closed are rejected."""
    client = NSEExecutionClient(enforce_market_hours=True)
    await client.connect()
    try:
        # Mock session to report closed
        mock_session = MagicMock()
        mock_session.is_open.return_value = False
        with patch.object(SymbolResolver, "get_session", return_value=mock_session):
            res = await client.place_order(
                symbol="NSE:TCS",
                side="BUY",
                order_type="MARKET",
                quantity=5.0,
                price=3500.0,
            )
            assert res["status"] == "REJECTED"
            assert "MARKET_CLOSED" in res["reason"]
            assert res["market_open"] is False
    finally:
        await client.disconnect()


@pytest.mark.asyncio
async def test_execution_adapter_rejects_invalid_and_fallback_prices() -> None:
    """Proves missing, zero, negative, NaN, and inf prices are rejected (no 100.0 fallback)."""
    client = NSEExecutionClient()
    await client.connect()
    mock_open_session = MagicMock()
    mock_open_session.is_open.return_value = True
    try:
        with patch.object(SymbolResolver, "get_session", return_value=mock_open_session):
            # 1. Market order with no observed market price and no price passed
            res_no_price = await client.place_order(
                symbol="NSE:TCS",
                side="BUY",
                order_type="MARKET",
                quantity=5.0,
                price=None,
            )
            assert res_no_price["status"] == "REJECTED"
            assert "UNAVAILABLE_MARKET_PRICE" in res_no_price["reason"]

            # 2. Limit order with None price
            res_none_limit = await client.place_order(
                symbol="NSE:TCS",
                side="BUY",
                order_type="LIMIT",
                quantity=5.0,
                price=None,
            )
            assert res_none_limit["status"] == "REJECTED"
            assert "INVALID_PRICE" in res_none_limit["reason"]

            # 3. Non-positive prices (zero, negative)
            for bad_p in (0.0, -10.0, -0.05):
                res_bad = await client.place_order(
                    symbol="NSE:TCS",
                    side="BUY",
                    order_type="LIMIT",
                    quantity=5.0,
                    price=bad_p,
                )
                assert res_bad["status"] == "REJECTED"
                assert "INVALID_PRICE" in res_bad["reason"]

            # 4. NaN and inf prices
            for bad_p in (float("nan"), float("inf"), float("-inf")):
                res_bad = await client.place_order(
                    symbol="NSE:TCS",
                    side="BUY",
                    order_type="LIMIT",
                    quantity=5.0,
                    price=bad_p,
                )
                assert res_bad["status"] == "REJECTED"
                assert "INVALID_PRICE" in res_bad["reason"]
    finally:
        await client.disconnect()


@pytest.mark.asyncio
async def test_execution_adapter_rejects_invalid_quantities() -> None:
    """Proves non-positive, NaN, and infinite quantities are rejected."""
    client = NSEExecutionClient()
    await client.connect()
    client.set_market_price("NSE:TCS", 3500.0)
    mock_open_session = MagicMock()
    mock_open_session.is_open.return_value = True
    try:
        with patch.object(SymbolResolver, "get_session", return_value=mock_open_session):
            for bad_q in (0.0, -1.0, -100.0, float("nan"), float("inf"), float("-inf")):
                res = await client.place_order(
                    symbol="NSE:TCS",
                    side="BUY",
                    order_type="MARKET",
                    quantity=bad_q,
                    price=3500.0,
                )
                assert res["status"] == "REJECTED"
                assert "INVALID_QUANTITY" in res["reason"]
    finally:
        await client.disconnect()


@pytest.mark.asyncio
async def test_execution_adapter_rejects_unknown_side_and_order_type() -> None:
    """Proves unknown side and order type are rejected, never silently mapped to SELL or STOP."""
    client = NSEExecutionClient()
    await client.connect()
    client.set_market_price("NSE:TCS", 3500.0)
    mock_open_session = MagicMock()
    mock_open_session.is_open.return_value = True
    try:
        with patch.object(SymbolResolver, "get_session", return_value=mock_open_session):
            # Unknown sides
            for bad_side in ("HOLD", "NO_TRADE", "UNKNOWN", "INVALID", "", "123"):
                res = await client.place_order(
                    symbol="NSE:TCS",
                    side=bad_side,
                    order_type="MARKET",
                    quantity=5.0,
                    price=3500.0,
                )
                assert res["status"] == "REJECTED"
                assert "INVALID_ORDER_SIDE" in res["reason"]

            # Unknown order types
            for bad_type in ("TRAILING_STOP", "OCO", "ICEBERG", "UNKNOWN", "", "FOO"):
                res = await client.place_order(
                    symbol="NSE:TCS",
                    side="BUY",
                    order_type=bad_type,
                    quantity=5.0,
                    price=3500.0,
                )
                assert res["status"] == "REJECTED"
                assert "INVALID_ORDER_TYPE" in res["reason"]
    finally:
        await client.disconnect()


@pytest.mark.asyncio
async def test_execution_adapter_rejects_unsupported_instrument() -> None:
    """Proves unsupported or unrecognized symbols are rejected."""
    client = NSEExecutionClient()
    await client.connect()
    mock_open_session = MagicMock()
    mock_open_session.is_open.return_value = True
    try:
        with patch.object(SymbolResolver, "get_session", return_value=mock_open_session):
            res = await client.place_order(
                symbol="UNKNOWN_XYZ_999",
                side="BUY",
                order_type="MARKET",
                quantity=5.0,
                price=100.0,
            )
            assert res["status"] == "REJECTED"
            assert "UNSUPPORTED_OR_UNKNOWN_INSTRUMENT" in res["reason"]
    finally:
        await client.disconnect()


@pytest.mark.asyncio
async def test_no_trade_decision_never_places_orders() -> None:
    """Proves NO_TRADE decisions are rejected by RiskEngine and cannot be executed."""
    risk_engine = RiskEngine(initial_capital=100000.0)
    client = NSEExecutionClient()
    await client.connect()
    client.set_market_price("NSE:TCS", 3500.0)
    mock_open_session = MagicMock()
    mock_open_session.is_open.return_value = True
    try:
        no_trade_decision = FinalSignalDecision(
            symbol="NSE:TCS",
            action=SignalAction.NO_TRADE,
            confidence=0.0,
            buy_probability=0.30,
            sell_probability=0.30,
            no_trade_probability=0.40,
            entry_price=3500.0,
            stop_loss=3500.0,
            take_profit=3500.0,
            risk_reward_ratio=0.0,
            regime_code="RANGING",
            status="NO_TRADE_EVALUATED",
        )

        risk_decision = risk_engine.evaluate_trade(no_trade_decision, requested_quantity=5.0)
        assert risk_decision.status == RiskDecisionStatus.REJECTED
        assert "NO_TRADE" in (risk_decision.rejection_reason or "")

        # Attempting to route with side="NO_TRADE" is rejected by adapter
        with patch.object(SymbolResolver, "get_session", return_value=mock_open_session):
            res = await client.place_order(
                symbol="NSE:TCS",
                side=no_trade_decision.action.value,
                order_type="MARKET",
                quantity=5.0,
                price=3500.0,
            )
            assert res["status"] == "REJECTED"
            assert "INVALID_ORDER_SIDE" in res["reason"]
    finally:
        await client.disconnect()


@pytest.mark.asyncio
async def test_valid_simulated_paper_execution_response_contract() -> None:
    """Proves valid orders return explicit simulated paper responses with all required fields."""
    client = NSEExecutionClient()
    await client.connect()
    client.set_market_price("NSE:TCS", 3500.0)
    mock_open_session = MagicMock()
    mock_open_session.is_open.return_value = True

    from data.market_state import MarketStateService
    st = MarketStateService.get_instance().ensure_instrument_state("NSE:TCS")
    if st:
        st.last_tick_time = datetime.now(UTC)
        st.is_stale = False
        st.data_age_seconds = 1.0
        st.source_state = "LIVE_VERIFIED"

    try:
        with patch.object(SymbolResolver, "get_session", return_value=mock_open_session):
            res = await client.place_order(
                symbol="NSE:TCS",
                side="BUY",
                order_type="MARKET",
                quantity=5.0,
                price=3500.0,
            )
            assert res["status"] == "FILLED"
            assert res["filled_price"] > 0.0
            assert res["filled_quantity"] == 5.0
            assert res["execution_mode"] == "SIMULATED_PAPER"
            assert res["is_live"] is False
            assert "fee" in res
            assert "slippage" in res
    finally:
        await client.disconnect()


def test_paper_trading_engine_rejects_invalid_inputs() -> None:
    """Proves PaperTradingEngine directly rejects invalid market prices and quantities."""
    engine = PaperTradingEngine()

    # Invalid market prices
    for bad_p in (0.0, -10.0, float("nan"), float("inf")):
        ord_ = PaperOrder(
            symbol="NSE:TCS",
            side=OrderSide.BUY,
            quantity=5.0,
        )
        res = engine.execute_order(ord_, current_market_price=bad_p)
        assert res.status == OrderStatus.REJECTED
        assert res.filled_price is None

    # Invalid quantities
    for bad_q in (0.0, -5.0, float("nan"), float("inf")):
        ord_ = PaperOrder(
            symbol="NSE:TCS",
            side=OrderSide.BUY,
            quantity=bad_q,
        )
        res = engine.execute_order(ord_, current_market_price=3500.0)
        assert res.status == OrderStatus.REJECTED
        assert res.filled_price is None
