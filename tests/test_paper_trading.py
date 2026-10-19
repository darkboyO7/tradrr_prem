"""Tests for Paper Trading Engine and Order Execution Simulator."""

from datetime import UTC, datetime

from core.events import Candle
from paper_trading.engine import PaperTradingEngine
from paper_trading.models import OrderSide, OrderStatus, OrderType, PaperOrder
from paper_trading.simulator import ExecutionSimulator


def test_execution_simulator_fees_and_slippage():
    sim = ExecutionSimulator(half_spread_bps=1.0, taker_fee_pct=0.001, slippage_base_bps=2.0)
    order = PaperOrder(
        order_id="ord_1",
        symbol="BTC-USD",
        side=OrderSide.BUY,
        order_type=OrderType.MARKET,
        quantity=1.0,
    )

    market_price = 50000.0
    fill_price, fee, slippage = sim.simulate_fill(order, market_price)

    # Buy price must be higher than market price (slippage + spread penalty)
    assert fill_price > market_price
    assert fee > 0.0
    assert slippage > 0.0


def test_paper_trading_buy_and_manual_close():
    engine = PaperTradingEngine(initial_cash=100000.0)

    # 1. Submit Market Buy order for 1.0 BTC at $60,000
    order = PaperOrder(
        order_id="ord_buy_1",
        symbol="BTC-USD",
        side=OrderSide.BUY,
        order_type=OrderType.MARKET,
        quantity=1.0,
        stop_loss=58000.0,
        take_profit=64000.0,
    )
    executed = engine.execute_order(order, current_market_price=60000.0)
    assert executed.status == OrderStatus.FILLED
    assert "BTC-USD" in engine.positions

    pos = engine.positions["BTC-USD"]
    assert pos.quantity == 1.0
    assert pos.is_open

    # 2. Market price moves up to $62,000 -> Unrealized PnL = ~$2,000
    engine.on_tick("BTC-USD", 62000.0)
    assert pos.unrealized_pnl > 1800.0
    assert engine.equity > 101800.0

    # 3. Submit Sell order to close position at $62,000
    close_order = PaperOrder(
        order_id="ord_sell_1",
        symbol="BTC-USD",
        side=OrderSide.SELL,
        order_type=OrderType.MARKET,
        quantity=1.0,
    )
    engine.execute_order(close_order, current_market_price=62000.0)
    assert "BTC-USD" not in engine.positions
    assert len(engine.closed_positions) == 1
    closed = engine.closed_positions[0]
    assert closed.realized_pnl > 1800.0
    assert engine.cash > 101800.0


def test_paper_trading_stop_loss_trigger():
    engine = PaperTradingEngine(initial_cash=50000.0)

    order = PaperOrder(
        order_id="ord_buy_sl",
        symbol="ETH-USD",
        side=OrderSide.BUY,
        order_type=OrderType.MARKET,
        quantity=10.0,
        stop_loss=2900.0,
        take_profit=3300.0,
    )
    engine.execute_order(order, current_market_price=3000.0)
    assert "ETH-USD" in engine.positions

    # Price drops to $2850 (breaching stop loss at $2900)
    events = engine.on_tick("ETH-USD", 2850.0)
    assert len(events) == 1
    assert "STOP_LOSS_HIT" in events[0]

    # Position must be automatically closed
    assert "ETH-USD" not in engine.positions
    assert len(engine.closed_positions) == 1
    closed = engine.closed_positions[0]
    assert closed.close_reason.startswith("STOP_LOSS_HIT")
    assert closed.realized_pnl < 0.0


def test_paper_trading_take_profit_candle():
    engine = PaperTradingEngine(initial_cash=50000.0)

    order = PaperOrder(
        order_id="ord_buy_tp",
        symbol="SOL-USD",
        side=OrderSide.BUY,
        order_type=OrderType.MARKET,
        quantity=100.0,
        stop_loss=140.0,
        take_profit=170.0,
    )
    engine.execute_order(order, current_market_price=150.0)

    # Ingest candle where high reaches $172 (breaches take-profit at $170)
    candle = Candle(
        symbol="SOL-USD",
        timeframe="1m",
        timestamp=datetime(2026, 10, 4, 12, 0, 0, tzinfo=UTC),
        open=155.0,
        high=172.0,
        low=154.0,
        close=168.0,
        volume=500.0,
        is_closed=True,
    )
    events = engine.on_candle(candle)
    assert len(events) >= 1
    assert "TAKE_PROFIT_HIT" in events[0]

    # Position must be automatically closed
    assert "SOL-USD" not in engine.positions
    assert len(engine.closed_positions) == 1
    closed = engine.closed_positions[0]
    assert closed.realized_pnl > 0.0
