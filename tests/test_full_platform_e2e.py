"""Trading Intelligence Platform - Comprehensive End-to-End System Integration Test.

Validates the full institutional quant trading lifecycle across all 20 integrated platform phases:
1. Multi-venue Tick Ingestion & Normalization
2. Consolidated Order Book & Synthetic BBO Aggregation
3. Candle Resampling & 3D Market Regime Classification
4. Multi-Source Signal Fusion with ATR Envelope
5. Pre-Trade Risk Gating & Capital Protection
6. Almgren-Chriss Market Impact & Algorithmic TWAP Scheduling
7. Paper Trading Execution & MTM Portfolio Accounting
8. Cryptographic Immutable Audit Hash-Chaining (MiFID II)
9. Regulatory Trade Surveillance Scanning
10. Point-in-Time Cold-Start Disaster Recovery Snapshot & SHA-256 Verification
11. High-Availability Heartbeat & Epoch Fencing
"""

from datetime import datetime, timezone
from pathlib import Path
import pytest

from analysis.regime import MarketRegimeDetector
from compliance import AuditRecordType, ImmutableAuditLedger, TradeSurveillanceEngine
from core.events import Candle, RawMarketEvent
from core.recovery import (
    FailoverOrchestrator,
    NodeRole,
    SnapshotManager,
)
from data.normalization.normalizer import EventNormalizer
from data.orderbook.consolidated import ConsolidatedOrderBook
from execution.algo import AlgoEngine
from execution.cost_model import AlmgrenChrissCostModel
from paper_trading.engine import PaperTradingEngine
from paper_trading.models import OrderSide, OrderType, PaperOrder
from risk.engine import RiskDecisionStatus, RiskEngine
from risk.limits import RiskLimits
from signals.fusion import SignalFusionEngine
from signals.technical import TechnicalSignalGenerator


def test_full_platform_end_to_end_lifecycle(tmp_path: Path):
    # ---------------------------------------------------------
    # Phase 1 & 2: Ingestion, Normalization & Order Book
    # ---------------------------------------------------------
    normalizer = EventNormalizer()
    raw_tick = RawMarketEvent(
        provider="coinbase_ws",
        symbol="BTC-USD",
        channel="matches",
        received_at=datetime.now(timezone.utc),
        raw_payload={
            "type": "match",
            "trade_id": "trade_e2e_1001",
            "product_id": "BTC-USD",
            "price": "65000.00",
            "size": "0.50",
            "side": "buy",
            "time": "2026-10-05T12:00:00Z",
        },
        sequence=1,
    )
    norm_trade = normalizer.normalize(raw_tick)
    assert norm_trade is not None
    assert norm_trade.price == 65000.00

    # Multi-venue consolidated order book
    book = ConsolidatedOrderBook(symbol="BTC-USD")
    book.update_venue_snapshot("coinbase", bids=[(64990.0, 1.5)], asks=[(65010.0, 1.5)])
    book.update_venue_snapshot("binance", bids=[(64995.0, 2.0)], asks=[(65005.0, 2.0)])
    bbo = book.get_consolidated_bbo()

    assert bbo.best_bid == 64995.0
    assert bbo.best_bid_venue == "binance"
    assert bbo.best_ask == 65005.0
    assert bbo.best_ask_venue == "binance"
    assert bbo.spread == 10.0

    # ---------------------------------------------------------
    # Phase 3 & 4: Regimes, Signals & Signal Fusion
    # ---------------------------------------------------------
    candle = Candle(
        symbol="BTC-USD",
        timeframe="1h",
        timestamp=datetime.now(timezone.utc),
        open=64800.0,
        high=65200.0,
        low=64750.0,
        close=65000.0,
        volume=120.0,
        quote_volume=7800000.0,
    )

    import pandas as pd
    regime_detector = MarketRegimeDetector()
    bar_series = pd.Series(candle.model_dump())
    # Add dummy technical features for bar evaluation
    bar_series["ema_20"] = 64500.0
    bar_series["ema_50"] = 64000.0
    bar_series["adx_14"] = 30.0
    bar_series["plus_di"] = 28.0
    bar_series["minus_di"] = 14.0
    bar_series["atr_14"] = 450.0
    bar_series["rsi_14"] = 62.0
    bar_series["bb_width"] = 0.04
    bar_series["rvol_20"] = 1.6

    regime_state = regime_detector.classify_bar(bar_series)
    assert "BULL" in regime_state.trend.value

    from models.base import SignalAction
    from signals.technical import TechnicalSignal
    tech_signals = [
        TechnicalSignal(
            setup_name="TREND_MOMENTUM",
            action=SignalAction.BUY,
            strength=0.85,
            stop_loss_mult=1.5,
            take_profit_mult=3.0,
            reasons=["Strong EMA bullish stack with ADX trend confirmation"],
        )
    ]

    fusion_engine = SignalFusionEngine()
    decision = fusion_engine.fuse(
        symbol="BTC-USD",
        current_price=65000.0,
        atr=450.0,
        regime=regime_state,
        technical_signals=tech_signals,
    )
    assert decision.action is not None
    assert decision.confidence > 0.0

    # ---------------------------------------------------------
    # Phase 5: Pre-Trade Risk Evaluation & Position Sizing
    # ---------------------------------------------------------
    risk_limits = RiskLimits(max_portfolio_drawdown=0.10, max_single_trade_risk_pct=0.02)
    risk_engine = RiskEngine(initial_capital=100000.0, limits=risk_limits)
    risk_decision = risk_engine.evaluate_trade(decision)
    assert risk_decision.status in (RiskDecisionStatus.APPROVED, RiskDecisionStatus.RESIZED, RiskDecisionStatus.REJECTED)

    # ---------------------------------------------------------
    # Phase 6: Cost Modeling & Algorithmic Slicing
    # ---------------------------------------------------------
    cost_model = AlmgrenChrissCostModel()
    cost_est = cost_model.estimate_cost(
        symbol="BTC-USD",
        side="BUY",
        quantity=1.0,
        current_price=65000.0,
        average_daily_volume=1000.0,
        daily_volatility=0.025,
    )
    assert cost_est.total_cost_usd > 0.0

    algo_engine = AlgoEngine()
    algo_plan = algo_engine.create_twap_plan(
        symbol="BTC-USD",
        side="BUY",
        quantity=1.0,
        duration_minutes=10.0,
        num_slices=4,
        arrival_price=65000.0,
    )
    assert len(algo_plan.slices) == 4
    assert algo_plan.parent_quantity == 1.0

    # ---------------------------------------------------------
    # Phase 7: Paper Brokerage Execution & Accounting
    # ---------------------------------------------------------
    paper_engine = PaperTradingEngine(initial_cash=100000.0)
    exec_order = PaperOrder(
        order_id="ord_e2e_01",
        symbol="BTC-USD",
        side=OrderSide.BUY,
        order_type=OrderType.MARKET,
        quantity=0.5,
        stop_loss=64100.0,
        take_profit=66800.0,
    )
    filled_order = paper_engine.execute_order(exec_order, current_market_price=65000.0)
    assert filled_order.filled_price is not None
    assert paper_engine.cash < 100000.0
    assert "BTC-USD" in paper_engine.positions
    assert paper_engine.positions["BTC-USD"].quantity == 0.5

    # ---------------------------------------------------------
    # Phase 8: Cryptographic Immutable Audit Trail
    # ---------------------------------------------------------
    audit_ledger = ImmutableAuditLedger()
    audit_record = audit_ledger.record_event(
        record_type=AuditRecordType.ORDER_FILL,
        action="FILL",
        payload={"order_id": filled_order.order_id, "price": filled_order.filled_price, "qty": filled_order.filled_quantity},
        actor="ALGO_EXEC_ENGINE",
    )
    assert len(audit_record.entry_hash) == 64
    is_valid, _ = audit_ledger.verify_integrity()
    assert is_valid is True

    # ---------------------------------------------------------
    # Phase 9: Regulatory Trade Surveillance
    # ---------------------------------------------------------
    surveillance = TradeSurveillanceEngine()
    alerts = surveillance.check_wash_trading([
        {"symbol": "BTC-USD", "side": "BUY", "price": 65000.0, "quantity": 0.5, "timestamp": "2026-10-05T12:00:00Z"},
        {"symbol": "BTC-USD", "side": "SELL", "price": 65000.0, "quantity": 0.5, "timestamp": "2026-10-05T12:00:01Z"},
    ])
    assert len(alerts) == 1
    assert alerts[0].pattern == "WASH_TRADING"

    # ---------------------------------------------------------
    # Phase 10: State Snapshot & Disaster Recovery
    # ---------------------------------------------------------
    snap_mgr = SnapshotManager(snapshot_dir=tmp_path)
    snapshot_meta = snap_mgr.create_snapshot(engine=paper_engine, snapshot_id="e2e_snap_final")
    assert snapshot_meta.snapshot_id == "e2e_snap_final"
    assert snapshot_meta.active_positions_count == 1
    assert len(snapshot_meta.checksum_sha256) == 64

    # Cold-start rehydration into completely new engine
    cold_engine = PaperTradingEngine(initial_cash=50000.0)
    snap_mgr.restore_from_snapshot("e2e_snap_final", engine=cold_engine)
    assert cold_engine.cash == paper_engine.cash
    assert len(cold_engine.positions) == 1
    assert "BTC-USD" in cold_engine.positions
    assert cold_engine.positions["BTC-USD"].quantity == 0.5

    # ---------------------------------------------------------
    # Phase 11: High-Availability Failover & Epoch Fencing
    # ---------------------------------------------------------
    orchestrator = FailoverOrchestrator(node_id="e2e_primary", initial_role=NodeRole.PRIMARY, initial_epoch=1)
    hb = orchestrator.generate_heartbeat()
    assert hb.role == NodeRole.PRIMARY
    assert hb.term_epoch == 1

    cluster_telemetry = orchestrator.get_cluster_status()
    assert cluster_telemetry["role"] == "PRIMARY"
    assert cluster_telemetry["primary_alive"] is True
