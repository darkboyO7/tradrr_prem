"""Tests for Institutional Real-Time Order Flow Toxicity & Adverse Selection Engine (VPIN & Roll Spread)."""

import numpy as np
import pytest

from analysis.flow_toxicity import (
    FlowToxicityRegime,
    OrderFlowToxicityEngine,
    ToxicityMetricsSnapshot,
    VolumeBucket,
)


@pytest.fixture
def btc_toxicity_engine():
    """Setup toxicity engine with 10 BTC bucket volume and 20 bucket window."""
    return OrderFlowToxicityEngine(
        symbol="BTC-USD",
        bucket_volume=10.0,
        vpin_window_buckets=20,
    )


def test_benign_order_flow_vpin(btc_toxicity_engine):
    """Test balanced alternating buy/sell flow resulting in low VPIN (benign flow)."""
    eng = btc_toxicity_engine
    eng.update_bbo(bid=65000.0, ask=65002.0)
    t = 1_000_000_000

    # Feed 200 trades alternating buy and sell of 1.0 BTC
    # Total volume = 200 BTC = 20 buckets of 10 BTC each
    # In each bucket: 5 buy, 5 sell -> order imbalance ~ 0 -> VPIN ~ 0
    for i in range(200):
        t += 100_000_000
        side = "BUY" if i % 2 == 0 else "SELL"
        p = 65002.0 if side == "BUY" else 65000.0
        eng.process_trade(price=p, quantity=1.0, timestamp_ns=t, side=side)

    snap = eng.get_metrics_snapshot()
    assert isinstance(snap, ToxicityMetricsSnapshot)
    assert snap.completed_buckets_count == 20
    assert snap.current_vpin < 0.15
    assert snap.toxicity_regime == FlowToxicityRegime.BENIGN
    assert "BENIGN" in snap.recommendation


def test_toxic_unidirectional_order_flow(btc_toxicity_engine):
    """Test aggressive unidirectional buy flow resulting in extreme VPIN."""
    eng = btc_toxicity_engine
    t = 1_000_000_000

    # Feed 100 consecutive buy trades of 2.0 BTC (200 BTC total = 20 buckets)
    # In each bucket: 10 buy, 0 sell -> imbalance = 10 -> VPIN = 1.0 (Extreme Adverse)
    for i in range(100):
        t += 50_000_000
        p = 65000.0 + i * 2.0  # Climbing price
        eng.process_trade(price=p, quantity=2.0, timestamp_ns=t, side="BUY")

    snap = eng.get_metrics_snapshot()
    assert snap.current_vpin == pytest.approx(1.0, rel=1e-3)
    assert snap.toxicity_regime == FlowToxicityRegime.EXTREME_ADVERSE
    assert "CRITICAL TOXICITY" in snap.recommendation


def test_roll_effective_spread():
    """Test Roll (1984) model recovering effective spread from negative price autocovariance."""
    eng = OrderFlowToxicityEngine(symbol="ETH-USD")

    # True spread s = 2.0 (half-spread = 1.0) with random buy/sell arrivals
    np.random.seed(42)
    q = np.random.choice([-1.0, 1.0], size=500)
    prices = 3000.0 + 1.0 * q

    t = 1_000_000_000
    for p in prices:
        t += 10_000
        eng.process_trade(price=float(p), quantity=0.5, timestamp_ns=t)

    spread, bps = eng.compute_roll_spread()
    # Roll spread = 2 * sqrt(-cov) ~= 2.0
    assert spread > 1.7
    assert spread < 2.3
    assert bps > 0.0


def test_lee_ready_trade_direction_classification():
    """Test quote rule and tick rule trade direction classification."""
    eng = OrderFlowToxicityEngine(symbol="SOL-USD", bucket_volume=5.0)

    # Set BBO mid at 150.0 (bid 149.9, ask 150.1)
    eng.update_bbo(bid=149.9, ask=150.1)

    # Trade at 150.05 > mid -> Classified as BUY
    b1 = eng.process_trade(price=150.05, quantity=5.0, timestamp_ns=1000)
    assert b1 is not None
    assert b1.buy_volume == 5.0
    assert b1.sell_volume == 0.0

    # Trade at 149.85 < mid -> Classified as SELL
    b2 = eng.process_trade(price=149.85, quantity=5.0, timestamp_ns=2000)
    assert b2 is not None
    assert b2.buy_volume == 0.0
    assert b2.sell_volume == 5.0


def test_kyle_lambda_and_snapshot_serialization(btc_toxicity_engine):
    """Test Kyle's lambda price impact computation and dictionary serialization."""
    eng = btc_toxicity_engine

    # Trades with positive price response to buy direction
    t = 1_000_000_000
    p = 60000.0
    for i in range(20):
        t += 1000
        direction = 1 if i % 2 == 0 else -1
        p += direction * 5.0
        eng.process_trade(price=p, quantity=1.0, timestamp_ns=t, side="BUY" if direction > 0 else "SELL")

    kyle = eng.compute_kyle_lambda()
    assert kyle >= 0.0

    snap = eng.get_metrics_snapshot()
    d = snap.to_dict()
    assert d["symbol"] == "BTC-USD"
    assert "current_vpin" in d
    assert "roll_effective_spread" in d
    assert "kyle_lambda_impact" in d
