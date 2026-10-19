"""Tests for Regulatory Trade Surveillance Engine and Market Abuse Detection."""

from datetime import datetime, timedelta, timezone
import pytest

from compliance.surveillance import (
    AbusePatternType,
    SurveillanceAlert,
    SurveillanceAlertSeverity,
    TradeSurveillanceEngine,
)


def test_surveillance_wash_trading_detection():
    """Verify detection of self-crossing wash trades at matching prices."""
    engine = TradeSurveillanceEngine(wash_window_seconds=5.0)
    now = datetime.now(timezone.utc)

    orders = [
        {
            "order_id": "ord_buy_1",
            "symbol": "BTC-USD",
            "side": "BUY",
            "price": 65000.0,
            "quantity": 1.0,
            "account_id": "acct_prime",
            "timestamp": now,
        },
        {
            "order_id": "ord_sell_1",
            "symbol": "BTC-USD",
            "side": "SELL",
            "price": 65000.0,
            "quantity": 1.0,
            "account_id": "acct_prime",
            "timestamp": now + timedelta(seconds=1.5),
        },
    ]

    alerts = engine.check_wash_trading(orders)
    assert len(alerts) == 1
    alert = alerts[0]
    assert alert.pattern == AbusePatternType.WASH_TRADING
    assert alert.severity == SurveillanceAlertSeverity.HIGH
    assert alert.symbol == "BTC-USD"
    assert "Wash trade" in alert.description or "wash trade" in alert.description


def test_surveillance_spoofing_and_layering_detection():
    """Verify detection of excessive cancellations and layering patterns."""
    engine = TradeSurveillanceEngine(spoof_cancel_ratio_threshold=4.0)
    now = datetime.now(timezone.utc)

    # 5 submissions followed by 5 rapid cancels on BUY side, then 1 fill on SELL side
    events = []
    for i in range(5):
        events.append({
            "event_type": "SUBMISSION",
            "symbol": "SOL-USD",
            "side": "BUY",
            "price": 140.0 - (i * 0.1),
            "timestamp": now + timedelta(milliseconds=i * 200),
        })
    for i in range(5):
        events.append({
            "event_type": "CANCEL",
            "symbol": "SOL-USD",
            "side": "BUY",
            "timestamp": now + timedelta(milliseconds=1500 + i * 100),
        })
    events.append({
        "event_type": "FILL",
        "symbol": "SOL-USD",
        "side": "SELL",
        "price": 141.0,
        "timestamp": now + timedelta(seconds=2.5),
    })

    alerts = engine.check_spoofing_and_layering(events, max_duration_seconds=5.0)
    assert len(alerts) == 1
    alert = alerts[0]
    assert alert.pattern == AbusePatternType.SPOOFING_LAYERING
    assert alert.severity == SurveillanceAlertSeverity.HIGH
    assert alert.symbol == "SOL-USD"


def test_surveillance_quote_stuffing_detection():
    """Verify detection of excessive message bursts exceeding rate ceiling."""
    engine = TradeSurveillanceEngine(quote_stuffing_rate_per_sec=30)
    now = datetime.now(timezone.utc)

    # 40 messages submitted in the exact same second
    messages = [
        {"symbol": "ETH-USD", "msg_id": f"msg_{i}", "timestamp": now}
        for i in range(40)
    ]

    alerts = engine.check_quote_stuffing(messages)
    assert len(alerts) == 1
    alert = alerts[0]
    assert alert.pattern == AbusePatternType.QUOTE_STUFFING
    assert alert.symbol == "ETH-USD"
    assert alert.evidence["observed_rate_msgs_per_sec"] == 40


def test_surveillance_scan_all_clean():
    """Verify clean orders generate zero false-positive alerts."""
    engine = TradeSurveillanceEngine()
    now = datetime.now(timezone.utc)

    # Normal sequential orders spaced out in time
    clean_orders = [
        {"symbol": "BTC-USD", "side": "BUY", "price": 64000.0, "quantity": 0.5, "timestamp": now},
        {"symbol": "ETH-USD", "side": "BUY", "price": 3400.0, "quantity": 2.0, "timestamp": now + timedelta(minutes=10)},
    ]

    alerts = engine.scan_all(clean_orders)
    assert len(alerts) == 0
