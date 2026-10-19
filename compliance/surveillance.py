"""Trading Intelligence Platform - Regulatory Trade Surveillance Engine.

Monitors order flow and execution events for market abuse patterns in accordance with
MAR (Market Abuse Regulation) and FINRA Rule 5210:
- Wash Trading (pre-arranged trading with no change in beneficial ownership)
- Spoofing & Layering (submitting non-bona fide orders to manipulate order book depth)
- Quote Stuffing (excessive message bursts designed to induce queue latency)
- Momentum Ignition (aggressive orders intended to trigger stops or cascades)
"""

from collections import defaultdict
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional
import uuid
from pydantic import BaseModel, Field

from core.logging import get_logger

logger = get_logger("trade_surveillance")


class SurveillanceAlertSeverity(str, Enum):
    """Regulatory severity classifications."""

    INFO = "INFO"
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class AbusePatternType(str, Enum):
    """Catalog of detected market abuse patterns."""

    WASH_TRADING = "WASH_TRADING"
    SPOOFING_LAYERING = "SPOOFING_LAYERING"
    QUOTE_STUFFING = "QUOTE_STUFFING"
    MOMENTUM_IGNITION = "MOMENTUM_IGNITION"


class SurveillanceAlert(BaseModel):
    """Regulatory surveillance alert notification."""

    alert_id: str = Field(default_factory=lambda: f"alt_{uuid.uuid4().hex[:8]}")
    pattern: AbusePatternType
    severity: SurveillanceAlertSeverity
    symbol: str
    detected_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    description: str
    evidence: Dict[str, Any]


def _parse_timestamp(ts: Any) -> datetime:
    """Safely parse datetime object or ISO-8601 string to timezone-aware UTC datetime."""
    if isinstance(ts, datetime):
        return ts if ts.tzinfo is not None else ts.replace(tzinfo=timezone.utc)
    if isinstance(ts, str):
        try:
            return datetime.fromisoformat(ts.replace("Z", "+00:00"))
        except Exception:
            pass
    return datetime.now(timezone.utc)


class TradeSurveillanceEngine:
    """Surveillance detector flagging prohibited abusive trading patterns."""

    def __init__(
        self,
        wash_window_seconds: float = 5.0,
        spoof_cancel_ratio_threshold: float = 5.0,
        quote_stuffing_rate_per_sec: int = 50,
    ) -> None:
        self.wash_window_seconds = wash_window_seconds
        self.spoof_cancel_ratio_threshold = spoof_cancel_ratio_threshold
        self.quote_stuffing_rate_per_sec = quote_stuffing_rate_per_sec

    def check_wash_trading(
        self,
        orders: List[Dict[str, Any]],
    ) -> List[SurveillanceAlert]:
        """Detect potential wash trading: simultaneous buy/sell orders at matching prices."""
        alerts: List[SurveillanceAlert] = []
        if len(orders) < 2:
            return alerts

        # Sort orders chronologically
        sorted_orders = sorted(
            orders,
            key=lambda o: _parse_timestamp(o.get("timestamp")),
        )

        for i in range(len(sorted_orders)):
            o1 = sorted_orders[i]
            t1 = _parse_timestamp(o1.get("timestamp"))
            sym1 = o1.get("symbol", "").upper()
            side1 = o1.get("side", "").upper()
            p1 = float(o1.get("price", 0.0))
            q1 = float(o1.get("quantity", 0.0))
            acct1 = o1.get("account_id", "default")

            for j in range(i + 1, len(sorted_orders)):
                o2 = sorted_orders[j]
                t2 = _parse_timestamp(o2.get("timestamp"))
                sym2 = o2.get("symbol", "").upper()
                side2 = o2.get("side", "").upper()
                p2 = float(o2.get("price", 0.0))
                q2 = float(o2.get("quantity", 0.0))
                acct2 = o2.get("account_id", "default")

                if sym1 != sym2:
                    continue

                # Check time proximity window
                time_delta = abs((t2 - t1).total_seconds())
                if time_delta > self.wash_window_seconds:
                    break

                # Wash condition: opposite sides, matching or crossing prices, close quantities
                if side1 != side2 and abs(p1 - p2) <= (p1 * 0.0005) and abs(q1 - q2) <= (q1 * 0.05):
                    # Flag potential wash trade
                    alert = SurveillanceAlert(
                        pattern=AbusePatternType.WASH_TRADING,
                        severity=SurveillanceAlertSeverity.HIGH if acct1 == acct2 else SurveillanceAlertSeverity.MEDIUM,
                        symbol=sym1,
                        description=(
                            f"Suspicious self-crossing wash trade detected on {sym1}: "
                            f"BUY {q1} @ {p1} and SELL {q2} @ {p2} within {time_delta:.2f}s"
                        ),
                        evidence={
                            "order_1": o1,
                            "order_2": o2,
                            "time_delta_seconds": time_delta,
                            "price_diff": abs(p1 - p2),
                            "quantity_diff": abs(q1 - q2),
                            "accounts": [acct1, acct2],
                        },
                    )
                    alerts.append(alert)
                    logger.warning("Surveillance alert: Wash trading detected", alert_id=alert.alert_id, symbol=sym1)

        return alerts

    def check_spoofing_and_layering(
        self,
        order_events: List[Dict[str, Any]],
        max_duration_seconds: float = 10.0,
    ) -> List[SurveillanceAlert]:
        """Detect spoofing/layering: rapid order submission and cancellation on one side before fill on opposite side."""
        alerts: List[SurveillanceAlert] = []
        if len(order_events) < 5:
            return alerts

        # Group events by symbol
        symbol_events = defaultdict(list)
        for ev in order_events:
            symbol_events[ev.get("symbol", "").upper()].append(ev)

        for sym, events in symbol_events.items():
            cancels_count = sum(1 for e in events if e.get("event_type") == "CANCEL")
            fills_count = sum(1 for e in events if e.get("event_type") == "FILL")
            submissions_count = sum(1 for e in events if e.get("event_type") == "SUBMISSION")

            # Check for excessive cancellations relative to fills
            if submissions_count >= 5 and cancels_count >= 4 and fills_count <= 1:
                # Inspect timing: are cancels occurring within rapid succession?
                timestamps = [_parse_timestamp(e.get("timestamp")) for e in events if e.get("timestamp")]
                if timestamps:
                    span = (max(timestamps) - min(timestamps)).total_seconds()
                    if span <= max_duration_seconds:
                        cancel_ratio = cancels_count / max(fills_count, 1)
                        if cancel_ratio >= self.spoof_cancel_ratio_threshold:
                            alert = SurveillanceAlert(
                                pattern=AbusePatternType.SPOOFING_LAYERING,
                                severity=SurveillanceAlertSeverity.HIGH,
                                symbol=sym,
                                description=(
                                    f"Potential spoofing/layering pattern detected on {sym}: "
                                    f"{submissions_count} submissions, {cancels_count} cancellations, "
                                    f"{fills_count} fills within {span:.2f}s (Cancel/Fill ratio: {cancel_ratio:.1f})"
                                ),
                                evidence={
                                    "submissions": submissions_count,
                                    "cancellations": cancels_count,
                                    "fills": fills_count,
                                    "cancel_ratio": cancel_ratio,
                                    "duration_seconds": span,
                                },
                            )
                            alerts.append(alert)
                            logger.warning("Surveillance alert: Spoofing/Layering detected", alert_id=alert.alert_id, symbol=sym)

        return alerts

    def check_quote_stuffing(
        self,
        messages: List[Dict[str, Any]],
    ) -> List[SurveillanceAlert]:
        """Detect quote stuffing: abnormal burst of order messages exceeding rate ceiling."""
        alerts: List[SurveillanceAlert] = []
        if not messages:
            return alerts

        # Bin messages by 1-second UTC window
        second_bins = defaultdict(list)
        for msg in messages:
            ts = _parse_timestamp(msg.get("timestamp"))
            sec_key = int(ts.timestamp())
            second_bins[sec_key].append(msg)

        for sec_key, msgs_in_sec in second_bins.items():
            rate = len(msgs_in_sec)
            if rate > self.quote_stuffing_rate_per_sec:
                sym = msgs_in_sec[0].get("symbol", "PORTFOLIO").upper()
                alert = SurveillanceAlert(
                    pattern=AbusePatternType.QUOTE_STUFFING,
                    severity=SurveillanceAlertSeverity.CRITICAL if rate > (self.quote_stuffing_rate_per_sec * 2) else SurveillanceAlertSeverity.HIGH,
                    symbol=sym,
                    description=(
                        f"Quote stuffing threshold breached: {rate} messages submitted in 1-second window "
                        f"(ceiling: {self.quote_stuffing_rate_per_sec} msg/sec)"
                    ),
                    evidence={
                        "timestamp_second": sec_key,
                        "observed_rate_msgs_per_sec": rate,
                        "threshold_msgs_per_sec": self.quote_stuffing_rate_per_sec,
                    },
                )
                alerts.append(alert)
                logger.warning("Surveillance alert: Quote stuffing detected", rate=rate, second=sec_key)

        return alerts

    def scan_all(
        self,
        order_events: List[Dict[str, Any]],
    ) -> List[SurveillanceAlert]:
        """Run full battery of regulatory surveillance scans."""
        all_alerts: List[SurveillanceAlert] = []
        all_alerts.extend(self.check_wash_trading(order_events))
        all_alerts.extend(self.check_spoofing_and_layering(order_events))
        all_alerts.extend(self.check_quote_stuffing(order_events))
        return all_alerts
