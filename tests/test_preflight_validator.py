"""Tests for Institutional Pre-Trade Transaction Pre-Flight & FIX Protocol Compliance Validator."""

from datetime import datetime, timedelta, timezone
import pytest
from compliance.preflight_validator import (
    FIXNewOrderSingle,
    PreFlightRejectReason,
    PreFlightRiskLimits,
    PreFlightValidationResult,
    PreFlightValidator,
)


def test_preflight_approval_clean_order():
    """Verify clean institutional orders within risk limits are approved with sub-microsecond telemetry."""
    limits = PreFlightRiskLimits(
        max_single_order_notional_usd=500000.0,
        max_price_collar_pct=0.03,
    )
    validator = PreFlightValidator(limits=limits)

    order = FIXNewOrderSingle(
        cl_ord_id="ORD_20261005_001",
        symbol="BTC-USD",
        side="BUY",
        order_qty=1.5,
        ord_type="LIMIT",
        price=65100.0,
        account="PROP_ALPHA_01",
    )

    result = validator.validate_order(order, reference_mid_price=65000.0)
    assert result.is_approved is True
    assert result.reject_reason == PreFlightRejectReason.APPROVED
    assert result.cl_ord_id == "ORD_20261005_001"
    assert result.evaluated_notional_usd == 1.5 * 65100.0
    assert result.validation_latency_micros > 0.0


def test_fat_finger_price_collar_rejection():
    """Verify orders outside regulatory price collar bands are gated as FAT_FINGER_PRICE_COLLAR_BREACH."""
    validator = PreFlightValidator()

    # Mid = 65,000 -> 3% collar is +/- 1,950 (range: 63,050 to 66,950)
    fat_finger_order = FIXNewOrderSingle(
        cl_ord_id="ORD_FAT_001",
        symbol="BTC-USD",
        side="BUY",
        order_qty=1.0,
        ord_type="LIMIT",
        price=72000.0,  # ~10.7% deviation
    )

    result = validator.validate_order(fat_finger_order, reference_mid_price=65000.0)
    assert result.is_approved is False
    assert result.reject_reason == PreFlightRejectReason.FAT_FINGER_PRICE_COLLAR_BREACH
    assert "Price collar breach" in result.reject_message


def test_single_order_notional_and_quantity_caps():
    """Verify single-order size and notional limits block oversized orders."""
    limits = PreFlightRiskLimits(
        max_single_order_notional_usd=200000.0,
        max_single_order_qty={"BTC-USD": 5.0, "DEFAULT": 100.0},
    )
    validator = PreFlightValidator(limits=limits)

    # 1. Exceeds quantity cap
    qty_order = FIXNewOrderSingle(
        cl_ord_id="ORD_QTY_001",
        symbol="BTC-USD",
        side="BUY",
        order_qty=8.0,  # Limit is 5.0
        ord_type="LIMIT",
        price=20000.0,
    )
    res_qty = validator.validate_order(qty_order, reference_mid_price=20000.0)
    assert res_qty.is_approved is False
    assert res_qty.reject_reason == PreFlightRejectReason.EXCEEDS_SINGLE_ORDER_QUANTITY

    # 2. Exceeds notional cap
    notional_order = FIXNewOrderSingle(
        cl_ord_id="ORD_NOTIONAL_001",
        symbol="BTC-USD",
        side="BUY",
        order_qty=4.0,  # Qty OK (<= 5)
        ord_type="LIMIT",
        price=60000.0,  # 4 * 60,000 = $240,000 > $200,000
    )
    res_notional = validator.validate_order(notional_order, reference_mid_price=60000.0)
    assert res_notional.is_approved is False
    assert res_notional.reject_reason == PreFlightRejectReason.EXCEEDS_SINGLE_ORDER_NOTIONAL


def test_short_sale_locate_requirement():
    """Verify short sales without borrow locate identifiers are rejected under Reg SHO controls."""
    validator = PreFlightValidator()

    # 1. Short sale without locate
    unlocated_short = FIXNewOrderSingle(
        cl_ord_id="ORD_SHORT_001",
        symbol="BTC-USD",
        side="SELL_SHORT",
        order_qty=1.0,
        ord_type="LIMIT",
        price=65000.0,
        locate_reqd=True,
        locate_id=None,
    )
    res_unlocated = validator.validate_order(unlocated_short, reference_mid_price=65000.0)
    assert res_unlocated.is_approved is False
    assert res_unlocated.reject_reason == PreFlightRejectReason.SHORT_SALE_LOCATE_MISSING

    # 2. Short sale with valid locate
    located_short = FIXNewOrderSingle(
        cl_ord_id="ORD_SHORT_002",
        symbol="BTC-USD",
        side="SELL_SHORT",
        order_qty=1.0,
        ord_type="LIMIT",
        price=65000.0,
        locate_reqd=True,
        locate_id="LOCATE_CUSTODY_PRIME_99",
    )
    res_located = validator.validate_order(located_short, reference_mid_price=65000.0)
    assert res_located.is_approved is True
    assert res_located.reject_reason == PreFlightRejectReason.APPROVED


def test_duplicate_clordid_and_burst_rate_limiting():
    """Verify duplicate order rejection and sliding-window rate burst throttling."""
    limits = PreFlightRiskLimits(max_orders_per_second=3)
    validator = PreFlightValidator(limits=limits)

    order1 = FIXNewOrderSingle(
        cl_ord_id="CL_ORD_UNIQUE_1",
        symbol="ETH-USD",
        side="BUY",
        order_qty=1.0,
        ord_type="LIMIT",
        price=3500.0,
    )
    res1 = validator.validate_order(order1, reference_mid_price=3500.0)
    assert res1.is_approved is True

    # Duplicate submission
    res_dup = validator.validate_order(order1, reference_mid_price=3500.0)
    assert res_dup.is_approved is False
    assert res_dup.reject_reason == PreFlightRejectReason.DUPLICATE_CLORD_ID

    # Fill burst window up to 3 orders
    order2 = FIXNewOrderSingle(cl_ord_id="CL_ORD_UNIQUE_2", symbol="ETH-USD", side="BUY", order_qty=1.0, price=3500.0)
    order3 = FIXNewOrderSingle(cl_ord_id="CL_ORD_UNIQUE_3", symbol="ETH-USD", side="BUY", order_qty=1.0, price=3500.0)
    assert validator.validate_order(order2, reference_mid_price=3500.0).is_approved is True
    assert validator.validate_order(order3, reference_mid_price=3500.0).is_approved is True

    # 4th order within same second exceeds burst limit
    order4 = FIXNewOrderSingle(cl_ord_id="CL_ORD_UNIQUE_4", symbol="ETH-USD", side="BUY", order_qty=1.0, price=3500.0)
    res_burst = validator.validate_order(order4, reference_mid_price=3500.0)
    assert res_burst.is_approved is False
    assert res_burst.reject_reason == PreFlightRejectReason.BURST_RATE_LIMIT_EXCEEDED


def test_cumulative_daily_gross_ceiling_and_reset():
    """Verify cumulative gross trading turnover caps and daily counter reset."""
    limits = PreFlightRiskLimits(
        max_single_order_notional_usd=50000.0,
        max_daily_gross_notional_usd=70000.0,
    )
    validator = PreFlightValidator(limits=limits)

    # Order 1: $40,000 -> Approved (cumulative = $40,000)
    o1 = FIXNewOrderSingle(cl_ord_id="O1", symbol="SOL-USD", side="BUY", order_qty=200.0, price=200.0)
    assert validator.validate_order(o1, reference_mid_price=200.0).is_approved is True

    # Order 2: $40,000 -> Projected = $80,000 > $70,000 ceiling -> Rejected
    o2 = FIXNewOrderSingle(cl_ord_id="O2", symbol="SOL-USD", side="BUY", order_qty=200.0, price=200.0)
    res2 = validator.validate_order(o2, reference_mid_price=200.0)
    assert res2.is_approved is False
    assert res2.reject_reason == PreFlightRejectReason.EXCEEDS_DAILY_GROSS_CREDIT_CEILING

    # Reset
    validator.reset_daily_limits()
    assert validator._cumulative_daily_gross_usd == 0.0
    res3 = validator.validate_order(o2, reference_mid_price=200.0)
    assert res3.is_approved is True
