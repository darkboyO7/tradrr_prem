"""Tests for Institutional Portfolio Margin, Haircut Model, and Cross-Collateralization Engine."""

import pytest

from risk.portfolio_margin import (
    CollateralAsset,
    MarginPosition,
    MarginStatusLevel,
    PortfolioMarginEngine,
    PortfolioMarginReport,
    SubAccountBook,
)


def test_haircut_discounting():
    engine = PortfolioMarginEngine()

    assert engine.get_haircut("USD") == 0.0
    assert engine.get_haircut("BTC") == 0.10
    assert engine.get_haircut("ETH") == 0.15
    assert engine.get_haircut("UNKNOWN_ALT") == 0.35

    btc_collateral = CollateralAsset(asset="BTC", quantity=2.0, mark_price=60000.0, haircut_pct=0.10)
    assert btc_collateral.nominal_value_usd == 120000.0
    assert btc_collateral.collateral_value_usd == 108000.0  # 10% discount


def test_margin_evaluation_healthy():
    engine = PortfolioMarginEngine()

    collateral = [
        CollateralAsset(asset="BTC", quantity=1.0, mark_price=65000.0, haircut_pct=0.10),
    ]
    positions = [
        MarginPosition(symbol="BTC-USD", side="LONG", quantity=0.5, mark_price=65000.0, maintenance_margin_rate=0.05),
    ]

    report = engine.evaluate_margin(collateral_assets=collateral, positions=positions, cash_balance=20000.0)

    assert isinstance(report, PortfolioMarginReport)
    # Total collateral = 20000 (cash) + 58500 (BTC discounted) = 78500
    assert report.total_collateral_value_usd == 78500.0
    # Notional = 0.5 * 65000 = 32500, MM = 32500 * 0.05 = 1625.0
    assert report.total_maintenance_margin_usd == 1625.0
    assert report.margin_usage_pct < 10.0
    assert report.status == MarginStatusLevel.HEALTHY
    assert "NORMAL" in report.action_required


def test_margin_call_and_liquidation_breach():
    engine = PortfolioMarginEngine()

    collateral = []
    # Cash = $10,000
    # 1. High leverage: notional $170,000 at 5% MM = $8,500 (85% usage) -> MARGIN_CALL
    pos_heavy = [
        MarginPosition(symbol="BTC-USD", side="LONG", quantity=2.6154, mark_price=65000.0, maintenance_margin_rate=0.05),
    ]
    rep_call = engine.evaluate_margin(collateral_assets=collateral, positions=pos_heavy, cash_balance=10000.0)

    assert 80.0 <= rep_call.margin_usage_pct < 95.0
    assert rep_call.status == MarginStatusLevel.MARGIN_CALL
    assert "MARGIN_CALL_ALERT" in rep_call.action_required

    # 2. Extreme leverage: notional $195,000 at 5% MM = $9,750 (97.5% usage) -> LIQUIDATION_BREACH
    pos_extreme = [
        MarginPosition(symbol="BTC-USD", side="LONG", quantity=3.0, mark_price=65000.0, maintenance_margin_rate=0.05),
    ]
    rep_breach = engine.evaluate_margin(collateral_assets=collateral, positions=pos_extreme, cash_balance=10000.0)

    assert rep_breach.margin_usage_pct >= 95.0
    assert rep_breach.status == MarginStatusLevel.LIQUIDATION_BREACH
    assert "IMMEDIATE_DELEVERAGING" in rep_breach.action_required


def test_hedged_netting_benefit():
    engine = PortfolioMarginEngine(hedged_netting_discount=0.60)

    # 2.0 Long BTC and 1.0 Short BTC
    positions = [
        MarginPosition(symbol="BTC-USD", side="LONG", quantity=2.0, mark_price=60000.0, maintenance_margin_rate=0.05),
        MarginPosition(symbol="BTC-USD", side="SHORT", quantity=1.0, mark_price=60000.0, maintenance_margin_rate=0.05),
    ]

    report = engine.evaluate_margin(collateral_assets=[], positions=positions, cash_balance=50000.0)

    # Gross MM = (120k + 60k) * 0.05 = 9000
    assert report.total_maintenance_margin_usd == 9000.0
    # Hedged notional = 60k, relief = 60k * 0.05 * 0.60 = 1800
    assert report.netting_benefit_usd == 1800.0
    # Net MM = 9000 - 1800 = 7200
    assert report.net_maintenance_margin_usd == 7200.0


def test_subaccount_segregation():
    engine = PortfolioMarginEngine()

    book1 = SubAccountBook(
        subaccount_id="sub_trend_01",
        strategy_name="ALPHA_TREND",
        cash=25000.0,
        allocated_risk_budget_usd=30000.0,
        positions=[
            MarginPosition(symbol="BTC-USD", side="LONG", quantity=0.2, mark_price=65000.0),
        ],
    )
    book2 = SubAccountBook(
        subaccount_id="sub_arb_02",
        strategy_name="STAT_ARB",
        cash=15000.0,
        allocated_risk_budget_usd=20000.0,
        positions=[
            MarginPosition(symbol="ETH-USD", side="LONG", quantity=2.0, mark_price=3000.0),
        ],
    )

    engine.register_subaccount(book1)
    engine.register_subaccount(book2)

    report = engine.evaluate_margin(collateral_assets=[], positions=[], cash_balance=40000.0)

    assert "sub_trend_01" in report.subaccount_summaries
    assert "sub_arb_02" in report.subaccount_summaries
    assert report.subaccount_summaries["sub_trend_01"]["strategy_name"] == "ALPHA_TREND"
    assert report.subaccount_summaries["sub_trend_01"]["positions_count"] == 1
    assert report.subaccount_summaries["sub_arb_02"]["total_notional_usd"] == 6000.0
