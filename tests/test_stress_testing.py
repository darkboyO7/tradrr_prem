"""Tests for Macro Portfolio Stress Testing Engine."""

import pytest

from risk.limits import RiskLimits
from risk.stress_testing import (
    AssetStressImpact,
    PortfolioStressTester,
    StressScenarioReport,
    StressScenarioType,
)


def test_stress_test_covid_liquidity_crunch():
    """Verify simulation of 2020 COVID crash with severe drawdowns and circuit breaker trips."""
    tester = PortfolioStressTester()
    portfolio_equity = 100_000.0
    allocations = {
        "BTC-USD": 50_000.0,
        "ETH-USD": 30_000.0,
        "SOL-USD": 20_000.0,
    }

    report = tester.run_scenario(
        scenario_type=StressScenarioType.COVID_LIQUIDITY_CRUNCH_2020,
        portfolio_equity=portfolio_equity,
        allocations_usd=allocations,
    )

    assert isinstance(report, StressScenarioReport)
    assert report.scenario == StressScenarioType.COVID_LIQUIDITY_CRUNCH_2020
    assert report.initial_equity == 100_000.0
    assert report.post_shock_equity < report.initial_equity
    assert report.total_dollar_loss > 40_000.0
    assert report.portfolio_drawdown_pct > 40.0

    # Drawdown limits must be tripped
    assert report.daily_drawdown_limit_breached is True
    assert report.max_drawdown_limit_breached is True
    assert len(report.recommended_mitigations) >= 1

    # Verify asset-level impacts
    assert "BTC-USD" in report.asset_impacts
    assert report.asset_impacts["BTC-USD"].shock_pct == -48.0
    assert "ETH-USD" in report.asset_impacts
    assert report.asset_impacts["ETH-USD"].shock_pct == -52.0


def test_stress_test_stop_loss_capping_and_slippage():
    """Verify stop-losses cap disaster losses while applying scenario slippage widening."""
    tester = PortfolioStressTester()
    portfolio_equity = 100_000.0
    allocations = {"BTC-USD": 50_000.0}

    # BTC price = $60,000, Stop Loss set at $57,000 (5% loss target)
    current_prices = {"BTC-USD": 60_000.0}
    stop_losses = {"BTC-USD": 57_000.0}

    # Scenario shock without SL is -48% ($24,000 loss)
    report_unprotected = tester.run_scenario(
        scenario_type=StressScenarioType.COVID_LIQUIDITY_CRUNCH_2020,
        portfolio_equity=portfolio_equity,
        allocations_usd=allocations,
    )

    # Scenario with SL
    report_protected = tester.run_scenario(
        scenario_type=StressScenarioType.COVID_LIQUIDITY_CRUNCH_2020,
        portfolio_equity=portfolio_equity,
        allocations_usd=allocations,
        stop_losses=stop_losses,
        current_prices=current_prices,
    )

    # Protected loss should be significantly less than 48% (around 5% + slippage)
    assert report_protected.asset_impacts["BTC-USD"].stop_loss_triggered is True
    assert report_protected.total_dollar_loss < report_unprotected.total_dollar_loss
    assert report_protected.total_dollar_loss < 10_000.0  # Capped well below $24,000


def test_stress_test_concentration_risk_mitigation():
    """Verify mitigation recommendations triggered by oversized asset allocation."""
    limits = RiskLimits(max_single_position_pct=0.30)
    tester = PortfolioStressTester(limits=limits)

    # 70% in BTC violates 30% limit
    allocations = {"BTC-USD": 70_000.0, "ETH-USD": 30_000.0}
    report = tester.run_scenario(
        scenario_type=StressScenarioType.FLASH_CRASH_GAP_RISK,
        portfolio_equity=100_000.0,
        allocations_usd=allocations,
    )

    has_concentration_warning = any("Concentration risk: BTC-USD" in m for m in report.recommended_mitigations)
    assert has_concentration_warning is True


def test_stress_test_run_all_scenarios():
    """Verify execution of entire macro stress testing suite."""
    tester = PortfolioStressTester()
    allocations = {"BTC-USD": 50_000.0, "ETH-USD": 30_000.0}

    reports = tester.run_all_scenarios(portfolio_equity=100_000.0, allocations_usd=allocations)
    assert len(reports) == 5
    scenario_types = {r.scenario for r in reports}
    assert StressScenarioType.COVID_LIQUIDITY_CRUNCH_2020 in scenario_types
    assert StressScenarioType.FTX_SOLVENCY_CASCADE_2022 in scenario_types
    assert StressScenarioType.FLASH_CRASH_GAP_RISK in scenario_types
    assert StressScenarioType.MACRO_INTEREST_RATE_HIKE in scenario_types
    assert StressScenarioType.LIQUIDITY_VACUUM in scenario_types
