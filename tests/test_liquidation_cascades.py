"""Tests for Continuous Dynamic Margin Health & Multi-Tier Liquidation Cascades Engine."""

import pytest

from risk.liquidation_cascades import (
    DynamicCollateralAsset,
    DynamicMarginHealthEngine,
    LiquidationLadderPlan,
    LiquidationStage,
    SolvencyState,
    SolvencyStressSimulationResult,
    TieredLiquidationOrder,
)


@pytest.fixture
def sample_portfolio():
    """Setup portfolio with 2 collateral assets and 2 leveraged positions."""
    collateral = [
        DynamicCollateralAsset(
            asset="BTC",
            quantity=2.0,
            base_mark_price=60000.0,
            base_haircut=0.10,
            volatility_haircut_sensitivity=1.0,
            concentration_haircut_penalty=0.5,
        ),
        DynamicCollateralAsset(
            asset="USDC",
            quantity=50000.0,
            base_mark_price=1.0,
            base_haircut=0.01,
            volatility_haircut_sensitivity=0.1,
            concentration_haircut_penalty=0.2,
        ),
    ]

    positions = [
        {
            "symbol": "BTC-USD",
            "side": "LONG",
            "quantity": 10.0,
            "mark_price": 60000.0,
            "entry_price": 60000.0,
            "mm_rate": 0.05,  # $30,000 MM
        },
        {
            "symbol": "ETH-USD",
            "side": "LONG",
            "quantity": 50.0,
            "mark_price": 3000.0,
            "entry_price": 3000.0,
            "mm_rate": 0.05,  # $7,500 MM
        },
    ]

    return collateral, positions


def test_optimal_solvency_state(sample_portfolio):
    """Test healthy portfolio evaluation with no liquidation required."""
    collateral, positions = sample_portfolio
    engine = DynamicMarginHealthEngine(target_healthy_usage_pct=0.70)

    # Initial equity is large (~$160k), MM is $37.5k -> usage ~23% -> OPTIMAL
    plan, state = engine.evaluate_solvency_and_cascade(
        collateral_assets=collateral,
        positions=positions,
    )

    assert state == SolvencyState.OPTIMAL
    assert plan.solvency_state == SolvencyState.OPTIMAL
    assert plan.current_margin_usage_pct < 0.50
    assert len(plan.liquidation_orders) == 0
    assert plan.required_margin_reduction_usd == 0.0
    assert plan.recommended_capital_injection_usd == 0.0


def test_margin_deficit_and_partial_unwind():
    """Test margin deficit triggering Stage 3 partial unwinding."""
    engine = DynamicMarginHealthEngine(target_healthy_usage_pct=0.70)

    # Low collateral ($25,000), large position needing $20,000 MM -> Usage = 20k / 25k = 80% (MARGIN_DEFICIT)
    collateral = [
        DynamicCollateralAsset(
            asset="USD",
            quantity=25000.0,
            base_mark_price=1.0,
            base_haircut=0.0,
        )
    ]
    positions = [
        {
            "symbol": "BTC-USD",
            "side": "LONG",
            "quantity": 5.0,
            "mark_price": 60000.0,
            "entry_price": 60000.0,
            "mm_rate": 0.0667,  # MM = 5 * 60k * 0.0667 = ~$20,000
        }
    ]

    plan, state = engine.evaluate_solvency_and_cascade(
        collateral_assets=collateral,
        positions=positions,
    )

    assert state == SolvencyState.MARGIN_DEFICIT
    assert plan.current_margin_usage_pct >= 0.75
    assert len(plan.liquidation_orders) > 0

    order = plan.liquidation_orders[0]
    assert order.stage == LiquidationStage.STAGE_3_PARTIAL_UNWIND
    assert order.symbol == "BTC-USD"
    assert order.side == "SELL"
    assert order.quantity == pytest.approx(1.5, rel=1e-3)  # 30% of 5.0
    assert plan.total_margin_to_be_freed_usd > 0.0
    assert plan.projected_post_action_margin_usage_pct < plan.current_margin_usage_pct


def test_insolvent_default_and_full_closeout():
    """Test equity below maintenance margin triggering Stage 5 full closeout."""
    engine = DynamicMarginHealthEngine()

    collateral = [
        DynamicCollateralAsset(
            asset="USD",
            quantity=10000.0,
            base_mark_price=1.0,
            base_haircut=0.0,
        )
    ]
    positions = [
        {
            "symbol": "BTC-USD",
            "side": "LONG",
            "quantity": 5.0,
            "mark_price": 50000.0,
            "entry_price": 52000.0,  # -$10,000 unrealized PnL -> equity depleted!
            "mm_rate": 0.05,        # MM = $12,500
        }
    ]

    plan, state = engine.evaluate_solvency_and_cascade(
        collateral_assets=collateral,
        positions=positions,
    )

    assert state == SolvencyState.INSOLVENT_DEFAULT
    assert len(plan.liquidation_orders) == 1
    order = plan.liquidation_orders[0]
    assert order.stage == LiquidationStage.STAGE_5_FULL_CLOSEOUT
    assert order.quantity == 5.0  # 100% unwind
    assert plan.insurance_fund_draw_required_usd > 0.0


def test_solvency_stress_simulation(sample_portfolio):
    """Test flash crash stress test simulation."""
    collateral, positions = sample_portfolio
    engine = DynamicMarginHealthEngine()

    res = engine.simulate_solvency_stress(
        collateral_assets=collateral,
        positions=positions,
        scenario_name="Crypto Black Swan 40%",
        price_shock_pct=-0.40,
        volatility_shock_multiplier=3.0,
    )

    assert isinstance(res, SolvencyStressSimulationResult)
    assert res.price_shock_pct == -0.40
    assert res.post_shock_equity < res.pre_shock_equity
    assert res.breach_stage_reached in (
        LiquidationStage.STAGE_3_PARTIAL_UNWIND,
        LiquidationStage.STAGE_4_AGGRESSIVE_DELEVERAGE,
        LiquidationStage.STAGE_5_FULL_CLOSEOUT,
    )


def test_dynamic_haircut_sensitivity():
    """Test non-linear haircut increase with volatility and concentration."""
    col = DynamicCollateralAsset(
        asset="BTC",
        quantity=10.0,
        base_mark_price=50000.0,
        base_haircut=0.05,
        volatility_haircut_sensitivity=1.5,
        concentration_haircut_penalty=0.8,
    )

    # Base low stress, low concentration
    h_low = col.calculate_effective_haircut(
        asset_volatility=0.10,
        portfolio_concentration_pct=0.20,
        market_stress_multiplier=1.0,
    )
    assert h_low == pytest.approx(0.05 + 1.5 * 0.10, rel=1e-4)

    # Extreme volatility and high concentration (75%)
    h_extreme = col.calculate_effective_haircut(
        asset_volatility=0.40,
        portfolio_concentration_pct=0.75,
        market_stress_multiplier=2.0,
    )
    # Expected: 0.05 + 1.5 * 0.4 * 2.0 + 0.8 * (0.75 - 0.25) = 0.05 + 1.20 + 0.40 = 1.65 -> clipped to 0.99
    assert h_extreme == 0.99
    assert col.get_collateral_value(50000.0, h_extreme) == pytest.approx(500000.0 * 0.01)
