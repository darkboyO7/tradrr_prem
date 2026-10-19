"""Trading Intelligence Platform - Macro Portfolio Stress Testing Engine.

Simulates extreme historical market crashes, liquidity dry-ups, gap risk, and
counterparty solvency shocks against active portfolios, evaluating circuit breaker
breaches and generating capital preservation recommendations.
"""

from datetime import datetime, timezone
from enum import Enum
from typing import Dict, List, Optional
from pydantic import BaseModel, Field

from core.logging import get_logger
from risk.limits import RiskLimits

logger = get_logger("stress_testing")


class StressScenarioType(str, Enum):
    """Catalog of canonical institutional stress test scenarios."""

    COVID_LIQUIDITY_CRUNCH_2020 = "COVID_LIQUIDITY_CRUNCH_2020"    # Black Thursday March 2020 cascade
    FTX_SOLVENCY_CASCADE_2022 = "FTX_SOLVENCY_CASCADE_2022"        # November 2022 contagion
    FLASH_CRASH_GAP_RISK = "FLASH_CRASH_GAP_RISK"                  # Instantaneous 20% gap-down with slippage
    MACRO_INTEREST_RATE_HIKE = "MACRO_INTEREST_RATE_HIKE"          # Broad risk-off multi-asset liquidation
    LIQUIDITY_VACUUM = "LIQUIDITY_VACUUM"                          # Top-of-book bid evaporation and 8x spread blow-out


class ScenarioDefinition(BaseModel):
    """Specification of a stress scenario shock parameters."""

    name: str
    description: str
    default_asset_shock_pct: float
    asset_specific_shocks_pct: Dict[str, float] = Field(default_factory=dict)
    volatility_expansion_multiplier: float = 1.0
    spread_widening_multiplier: float = 1.0
    slippage_multiplier: float = 1.0
    correlation_surge_level: float = 0.90


class AssetStressImpact(BaseModel):
    """Impact of scenario shock on a single portfolio position."""

    symbol: str
    initial_allocation_usd: float
    shock_pct: float
    loss_usd: float
    post_shock_value_usd: float
    stop_loss_triggered: bool = False
    effective_slippage_usd: float = 0.0


class StressScenarioReport(BaseModel):
    """Comprehensive stress test outcome report for a specific scenario."""

    scenario: StressScenarioType
    scenario_name: str
    initial_equity: float
    post_shock_equity: float
    total_dollar_loss: float
    portfolio_drawdown_pct: float
    daily_drawdown_limit_breached: bool
    max_drawdown_limit_breached: bool
    margin_call_triggered: bool
    asset_impacts: Dict[str, AssetStressImpact]
    recommended_mitigations: List[str]


class PortfolioStressTester:
    """Stress testing engine applying non-linear macro shocks to portfolio assets."""

    SCENARIO_CATALOG: Dict[StressScenarioType, ScenarioDefinition] = {
        StressScenarioType.COVID_LIQUIDITY_CRUNCH_2020: ScenarioDefinition(
            name="March 2020 Black Thursday Liquidity Cascade",
            description="Broad systemic panic with BTC -48%, ETH -52%, altcoins -65%, correlations near 1.0.",
            default_asset_shock_pct=-0.65,
            asset_specific_shocks_pct={
                "BTC-USD": -0.48,
                "ETH-USD": -0.52,
                "SOL-USD": -0.65,
            },
            volatility_expansion_multiplier=3.5,
            spread_widening_multiplier=4.0,
            slippage_multiplier=3.0,
            correlation_surge_level=0.98,
        ),
        StressScenarioType.FTX_SOLVENCY_CASCADE_2022: ScenarioDefinition(
            name="November 2022 FTX Counterparty Contagion",
            description="Solvency run and contagion leading to SOL -62%, BTC -25%, ETH -30%, exchange runs.",
            default_asset_shock_pct=-0.45,
            asset_specific_shocks_pct={
                "SOL-USD": -0.62,
                "BTC-USD": -0.25,
                "ETH-USD": -0.30,
            },
            volatility_expansion_multiplier=2.5,
            spread_widening_multiplier=5.0,
            slippage_multiplier=2.5,
            correlation_surge_level=0.85,
        ),
        StressScenarioType.FLASH_CRASH_GAP_RISK: ScenarioDefinition(
            name="Instantaneous Flash Gap-Down (20%)",
            description="Flash order book wipeout gapping down 20% past stop-losses with heavy execution slippage.",
            default_asset_shock_pct=-0.20,
            asset_specific_shocks_pct={},
            volatility_expansion_multiplier=2.0,
            spread_widening_multiplier=3.0,
            slippage_multiplier=2.0,
            correlation_surge_level=0.90,
        ),
        StressScenarioType.MACRO_INTEREST_RATE_HIKE: ScenarioDefinition(
            name="Macro Federal Reserve Interest Rate Shock",
            description="Unexpected 75bps rate hike triggering institutional derisking and crypto outflow.",
            default_asset_shock_pct=-0.25,
            asset_specific_shocks_pct={
                "BTC-USD": -0.18,
                "ETH-USD": -0.22,
            },
            volatility_expansion_multiplier=1.8,
            spread_widening_multiplier=1.5,
            slippage_multiplier=1.2,
            correlation_surge_level=0.88,
        ),
        StressScenarioType.LIQUIDITY_VACUUM: ScenarioDefinition(
            name="Extreme Liquidity Depth Evaporation",
            description="Market makers pull quotes, spreads widen 8x, market impact slippage surges 4x.",
            default_asset_shock_pct=-0.15,
            asset_specific_shocks_pct={},
            volatility_expansion_multiplier=2.2,
            spread_widening_multiplier=8.0,
            slippage_multiplier=4.0,
            correlation_surge_level=0.75,
        ),
    }

    def __init__(self, limits: Optional[RiskLimits] = None) -> None:
        self.limits = limits or RiskLimits()

    def run_scenario(
        self,
        scenario_type: StressScenarioType,
        portfolio_equity: float,
        allocations_usd: Dict[str, float],
        stop_losses: Optional[Dict[str, float]] = None,
        current_prices: Optional[Dict[str, float]] = None,
    ) -> StressScenarioReport:
        """Simulate a specific stress scenario shock against portfolio allocations."""
        scenario = self.SCENARIO_CATALOG.get(scenario_type)
        if not scenario:
            raise ValueError(f"Unknown scenario type: {scenario_type}")

        asset_impacts: Dict[str, AssetStressImpact] = {}
        total_dollar_loss = 0.0

        for symbol, alloc_usd in allocations_usd.items():
            sym_upper = symbol.upper()
            raw_shock = scenario.asset_specific_shocks_pct.get(sym_upper, scenario.default_asset_shock_pct)

            # Check if stop loss would trigger and cap the loss
            sl_triggered = False
            effective_shock = raw_shock
            slippage_usd = 0.0

            if stop_losses and current_prices and sym_upper in stop_losses and sym_upper in current_prices:
                sl_price = stop_losses[sym_upper]
                curr_price = current_prices[sym_upper]
                if curr_price > 0 and sl_price < curr_price:
                    # Intended SL percentage loss
                    sl_loss_pct = (sl_price - curr_price) / curr_price  # negative
                    if raw_shock < sl_loss_pct:
                        sl_triggered = True
                        # Apply slippage multiplier to gap past SL
                        additional_slip_pct = abs(sl_loss_pct) * 0.20 * scenario.slippage_multiplier
                        effective_shock = sl_loss_pct - additional_slip_pct
                        slippage_usd = round(alloc_usd * additional_slip_pct, 2)

            loss_usd = round(alloc_usd * abs(effective_shock), 2)
            post_shock_val = max(0.0, round(alloc_usd - loss_usd, 2))
            total_dollar_loss += loss_usd

            asset_impacts[sym_upper] = AssetStressImpact(
                symbol=sym_upper,
                initial_allocation_usd=alloc_usd,
                shock_pct=round(effective_shock * 100.0, 2),
                loss_usd=loss_usd,
                post_shock_value_usd=post_shock_val,
                stop_loss_triggered=sl_triggered,
                effective_slippage_usd=slippage_usd,
            )

        post_shock_equity = max(0.0, round(portfolio_equity - total_dollar_loss, 2))
        drawdown_pct = round((total_dollar_loss / portfolio_equity) * 100.0, 2) if portfolio_equity > 0 else 0.0

        # Evaluate risk limit breaches
        daily_breached = (drawdown_pct / 100.0) >= self.limits.daily_drawdown_limit_pct
        max_dd_breached = (drawdown_pct / 100.0) >= self.limits.max_drawdown_limit_pct
        margin_call = post_shock_equity <= (portfolio_equity * 0.20)

        # Generate institutional mitigation directives
        mitigations: List[str] = []
        if margin_call:
            mitigations.append("CRITICAL: Severe capital depletion risk (< 20% equity). Immediately liquidate 50% gross exposure.")
        if max_dd_breached:
            mitigations.append(f"CIRCUIT BREAKER: Max drawdown limit ({self.limits.max_drawdown_limit_pct*100:.1f}%) breached. Halt new entries.")
        if daily_breached:
            mitigations.append(f"DAILY LIMIT BREACH: Loss exceeds daily ceiling ({self.limits.daily_drawdown_limit_pct*100:.1f}%). De-risk portfolio.")

        # Asset concentration checks
        for sym, impact in asset_impacts.items():
            if impact.initial_allocation_usd / portfolio_equity > self.limits.max_single_position_pct:
                mitigations.append(
                    f"Concentration risk: {sym} constitutes > {self.limits.max_single_position_pct*100:.1f}% of capital. Trim allocation."
                )

        if not mitigations:
            mitigations.append("Portfolio parameters resilient to scenario. Maintain current risk limits.")

        return StressScenarioReport(
            scenario=scenario_type,
            scenario_name=scenario.name,
            initial_equity=portfolio_equity,
            post_shock_equity=post_shock_equity,
            total_dollar_loss=round(total_dollar_loss, 2),
            portfolio_drawdown_pct=drawdown_pct,
            daily_drawdown_limit_breached=daily_breached,
            max_drawdown_limit_breached=max_dd_breached,
            margin_call_triggered=margin_call,
            asset_impacts=asset_impacts,
            recommended_mitigations=mitigations,
        )

    def run_all_scenarios(
        self,
        portfolio_equity: float,
        allocations_usd: Dict[str, float],
        stop_losses: Optional[Dict[str, float]] = None,
        current_prices: Optional[Dict[str, float]] = None,
    ) -> List[StressScenarioReport]:
        """Execute all configured macro scenarios and return comprehensive stress suite."""
        reports = []
        for scen_type in self.SCENARIO_CATALOG.keys():
            rep = self.run_scenario(
                scenario_type=scen_type,
                portfolio_equity=portfolio_equity,
                allocations_usd=allocations_usd,
                stop_losses=stop_losses,
                current_prices=current_prices,
            )
            reports.append(rep)
        return reports
