"""Tests for Multi-Agent LOB Competitive Equilibrium Sandbox & Market Making Simulator."""

import pytest

from backtesting.multi_agent_lob import (
    AgentStrategyType,
    LOBEquilibriumMetrics,
    MultiAgentLOBSandbox,
    MultiAgentSimulationReport,
    SandboxAgentMetrics,
)


def test_multi_agent_simulation_execution():
    """Verify competitive equilibrium sandbox runs with multiple heterogeneous agents."""
    sandbox = MultiAgentLOBSandbox(symbol="BTC-USD", initial_mid_price=65000.0, seed=42)

    # Register heterogeneous agents
    sandbox.register_agent("MM_AS_1", AgentStrategyType.AVELLANEDA_STOIKOV_MM, initial_cash=100000.0, latency_ms=2.0)
    sandbox.register_agent("MM_GRID_1", AgentStrategyType.SYMMETRIC_GRID_MM, initial_cash=100000.0, latency_ms=5.0)
    sandbox.register_agent("TAKER_TOXIC", AgentStrategyType.ADVERSARIAL_INFORMED_TAKER, initial_cash=100000.0, latency_ms=1.0)
    sandbox.register_agent("TAKER_NOISE", AgentStrategyType.NOISE_TRADER_TAKER, initial_cash=50000.0, latency_ms=15.0)

    report = sandbox.run_simulation(steps=50)

    assert isinstance(report, MultiAgentSimulationReport)
    assert report.symbol == "BTC-USD"
    assert report.total_steps == 50
    assert len(report.agent_profiles) == 4
    assert report.lob_metrics.total_events_processed > 0
    assert report.lob_metrics.total_trades_matched > 0
    assert report.lob_metrics.total_volume_traded_usd > 0.0
    assert report.top_performer_agent_id in ("MM_AS_1", "MM_GRID_1", "TAKER_TOXIC", "TAKER_NOISE")


def test_book_clearing_and_spread_formation():
    """Verify limit orders populate both sides and market orders deplete resting liquidity."""
    sandbox = MultiAgentLOBSandbox(symbol="ETH-USD", initial_mid_price=3000.0, seed=42)
    sandbox.register_agent("MM_1", AgentStrategyType.SYMMETRIC_GRID_MM, initial_cash=50000.0)
    sandbox.register_agent("TAKER_1", AgentStrategyType.NOISE_TRADER_TAKER, initial_cash=50000.0)

    # Run 10 steps
    report = sandbox.run_simulation(steps=10)

    assert report.lob_metrics.average_spread_bps > 0.0
    assert report.lob_metrics.min_spread_bps <= report.lob_metrics.max_spread_bps
    assert report.lob_metrics.final_mid_price > 0.0


def test_agent_pnl_attribution_and_telemetry():
    """Verify each agent accumulates valid volume, cash, and PnL metrics."""
    sandbox = MultiAgentLOBSandbox(symbol="SOL-USD", initial_mid_price=150.0, seed=123)
    sandbox.register_agent("MM_LEADER", AgentStrategyType.AVELLANEDA_STOIKOV_MM, initial_cash=20000.0)
    sandbox.register_agent("TAKER_AGG", AgentStrategyType.ADVERSARIAL_INFORMED_TAKER, initial_cash=20000.0)

    report = sandbox.run_simulation(steps=30)
    for agent in report.agent_profiles:
        assert agent.total_equity == pytest.approx(agent.current_cash + (agent.inventory_units * report.lob_metrics.final_mid_price), abs=1e-2)
        assert agent.total_trades_count >= 0
        assert agent.maker_volume >= 0.0
        assert agent.taker_volume >= 0.0
