"""Tests for Micro-Structural Statistical Arbitrage & High-Frequency Lead-Lag Network Engine."""

import math
import numpy as np
import pytest

from analysis.lead_lag_network import (
    InstrumentCentralityProfile,
    InstrumentNetworkRole,
    LeadLagArbitrageSignal,
    LeadLagNetworkEngine,
    LeadLagNetworkReport,
    LeadLagSignalAction,
    PairwiseLeadLagLink,
)


@pytest.fixture
def synthetic_lead_lag_data():
    """Generate synthetic prices where BTC leads ETH by 2 steps, and ETH leads SOL by 1 step."""
    np.random.seed(42)
    T = 150
    # Base Brownian motion innovations
    eps = np.random.randn(T) * 0.005

    # Leader: BTC follows innovations immediately
    ret_btc = eps.copy()

    # Follower 1: ETH lags BTC by 2 steps
    ret_eth = np.zeros(T)
    ret_eth[2:] = ret_btc[:-2] * 0.8 + np.random.randn(T - 2) * 0.001

    # Follower 2: SOL lags ETH by 1 step (so lags BTC by 3 steps)
    ret_sol = np.zeros(T)
    ret_sol[1:] = ret_eth[:-1] * 0.7 + np.random.randn(T - 1) * 0.001

    # Integrate into prices
    p_btc = 60000.0 * np.exp(np.cumsum(ret_btc))
    p_eth = 3000.0 * np.exp(np.cumsum(ret_eth))
    p_sol = 150.0 * np.exp(np.cumsum(ret_sol))

    return {
        "BTC-USD": p_btc.tolist(),
        "ETH-USD": p_eth.tolist(),
        "SOL-USD": p_sol.tolist(),
    }


def test_lead_lag_network_calibration(synthetic_lead_lag_data):
    """Verify that BTC is identified as MARKET_LEADER with positive lead delay vs ETH and SOL."""
    symbols = ["BTC-USD", "ETH-USD", "SOL-USD"]
    engine = LeadLagNetworkEngine(symbols=symbols, max_lag_steps=4)

    engine.ingest_price_matrix(synthetic_lead_lag_data)
    report = engine.calibrate_lead_lag_network()

    assert isinstance(report, LeadLagNetworkReport)
    assert report.symbols == symbols
    assert len(report.profiles) == 3

    # Check network roles
    profiles_by_sym = {p.symbol: p for p in report.profiles}
    btc_prof = profiles_by_sym["BTC-USD"]
    assert btc_prof.network_role == InstrumentNetworkRole.MARKET_LEADER
    assert btc_prof.outflow_lead_score > btc_prof.inflow_lag_score

    # Verify pairwise link: BTC leads ETH
    btc_eth_links = [
        l for l in report.pairwise_links
        if l.leader_symbol == "BTC-USD" and l.follower_symbol == "ETH-USD"
    ]
    assert len(btc_eth_links) > 0
    link = btc_eth_links[0]
    # Optimal lag should be ~2 steps as injected
    assert link.optimal_lag_steps == 2
    assert link.lead_lag_strength > 0.60
    assert link.leadership_score > 0.0


def test_predictive_arbitrage_signal_generation(synthetic_lead_lag_data):
    """Verify signal emission when leading asset makes a large movement."""
    symbols = ["BTC-USD", "ETH-USD", "SOL-USD"]
    engine = LeadLagNetworkEngine(symbols=symbols, max_lag_steps=4, signal_threshold_bps=5.0)

    engine.ingest_price_matrix(synthetic_lead_lag_data)

    # Inject sharp upward move in BTC: +0.50% (+50 bps)
    last_btc = synthetic_lead_lag_data["BTC-USD"][-1]
    engine.update_price_tick("BTC-USD", last_btc * 1.005)

    report = engine.calibrate_lead_lag_network()
    assert len(report.generated_signals) >= 1

    sig = report.generated_signals[0]
    assert sig.leader_symbol == "BTC-USD"
    assert sig.action == LeadLagSignalAction.LONG_FOLLOWER
    assert sig.predicted_follower_return_bps > 0.0
    assert sig.confidence > 0.50


def test_empty_report_fallback():
    """Verify safe fallback report before sufficient history is buffered."""
    engine = LeadLagNetworkEngine(symbols=["BTC-USD", "ETH-USD"])
    report = engine.calibrate_lead_lag_network()

    assert len(report.profiles) == 2
    assert len(report.pairwise_links) == 0
    assert len(report.generated_signals) == 0
    assert report.matrix_spectral_gap == 0.0


def test_input_validation_and_serialization(synthetic_lead_lag_data):
    """Test validation errors and report serialization."""
    with pytest.raises(ValueError, match="At least 2"):
        LeadLagNetworkEngine(symbols=["ONLY_ONE"])

    symbols = ["BTC-USD", "ETH-USD", "SOL-USD"]
    engine = LeadLagNetworkEngine(symbols=symbols)
    engine.ingest_price_matrix(synthetic_lead_lag_data)
    rep = engine.calibrate_lead_lag_network()

    d = rep.to_dict()
    assert d["symbols"] == symbols
    assert "profiles" in d
    assert "pairwise_links" in d
    assert "top_predictive_pairs" in d
    assert "matrix_spectral_gap" in d
