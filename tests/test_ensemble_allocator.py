"""Tests for Multi-Armed Bandit Alpha Ensemble & Dynamic Strategy Allocator."""

import pytest
from analysis.ensemble_allocator import (
    AllocationDecision,
    BanditAlgorithmType,
    MultiArmedBanditAllocator,
    RewardUpdateResult,
    StrategyArmState,
)


def test_bandit_allocator_initialization_and_weights():
    """Verify bandit allocator initializes arms with equal weights and prior parameters."""
    allocator = MultiArmedBanditAllocator()
    assert len(allocator.arms) == 5
    assert "MOMENTUM_TREND" in allocator.arms
    assert "STAT_ARB_PAIRS" in allocator.arms
    assert "MARKET_MAKING_DMM" in allocator.arms

    for arm in allocator.arms.values():
        assert arm.pull_count == 0
        assert arm.mean_reward == 0.0
        assert arm.alpha_param == 1.0
        assert arm.beta_param == 1.0
        assert abs(arm.current_weight - 0.20) < 1e-4


def test_iterative_water_filling_weight_bounds():
    """Verify water-filling bounds enforcement under extreme score divergence."""
    allocator = MultiArmedBanditAllocator(min_weight=0.05, max_weight=0.40)
    # One strategy massively dominates
    extreme_scores = {
        "MOMENTUM_TREND": 1000.0,
        "STAT_ARB_PAIRS": 0.01,
        "MARKET_MAKING_DMM": 0.01,
        "BASIS_CARRY": 0.01,
        "VOLATILITY_HARVEST": 0.01,
    }
    bounded = allocator._normalize_and_bound_weights(extreme_scores)

    # Max weight must not exceed 0.40
    assert bounded["MOMENTUM_TREND"] <= 0.4001
    # Min weights must not fall below 0.05
    for sid in ["STAT_ARB_PAIRS", "MARKET_MAKING_DMM", "BASIS_CARRY", "VOLATILITY_HARVEST"]:
        assert bounded[sid] >= 0.0499
    # Sum of weights must equal 1.0
    assert abs(sum(bounded.values()) - 1.0) < 1e-3


def test_bandit_reward_update_feedback():
    """Verify reward updates correctly track empirical returns, Beta parameters, and pull counts."""
    allocator = MultiArmedBanditAllocator()

    # Strategy 1 wins
    res1 = allocator.update_reward("MOMENTUM_TREND", realized_pnl_usd=500.0, notional_traded_usd=10000.0)
    assert res1.strategy_id == "MOMENTUM_TREND"
    assert res1.updated_pull_count == 1
    assert res1.normalized_reward > 0.0
    assert res1.updated_mean_reward > 0.0
    assert allocator.arms["MOMENTUM_TREND"].alpha_param > 1.0

    # Strategy 2 loses
    res2 = allocator.update_reward("STAT_ARB_PAIRS", realized_pnl_usd=-400.0, notional_traded_usd=10000.0)
    assert res2.strategy_id == "STAT_ARB_PAIRS"
    assert res2.updated_pull_count == 1
    assert res2.normalized_reward < 0.0
    assert res2.updated_mean_reward < 0.0
    assert allocator.arms["STAT_ARB_PAIRS"].beta_param > 1.0


def test_regime_aware_hybrid_allocation():
    """Verify regime affinity multipliers adapt strategy capital to macro regimes."""
    allocator = MultiArmedBanditAllocator(algorithm=BanditAlgorithmType.REGIME_AWARE_HYBRID)

    # 1. Bull trending regime
    alloc_bull = allocator.allocate_capital(total_capital_usd=100000.0, current_regime="BULL_TRENDING")
    assert alloc_bull.market_regime == "BULL_TRENDING"
    assert alloc_bull.total_capital_usd == 100000.0
    assert abs(sum(alloc_bull.strategy_allocations.values()) - 1.0) < 1e-3
    bull_momentum = alloc_bull.strategy_allocations["MOMENTUM_TREND"]

    # 2. Mean reverting regime
    alloc_mr = allocator.allocate_capital(total_capital_usd=100000.0, current_regime="MEAN_REVERTING")
    assert alloc_mr.market_regime == "MEAN_REVERTING"
    mr_stat_arb = alloc_mr.strategy_allocations["STAT_ARB_PAIRS"]
    mr_momentum = alloc_mr.strategy_allocations["MOMENTUM_TREND"]

    # In bull trending, momentum allocation should exceed its allocation in mean reverting
    assert bull_momentum > mr_momentum
    # In mean reverting, stat arb should exceed momentum
    assert mr_stat_arb > mr_momentum


def test_bandit_algorithms_ucb1_thompson_exp3():
    """Verify each bandit algorithm variant produces valid, bounded allocations."""
    for algo in [BanditAlgorithmType.UCB1, BanditAlgorithmType.THOMPSON_SAMPLING, BanditAlgorithmType.EXP3]:
        allocator = MultiArmedBanditAllocator(algorithm=algo, random_seed=42)
        # Give some initial history
        allocator.update_reward("MOMENTUM_TREND", realized_pnl_usd=250.0)
        allocator.update_reward("BASIS_CARRY", realized_pnl_usd=150.0)

        decision = allocator.allocate_capital(total_capital_usd=50000.0)
        assert decision.algorithm_used == algo
        assert abs(sum(decision.strategy_allocations.values()) - 1.0) < 1e-3
        assert abs(sum(decision.strategy_capital_usd.values()) - 50000.0) < 1.0
        assert decision.exploration_share_pct > 0.0


def test_strategy_rankings_and_reset():
    """Verify ranking orders by mean reward and reset returns arms to initial priors."""
    allocator = MultiArmedBanditAllocator()
    allocator.update_reward("MOMENTUM_TREND", realized_pnl_usd=800.0)
    allocator.update_reward("BASIS_CARRY", realized_pnl_usd=300.0)
    allocator.update_reward("STAT_ARB_PAIRS", realized_pnl_usd=-500.0)

    rankings = allocator.get_strategy_rankings()
    assert len(rankings) == 5
    assert rankings[0]["strategy_id"] == "MOMENTUM_TREND"
    assert rankings[1]["strategy_id"] == "BASIS_CARRY"
    assert rankings[-1]["strategy_id"] == "STAT_ARB_PAIRS"

    # Reset
    allocator.reset()
    for arm in allocator.arms.values():
        assert arm.pull_count == 0
        assert arm.mean_reward == 0.0
        assert arm.alpha_param == 1.0
