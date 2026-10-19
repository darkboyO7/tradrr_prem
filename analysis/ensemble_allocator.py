"""Trading Intelligence Platform - Multi-Armed Bandit Alpha Ensemble & Dynamic Strategy Allocator.

Implements institutional algorithmic capital allocation across distinct trading strategies
(Momentum, Statistical Arbitrage, Market Making, Basis Carry, Volatility Harvesting) using
Multi-Armed Bandit algorithms (UCB1, Thompson Sampling, Exp3) conditioned on real-time
macro/micro market regimes.
"""

from datetime import datetime, timezone
from enum import Enum
import math
from typing import Any, Dict, List, Optional
import uuid
import numpy as np
from pydantic import BaseModel, Field

from core.logging import get_logger

logger = get_logger("ensemble_allocator")


class BanditAlgorithmType(str, Enum):
    """Bandit algorithm implementations for strategy capital allocation."""

    UCB1 = "UCB1"
    THOMPSON_SAMPLING = "THOMPSON_SAMPLING"
    EXP3 = "EXP3"
    REGIME_AWARE_HYBRID = "REGIME_AWARE_HYBRID"


class StrategyArmState(BaseModel):
    """Execution state, reward history, and statistical diagnostics for a strategy arm."""

    strategy_id: str
    strategy_name: str
    pull_count: int = 0
    cumulative_reward: float = 0.0
    mean_reward: float = 0.0
    variance_reward: float = 0.0
    alpha_param: float = 1.0  # Thompson Beta prior success
    beta_param: float = 1.0   # Thompson Beta prior failure
    exp3_weight: float = 1.0  # Exp3 unnormalized weight
    last_reward: float = 0.0
    current_weight: float = 0.20
    allocated_capital_usd: float = 0.0


class AllocationDecision(BaseModel):
    """Institutional portfolio capital allocation across strategy arms."""

    allocation_id: str = Field(default_factory=lambda: f"alloc_{uuid.uuid4().hex[:8]}")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    algorithm_used: BanditAlgorithmType
    market_regime: Optional[str] = None
    total_capital_usd: float
    strategy_allocations: Dict[str, float]
    strategy_capital_usd: Dict[str, float]
    exploration_share_pct: float
    strategy_diagnostics: Dict[str, Dict[str, Any]] = {}


class RewardUpdateResult(BaseModel):
    """Feedback telemetry following an arm reward update."""

    strategy_id: str
    raw_pnl_usd: float
    normalized_reward: float
    updated_pull_count: int
    updated_mean_reward: float
    updated_weight: float


class MultiArmedBanditAllocator:
    """Multi-Armed Bandit engine dynamically optimizing capital allocation across trading strategies."""

    DEFAULT_STRATEGIES = {
        "MOMENTUM_TREND": "Cross-Asset Momentum & Breakout Strategy",
        "STAT_ARB_PAIRS": "Cointegrated Pairs Statistical Arbitrage",
        "MARKET_MAKING_DMM": "Avellaneda-Stoikov Dynamic Market Maker",
        "BASIS_CARRY": "Perpetual Funding Rate & Basis Arbitrage",
        "VOLATILITY_HARVEST": "Delta-Neutral Options Volatility Harvesting",
    }

    # Regime affinity multipliers: boost or taper baseline bandit scores based on macro conditions
    REGIME_AFFINITY_MATRIX: Dict[str, Dict[str, float]] = {
        "BULL_TRENDING": {
            "MOMENTUM_TREND": 1.50,
            "STAT_ARB_PAIRS": 0.85,
            "MARKET_MAKING_DMM": 1.05,
            "BASIS_CARRY": 1.30,
            "VOLATILITY_HARVEST": 0.90,
        },
        "BEAR_TRENDING": {
            "MOMENTUM_TREND": 1.40,
            "STAT_ARB_PAIRS": 0.90,
            "MARKET_MAKING_DMM": 0.70,
            "BASIS_CARRY": 0.80,
            "VOLATILITY_HARVEST": 0.75,
        },
        "HIGH_VOLATILITY": {
            "MOMENTUM_TREND": 1.15,
            "STAT_ARB_PAIRS": 0.60,
            "MARKET_MAKING_DMM": 0.40,
            "BASIS_CARRY": 0.70,
            "VOLATILITY_HARVEST": 1.45,
        },
        "LOW_VOLATILITY": {
            "MOMENTUM_TREND": 0.65,
            "STAT_ARB_PAIRS": 1.30,
            "MARKET_MAKING_DMM": 1.60,
            "BASIS_CARRY": 1.40,
            "VOLATILITY_HARVEST": 1.10,
        },
        "MEAN_REVERTING": {
            "MOMENTUM_TREND": 0.60,
            "STAT_ARB_PAIRS": 1.70,
            "MARKET_MAKING_DMM": 1.35,
            "BASIS_CARRY": 1.00,
            "VOLATILITY_HARVEST": 1.10,
        },
        "NEUTRAL": {
            "MOMENTUM_TREND": 1.00,
            "STAT_ARB_PAIRS": 1.00,
            "MARKET_MAKING_DMM": 1.00,
            "BASIS_CARRY": 1.00,
            "VOLATILITY_HARVEST": 1.00,
        },
    }

    def __init__(
        self,
        strategies: Optional[Dict[str, str]] = None,
        algorithm: BanditAlgorithmType = BanditAlgorithmType.REGIME_AWARE_HYBRID,
        min_weight: float = 0.05,
        max_weight: float = 0.45,
        exploration_factor: float = 1.0,
        learning_rate: float = 0.15,
        random_seed: Optional[int] = 42,
    ) -> None:
        self.algorithm = algorithm
        self.min_weight = min_weight
        self.max_weight = max_weight
        self.exploration_factor = exploration_factor
        self.learning_rate = learning_rate
        self.rng = np.random.default_rng(random_seed)

        strat_dict = strategies or self.DEFAULT_STRATEGIES
        n_strats = len(strat_dict)
        init_weight = 1.0 / max(1, n_strats)

        self.arms: Dict[str, StrategyArmState] = {}
        for sid, name in strat_dict.items():
            self.arms[sid] = StrategyArmState(
                strategy_id=sid,
                strategy_name=name,
                current_weight=init_weight,
            )

        self.total_rounds: int = 0
        self.history: List[AllocationDecision] = []

    def _normalize_and_bound_weights(self, raw_weights: Dict[str, float]) -> Dict[str, float]:
        """Iterative water-filling algorithm to strictly satisfy min_weight and max_weight bounds."""
        keys = list(raw_weights.keys())
        n = len(keys)
        if n == 0:
            return {}
        if n * self.min_weight > 1.0:
            return {k: 1.0 / n for k in keys}

        weights = {k: max(0.0001, raw_weights[k]) for k in keys}
        total = sum(weights.values())
        weights = {k: v / total for k, v in weights.items()}

        # Iterative clamping
        for _ in range(20):
            clamped = False
            for k in keys:
                if weights[k] < self.min_weight:
                    weights[k] = self.min_weight
                    clamped = True
                elif weights[k] > self.max_weight:
                    weights[k] = self.max_weight
                    clamped = True

            total = sum(weights.values())
            if total > 0 and abs(total - 1.0) > 1e-5:
                weights = {k: v / total for k, v in weights.items()}
            elif not clamped:
                break

        # Final normalization to ensure exact sum = 1.0
        final_total = sum(weights.values())
        norm_dict = {k: v / final_total for k, v in weights.items()}
        rounded = {k: round(v, 4) for k, v in norm_dict.items()}
        diff = round(1.0 - sum(rounded.values()), 4)
        if diff != 0:
            max_k = max(rounded, key=rounded.get)
            rounded[max_k] = round(rounded[max_k] + diff, 4)
        return rounded


    def _compute_ucb1_scores(self) -> Dict[str, float]:
        """Compute Upper Confidence Bound (UCB1) for all arms."""
        scores = {}
        t = max(1, self.total_rounds)
        for sid, arm in self.arms.items():
            if arm.pull_count == 0:
                scores[sid] = 1e6  # Force initial trial of untried arm
            else:
                exploration = self.exploration_factor * math.sqrt((2.0 * math.log(t)) / arm.pull_count)
                scores[sid] = arm.mean_reward + exploration
        return scores

    def _compute_thompson_samples(self) -> Dict[str, float]:
        """Draw samples from Bayesian Beta distributions for each arm."""
        samples = {}
        for sid, arm in self.arms.items():
            a = max(0.1, arm.alpha_param)
            b = max(0.1, arm.beta_param)
            sample = self.rng.beta(a, b)
            samples[sid] = float(sample)
        return samples

    def _compute_exp3_weights(self) -> Dict[str, float]:
        """Compute Exp3 exponential weights with uniform exploration mixture."""
        k = len(self.arms)
        gamma = max(0.01, min(0.5, self.learning_rate))
        total_w = sum(arm.exp3_weight for arm in self.arms.values())
        probs = {}
        for sid, arm in self.arms.items():
            p = (1.0 - gamma) * (arm.exp3_weight / max(1e-8, total_w)) + (gamma / k)
            probs[sid] = p
        return probs

    def allocate_capital(
        self,
        total_capital_usd: float = 100000.0,
        current_regime: Optional[str] = None,
    ) -> AllocationDecision:
        """Synthesize dynamic capital allocation weights across strategies."""
        self.total_rounds += 1
        regime_str = (current_regime or "NEUTRAL").upper()
        affinities = self.REGIME_AFFINITY_MATRIX.get(regime_str, self.REGIME_AFFINITY_MATRIX["NEUTRAL"])

        # 1. Compute raw scores based on selected bandit algorithm
        if self.algorithm == BanditAlgorithmType.UCB1:
            raw_scores = self._compute_ucb1_scores()
        elif self.algorithm == BanditAlgorithmType.THOMPSON_SAMPLING:
            raw_scores = self._compute_thompson_samples()
        elif self.algorithm == BanditAlgorithmType.EXP3:
            raw_scores = self._compute_exp3_weights()
        else:  # REGIME_AWARE_HYBRID
            ucb = self._compute_ucb1_scores()
            ts = self._compute_thompson_samples()
            raw_scores = {}
            for sid in self.arms:
                u_val = min(10.0, ucb[sid])
                t_val = ts[sid]
                aff = affinities.get(sid, 1.0)
                # Hybrid combines Thompson sample and UCB score, scaled by regime affinity
                raw_scores[sid] = (0.5 * u_val + 0.5 * t_val) * aff

        # Apply regime multiplier if pure algorithm used
        if self.algorithm != BanditAlgorithmType.REGIME_AWARE_HYBRID and regime_str != "NEUTRAL":
            raw_scores = {sid: score * affinities.get(sid, 1.0) for sid, score in raw_scores.items()}

        # 2. Normalize and enforce institutional risk bounds
        weights = self._normalize_and_bound_weights(raw_scores)

        # 3. Update arm states and compute dollar allocations
        capital_usd = {}
        diagnostics = {}
        for sid, w in weights.items():
            arm = self.arms[sid]
            arm.current_weight = w
            dollar_alloc = round(total_capital_usd * w, 2)
            arm.allocated_capital_usd = dollar_alloc
            capital_usd[sid] = dollar_alloc
            diagnostics[sid] = {
                "pull_count": arm.pull_count,
                "mean_reward": round(arm.mean_reward, 4),
                "alpha": round(arm.alpha_param, 2),
                "beta": round(arm.beta_param, 2),
                "raw_score": round(raw_scores.get(sid, 0.0), 4),
                "regime_affinity": affinities.get(sid, 1.0),
            }

        # Guarantee exact dollar allocation sum matching total_capital_usd
        tot_alloc = sum(capital_usd.values())
        diff_cents = round(total_capital_usd - tot_alloc, 2)
        if diff_cents != 0 and capital_usd:
            max_sid = max(capital_usd, key=capital_usd.get)
            capital_usd[max_sid] = round(capital_usd[max_sid] + diff_cents, 2)
            self.arms[max_sid].allocated_capital_usd = capital_usd[max_sid]


        decision = AllocationDecision(

            algorithm_used=self.algorithm,
            market_regime=regime_str,
            total_capital_usd=total_capital_usd,
            strategy_allocations=weights,
            strategy_capital_usd=capital_usd,
            exploration_share_pct=round(self.min_weight * len(self.arms) * 100.0, 2),
            strategy_diagnostics=diagnostics,
        )
        self.history.append(decision)
        return decision

    def update_reward(
        self,
        strategy_id: str,
        realized_pnl_usd: float,
        notional_traded_usd: float = 10000.0,
        sharpe_equivalent: Optional[float] = None,
    ) -> RewardUpdateResult:
        """Process reward feedback for a strategy following execution."""
        if strategy_id not in self.arms:
            raise KeyError(f"Strategy '{strategy_id}' not found in active arms.")

        arm = self.arms[strategy_id]
        arm.pull_count += 1

        # Calculate normalized return
        if sharpe_equivalent is not None:
            # Scale sharpe (-3 to +3) to [-1, 1] via tanh
            norm_reward = float(np.tanh(sharpe_equivalent / 2.0))
        else:
            base_ret = realized_pnl_usd / max(100.0, notional_traded_usd)
            # Map returns (-5% to +5%) into [-1, 1]
            norm_reward = float(np.tanh(base_ret * 20.0))

        # Update empirical mean and variance (Welford's algorithm)
        delta = norm_reward - arm.mean_reward
        arm.cumulative_reward += norm_reward
        arm.mean_reward += delta / arm.pull_count
        arm.last_reward = norm_reward

        # Update Thompson Beta priors: map [-1, 1] to [0, 1]
        p_success = (norm_reward + 1.0) / 2.0
        arm.alpha_param += p_success
        arm.beta_param += (1.0 - p_success)

        # Update Exp3 weights
        k = len(self.arms)
        eta = self.learning_rate
        prob_est = max(0.001, arm.current_weight)
        estimated_reward = norm_reward / prob_est
        exp_factor = math.exp(min(50.0, (eta * estimated_reward) / k))
        arm.exp3_weight = max(1e-6, arm.exp3_weight * exp_factor)

        logger.info(
            f"Bandit reward updated for {strategy_id}: pnl=${realized_pnl_usd:.2f}, "
            f"norm_reward={norm_reward:.4f}, mean={arm.mean_reward:.4f}, pulls={arm.pull_count}"
        )

        return RewardUpdateResult(
            strategy_id=strategy_id,
            raw_pnl_usd=realized_pnl_usd,
            normalized_reward=round(norm_reward, 4),
            updated_pull_count=arm.pull_count,
            updated_mean_reward=round(arm.mean_reward, 4),
            updated_weight=round(arm.current_weight, 4),
        )

    def get_strategy_rankings(self) -> List[Dict[str, Any]]:
        """Return strategies ranked by empirical mean reward."""
        sorted_arms = sorted(self.arms.values(), key=lambda a: a.mean_reward, reverse=True)
        rankings = []
        for rank, arm in enumerate(sorted_arms, start=1):
            rankings.append({
                "rank": rank,
                "strategy_id": arm.strategy_id,
                "strategy_name": arm.strategy_name,
                "mean_reward": round(arm.mean_reward, 4),
                "pull_count": arm.pull_count,
                "current_weight": round(arm.current_weight, 4),
                "allocated_capital_usd": arm.allocated_capital_usd,
            })
        return rankings

    def reset(self) -> None:
        """Reset all bandit arms to initial prior distributions."""
        n_strats = len(self.arms)
        init_weight = 1.0 / max(1, n_strats)
        self.total_rounds = 0
        self.history.clear()
        for arm in self.arms.values():
            arm.pull_count = 0
            arm.cumulative_reward = 0.0
            arm.mean_reward = 0.0
            arm.alpha_param = 1.0
            arm.beta_param = 1.0
            arm.exp3_weight = 1.0
            arm.last_reward = 0.0
            arm.current_weight = init_weight
            arm.allocated_capital_usd = 0.0
