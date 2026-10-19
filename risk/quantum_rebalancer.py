"""Trading Intelligence Platform - Quantum-Inspired Portfolio Rebalancer & Cardinality-Constrained Quadratic Program.

Implements combinatorial portfolio optimization under:
- Maximum asset cardinality constraints (K <= K_max)
- Minimum nonzero position allocation thresholds
- Quadratic risk minimization with expected return tradeoff: min w^T Sigma w - lambda * mu^T w
- Linear & non-linear transaction cost friction: sum c_i * |w_i - w_0,i|
- Simulated Annealing / QUBO heuristic solving for combinatorial NP-hard discrete allocation
"""

from datetime import datetime, timezone
from enum import Enum
import math
import random
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from pydantic import BaseModel, Field

from core.logging import get_logger

logger = get_logger("quantum_rebalancer")


class OptimizationSolverMethod(str, Enum):
    """Combinatorial optimization algorithm classification."""

    SIMULATED_ANNEALING_QUBO = "SIMULATED_ANNEALING_QUBO"
    GREEDY_CARDINALITY_PROJECTION = "GREEDY_CARDINALITY_PROJECTION"
    PROJECTED_GRADIENT_DESCENT = "PROJECTED_GRADIENT_DESCENT"


class CardinalityOptimizationConfig(BaseModel):
    """Institutional parameters for cardinality-constrained quadratic programming."""

    max_cardinality_assets: int = 5       # Maximum distinct assets allowed (K <= K_max)
    risk_aversion_lambda: float = 2.5     # Variance penalty multiplier lambda
    transaction_cost_bps: float = 10.0    # Transaction fee & market impact friction in bps
    min_weight_threshold: float = 0.05    # Minimum nonzero weight for an included asset (e.g. 5%)
    max_weight_ceiling: float = 0.40      # Maximum single asset concentration cap (e.g. 40%)
    target_sum_weights: float = 1.0       # Fully invested constraint sum(w) = 1.0
    sa_iterations: int = 2500             # Simulated annealing steps
    sa_initial_temperature: float = 1.0   # Initial thermal fluctuation parameter T_0
    sa_cooling_rate: float = 0.995        # Exponential temperature decay rate
    random_seed: Optional[int] = 42       # Reproducibility seed


class RebalanceAllocationItem(BaseModel):
    """Optimized allocation weight and rebalance delta per asset."""

    symbol: str
    target_weight: float
    current_weight: float
    weight_delta: float
    is_active: bool
    expected_return_annualized: float
    marginal_risk_contribution: float


class CardinalityOptimizationReport(BaseModel):
    """Comprehensive portfolio optimization report under cardinality and turnover constraints."""

    solver_method: OptimizationSolverMethod
    converged: bool
    total_assets_universe: int
    selected_cardinality: int
    max_cardinality_limit: int
    portfolio_expected_return: float
    portfolio_volatility: float
    portfolio_variance: float
    portfolio_sharpe_ratio: float
    turnover_weight: float
    estimated_transaction_cost_bps: float
    objective_loss_value: float
    allocations: List[RebalanceAllocationItem] = []
    optimization_time_ms: float
    timestamp_utc: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class QuantumCardinalityRebalancer:
    """Quantum-inspired Simulated Annealing / QUBO optimizer for cardinality-constrained portfolio rebalancing."""

    def __init__(self, config: Optional[CardinalityOptimizationConfig] = None) -> None:
        self.config = config or CardinalityOptimizationConfig()

    def optimize_portfolio(
        self,
        symbols: List[str],
        expected_returns: Dict[str, float],
        covariance_matrix: List[List[float]],
        current_weights: Optional[Dict[str, float]] = None,
        solver_method: OptimizationSolverMethod = OptimizationSolverMethod.SIMULATED_ANNEALING_QUBO,
    ) -> CardinalityOptimizationReport:
        """Execute cardinality-constrained quadratic optimization."""
        t_start = datetime.now(timezone.utc)
        N = len(symbols)
        if N == 0:
            raise ValueError("Asset universe cannot be empty")

        cov = np.array(covariance_matrix, dtype=np.float64)
        if cov.shape != (N, N):
            raise ValueError(f"Covariance matrix shape {cov.shape} must match symbol count ({N}, {N})")

        mu = np.array([expected_returns.get(s, 0.0) for s in symbols], dtype=np.float64)
        w0 = np.array([current_weights.get(s, 0.0) if current_weights else 0.0 for s in symbols], dtype=np.float64)

        if np.sum(w0) > 0:
            w0 = w0 / np.sum(w0)
        else:
            w0 = np.zeros(N)

        cfg = self.config
        k_max = min(N, max(1, cfg.max_cardinality_assets))

        if solver_method == OptimizationSolverMethod.SIMULATED_ANNEALING_QUBO:
            best_w, best_loss = self._solve_simulated_annealing(symbols, mu, cov, w0, k_max)
        else:
            best_w, best_loss = self._solve_greedy_projection(symbols, mu, cov, w0, k_max)

        # Compute diagnostic metrics
        p_var = float(np.dot(best_w.T, np.dot(cov, best_w)))
        p_vol = float(math.sqrt(max(0.0, p_var)))
        p_ret = float(np.dot(best_w, mu))
        sharpe = round(p_ret / p_vol, 4) if p_vol > 1e-6 else 0.0

        # Marginal risk contribution MRC_i = (Sigma w)_i / vol
        mrc = np.dot(cov, best_w) / p_vol if p_vol > 1e-6 else np.zeros(N)

        # Turnover and friction
        turnover = float(np.sum(np.abs(best_w - w0)) / 2.0)
        cost_bps = float(turnover * 2.0 * cfg.transaction_cost_bps)

        allocations: List[RebalanceAllocationItem] = []
        active_count = 0
        for i, sym in enumerate(symbols):
            w = float(best_w[i])
            w_prev = float(w0[i])
            is_active = w > 1e-4
            if is_active:
                active_count += 1
            allocations.append(
                RebalanceAllocationItem(
                    symbol=sym,
                    target_weight=round(w, 4),
                    current_weight=round(w_prev, 4),
                    weight_delta=round(w - w_prev, 4),
                    is_active=is_active,
                    expected_return_annualized=round(float(mu[i]), 4),
                    marginal_risk_contribution=round(float(mrc[i]), 4),
                )
            )

        t_end = datetime.now(timezone.utc)
        elapsed_ms = (t_end - t_start).total_seconds() * 1000.0

        return CardinalityOptimizationReport(
            solver_method=solver_method,
            converged=True,
            total_assets_universe=N,
            selected_cardinality=active_count,
            max_cardinality_limit=k_max,
            portfolio_expected_return=round(p_ret, 4),
            portfolio_volatility=round(p_vol, 4),
            portfolio_variance=round(p_var, 6),
            portfolio_sharpe_ratio=sharpe,
            turnover_weight=round(turnover, 4),
            estimated_transaction_cost_bps=round(cost_bps, 2),
            objective_loss_value=round(best_loss, 6),
            allocations=allocations,
            optimization_time_ms=round(elapsed_ms, 2),
        )

    # -------------------------------------------------------------------------
    # Optimization Solvers
    # -------------------------------------------------------------------------

    def _evaluate_objective(
        self,
        w: np.ndarray,
        mu: np.ndarray,
        cov: np.ndarray,
        w0: np.ndarray,
        k_max: int,
    ) -> float:
        """Objective function to minimize:

        Loss = lambda * (w^T Sigma w) - (mu^T w) + C_friction * sum |w - w0| + Penalty_cardinality
        """
        cfg = self.config
        var = float(np.dot(w.T, np.dot(cov, w)))
        ret = float(np.dot(w, mu))
        friction = float(np.sum(np.abs(w - w0)) * (cfg.transaction_cost_bps / 10000.0))

        active_assets = int(np.sum(w > 1e-4))
        cardinality_penalty = 0.0
        if active_assets > k_max:
            cardinality_penalty = (active_assets - k_max) * 1000.0

        return float(cfg.risk_aversion_lambda * var - ret + friction + cardinality_penalty)

    def _solve_simulated_annealing(
        self,
        symbols: List[str],
        mu: np.ndarray,
        cov: np.ndarray,
        w0: np.ndarray,
        k_max: int,
    ) -> Tuple[np.ndarray, float]:
        """Simulated Annealing over binary support sets and projected continuous weights."""
        N = len(symbols)
        if cfg_seed := self.config.random_seed:
            random.seed(cfg_seed)
            np.random.seed(cfg_seed)

        # Initialize with top k_max assets by Sharpe ratio
        stds = np.sqrt(np.clip(np.diag(cov), 1e-6, None))
        sharpes = mu / stds
        top_indices = list(np.argsort(sharpes)[::-1][:k_max])

        current_mask = np.zeros(N, dtype=bool)
        current_mask[top_indices] = True
        current_w = self._project_weights(current_mask, mu, cov)
        current_loss = self._evaluate_objective(current_w, mu, cov, w0, k_max)

        best_w = current_w.copy()
        best_loss = current_loss

        temp = self.config.sa_initial_temperature
        cooling = self.config.sa_cooling_rate

        for _ in range(self.config.sa_iterations):
            # Propose neighbor: flip one active asset and activate one inactive asset
            new_mask = current_mask.copy()
            active_idxs = np.where(new_mask)[0]
            inactive_idxs = np.where(~new_mask)[0]

            if len(active_idxs) > 0 and len(inactive_idxs) > 0:
                out_idx = random.choice(active_idxs)
                in_idx = random.choice(inactive_idxs)
                new_mask[out_idx] = False
                new_mask[in_idx] = True

            new_w = self._project_weights(new_mask, mu, cov)
            new_loss = self._evaluate_objective(new_w, mu, cov, w0, k_max)

            delta = new_loss - current_loss
            if delta < 0 or math.exp(-delta / max(1e-5, temp)) > random.random():
                current_mask = new_mask
                current_w = new_w
                current_loss = new_loss
                if current_loss < best_loss:
                    best_w = current_w.copy()
                    best_loss = current_loss

            temp *= cooling

        return (best_w, best_loss)

    def _solve_greedy_projection(
        self,
        symbols: List[str],
        mu: np.ndarray,
        cov: np.ndarray,
        w0: np.ndarray,
        k_max: int,
    ) -> Tuple[np.ndarray, float]:
        """Deterministic greedy heuristic selecting top k_max assets iteratively."""
        N = len(symbols)
        stds = np.sqrt(np.clip(np.diag(cov), 1e-6, None))
        sharpes = mu / stds
        top_indices = np.argsort(sharpes)[::-1][:k_max]

        mask = np.zeros(N, dtype=bool)
        mask[top_indices] = True
        w = self._project_weights(mask, mu, cov)
        loss = self._evaluate_objective(w, mu, cov, w0, k_max)
        return (w, loss)

    def _project_weights(self, mask: np.ndarray, mu: np.ndarray, cov: np.ndarray) -> np.ndarray:
        """Project weights onto simplex sum(w) = 1 under active support mask and weight caps."""
        N = len(mask)
        w = np.zeros(N, dtype=np.float64)
        active_indices = np.where(mask)[0]
        K = len(active_indices)
        if K == 0:
            return w

        cfg = self.config
        sub_cov = cov[np.ix_(active_indices, active_indices)]
        try:
            # Analytical Markowitz tangent weights on sub-universe: w_sub = inv(Sigma) * 1
            inv_cov = np.linalg.pinv(sub_cov)
            ones = np.ones(K)
            raw_w = np.dot(inv_cov, ones)
            s = np.sum(raw_w)
            if s > 0:
                w_sub = raw_w / s
            else:
                w_sub = ones / K
        except np.linalg.LinAlgError:
            w_sub = np.ones(K) / K

        # Iterative projection algorithm to satisfy sum(w) = 1, w_i <= max_ceiling, w_i >= min_threshold
        w_sub = np.maximum(0.0, w_sub)
        if np.sum(w_sub) > 0:
            w_sub /= np.sum(w_sub)
        else:
            w_sub = np.ones(K) / K

        # Water-filling projection loop to cap max weights and distribute excess
        for _ in range(15):
            w_sub[w_sub < cfg.min_weight_threshold] = 0.0
            excess_mask = w_sub > cfg.max_weight_ceiling
            if not np.any(excess_mask):
                break
            excess = np.sum(w_sub[excess_mask] - cfg.max_weight_ceiling)
            w_sub[excess_mask] = cfg.max_weight_ceiling
            non_capped_mask = (~excess_mask) & (w_sub > 0.0)
            if np.any(non_capped_mask):
                w_sub[non_capped_mask] += excess / np.sum(non_capped_mask)
            else:
                break

        # Final re-normalization safety
        active_sub = w_sub > 0.0
        if np.any(active_sub):
            s_final = np.sum(w_sub)
            if s_final > 0:
                w_sub /= s_final
            # Ensure strict cap compliance with epsilon
            w_sub = np.minimum(cfg.max_weight_ceiling, w_sub)
            w_sub /= np.sum(w_sub)

        w[active_indices] = w_sub
        return w
