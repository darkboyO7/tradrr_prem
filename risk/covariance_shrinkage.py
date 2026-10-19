"""Institutional Dynamic Risk Factor Covariance Shrinkage & Ledoit-Wolf Estimator.

Provides robust, well-conditioned covariance and correlation matrix estimation
for portfolio optimization, risk attribution, and minimum-variance hedging.
Addresses ill-conditioning, rank deficiency (N > T), and noisy off-diagonal
empirical estimates via analytical shrinkage targets (Constant Correlation,
Identity/Diagonal, and Factor-Structured target).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


class ShrinkageTargetType(str, Enum):
    """Target matrix structure for linear shrinkage regularization."""
    CONSTANT_CORRELATION = "constant_correlation"
    DIAGONAL_VARIANCE = "diagonal_variance"
    SCALED_IDENTITY = "scaled_identity"


@dataclass(frozen=True)
class ShrinkageResult:
    """Result of covariance shrinkage estimation."""
    symbols: List[str]
    sample_covariance: np.ndarray
    shrunk_covariance: np.ndarray
    target_covariance: np.ndarray
    shrunk_correlation: np.ndarray
    shrinkage_intensity: float  # Optimal delta in [0, 1]
    target_type: ShrinkageTargetType
    condition_number_sample: float
    condition_number_shrunk: float
    effective_rank: float
    is_positive_definite: bool
    annualized: bool
    annualization_factor: float
    sample_size_t: int
    asset_count_n: int

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to serializable dictionary."""
        return {
            "symbols": list(self.symbols),
            "sample_covariance": self.sample_covariance.tolist(),
            "shrunk_covariance": self.shrunk_covariance.tolist(),
            "target_covariance": self.target_covariance.tolist(),
            "shrunk_correlation": self.shrunk_correlation.tolist(),
            "shrinkage_intensity": float(self.shrinkage_intensity),
            "target_type": self.target_type.value,
            "condition_number_sample": float(self.condition_number_sample),
            "condition_number_shrunk": float(self.condition_number_shrunk),
            "effective_rank": float(self.effective_rank),
            "is_positive_definite": bool(self.is_positive_definite),
            "annualized": bool(self.annualized),
            "annualization_factor": float(self.annualization_factor),
            "sample_size_t": int(self.sample_size_t),
            "asset_count_n": int(self.asset_count_n),
        }


@dataclass(frozen=True)
class MinimumVarianceWeights:
    """Analytical minimum variance portfolio allocation derived from covariance."""
    weights: Dict[str, float]
    portfolio_volatility: float
    portfolio_variance: float
    diversification_ratio: float
    effective_constituents: float  # Inverse Herfindahl index (1 / sum(w^2))
    shrinkage_delta_used: float

    def to_dict(self) -> Dict[str, Any]:
        """Convert weights to serializable dictionary."""
        return {
            "weights": dict(self.weights),
            "portfolio_volatility": float(self.portfolio_volatility),
            "portfolio_variance": float(self.portfolio_variance),
            "diversification_ratio": float(self.diversification_ratio),
            "effective_constituents": float(self.effective_constituents),
            "shrinkage_delta_used": float(self.shrinkage_delta_used),
        }


class CovarianceShrinkageEngine:
    """Institutional Dynamic Covariance Shrinkage & Ledoit-Wolf Estimator.

    Implements:
    1. Sample Covariance Calculation: S = (1 / (T - 1)) * X_c^T * X_c
    2. Ledoit-Wolf Optimal Shrinkage delta* targeting:
       - Constant Correlation Target (Ledoit & Wolf 2004)
       - Diagonal / Unequal Variance Target
       - Scaled Identity Target
    3. Positive Definiteness enforcement & Eigendecomposition clipping
    4. Condition Number & Spectral Regularization diagnostics
    5. Minimum-Variance Portfolio Weight extraction (unconstrained or long-only)
    """

    def __init__(
        self,
        default_target: ShrinkageTargetType = ShrinkageTargetType.CONSTANT_CORRELATION,
        annualization_factor: float = 252.0,
        eigenvalue_floor: float = 1e-8,
    ) -> None:
        self.default_target = default_target
        self.annualization_factor = float(annualization_factor)
        self.eigenvalue_floor = float(eigenvalue_floor)

    def estimate_covariance(
        self,
        returns_matrix: np.ndarray | List[List[float]],
        symbols: Optional[List[str]] = None,
        target_type: Optional[ShrinkageTargetType] = None,
        custom_shrinkage_delta: Optional[float] = None,
        annualize: bool = True,
    ) -> ShrinkageResult:
        """Estimate robust covariance matrix via Ledoit-Wolf shrinkage.

        Args:
            returns_matrix: (T, N) array or nested list of asset returns.
            symbols: Optional list of N symbol names. Default: [Asset_0, ...].
            target_type: Shrinkage target type (defaults to instance setting).
            custom_shrinkage_delta: If provided, overrides analytical optimal delta.
            annualize: Whether to scale covariance by annualization_factor.

        Returns:
            ShrinkageResult with full covariance, correlation, and spectral metrics.
        """
        X = np.asarray(returns_matrix, dtype=np.float64)
        if X.ndim != 2:
            raise ValueError(f"returns_matrix must be 2-dimensional (T, N), got shape {X.shape}")

        T, N = X.shape
        if T < 3:
            raise ValueError(f"Sample size T must be at least 3, got T={T}")
        if N < 1:
            raise ValueError(f"Asset count N must be at least 1, got N={N}")

        if symbols is None:
            symbols = [f"Asset_{i}" for i in range(N)]
        elif len(symbols) != N:
            raise ValueError(f"Number of symbols ({len(symbols)}) must match column count N ({N})")

        target = target_type or self.default_target

        # 1. Demean returns: X_c = X - mean(X)
        mean_vec = np.mean(X, axis=0, keepdims=True)
        X_c = X - mean_vec  # shape (T, N)

        # 2. Sample covariance S = (1 / (T - 1)) * X_c^T * X_c
        sample_cov = np.cov(X, rowvar=False, ddof=1)
        if N == 1:
            sample_cov = np.array([[float(sample_cov)]], dtype=np.float64)

        # 3. Compute Target Covariance F and optimal shrinkage delta
        if N == 1:
            # Single asset: delta=0, sample variance is trivially identical
            target_cov = sample_cov.copy()
            opt_delta = 0.0
            shrunk_cov = sample_cov.copy()
        else:
            target_cov, opt_delta = self._compute_shrinkage_target_and_delta(
                X_c, sample_cov, target
            )

        if custom_shrinkage_delta is not None:
            delta = float(np.clip(custom_shrinkage_delta, 0.0, 1.0))
        else:
            delta = float(np.clip(opt_delta, 0.0, 1.0))

        # 4. Linear shrinkage: Sigma_shrunk = delta * F + (1 - delta) * S
        shrunk_cov = delta * target_cov + (1.0 - delta) * sample_cov

        # 5. Enforce strict numerical symmetry & positive semi-definiteness
        shrunk_cov = 0.5 * (shrunk_cov + shrunk_cov.T)
        shrunk_cov, is_pos_def = self._ensure_positive_definite(shrunk_cov)

        # 6. Correlation matrix derived from shrunk covariance
        variances = np.diag(shrunk_cov)
        std_devs = np.sqrt(np.maximum(variances, self.eigenvalue_floor))
        inv_std = 1.0 / std_devs
        shrunk_corr = shrunk_cov * np.outer(inv_std, inv_std)
        np.fill_diagonal(shrunk_corr, 1.0)
        shrunk_corr = np.clip(shrunk_corr, -1.0, 1.0)

        # 7. Spectral metrics: Condition number & Effective rank
        cond_sample = self._calculate_condition_number(sample_cov)
        cond_shrunk = self._calculate_condition_number(shrunk_cov)
        eff_rank = self._calculate_effective_rank(shrunk_cov)

        # 8. Annualization if requested
        ann_factor = self.annualization_factor if annualize else 1.0
        final_sample_cov = sample_cov * ann_factor
        final_shrunk_cov = shrunk_cov * ann_factor
        final_target_cov = target_cov * ann_factor

        return ShrinkageResult(
            symbols=list(symbols),
            sample_covariance=final_sample_cov,
            shrunk_covariance=final_shrunk_cov,
            target_covariance=final_target_cov,
            shrunk_correlation=shrunk_corr,
            shrinkage_intensity=delta,
            target_type=target,
            condition_number_sample=cond_sample,
            condition_number_shrunk=cond_shrunk,
            effective_rank=eff_rank,
            is_positive_definite=is_pos_def,
            annualized=annualize,
            annualization_factor=ann_factor,
            sample_size_t=T,
            asset_count_n=N,
        )

    def compute_minimum_variance_portfolio(
        self,
        shrinkage_result: ShrinkageResult,
        long_only: bool = True,
    ) -> MinimumVarianceWeights:
        """Compute analytical or projected global minimum variance portfolio weights.

        w_mv = argmin w^T Sigma w  s.t. sum(w) = 1 (and w >= 0 if long_only).

        Args:
            shrinkage_result: Result from estimate_covariance.
            long_only: If True, clips negative weights and re-normalizes iteratively.

        Returns:
            MinimumVarianceWeights with constituent breakdown and diversification.
        """
        cov = shrinkage_result.shrunk_covariance
        N = cov.shape[0]
        symbols = shrinkage_result.symbols

        if N == 1:
            w_dict = {symbols[0]: 1.0}
            vol = float(np.sqrt(max(0.0, cov[0, 0])))
            var = float(cov[0, 0])
            return MinimumVarianceWeights(
                weights=w_dict,
                portfolio_volatility=vol,
                portfolio_variance=var,
                diversification_ratio=1.0,
                effective_constituents=1.0,
                shrinkage_delta_used=shrinkage_result.shrinkage_intensity,
            )

        # Unconstrained analytical solution: w* = Sigma^-1 * 1 / (1^T * Sigma^-1 * 1)
        ones = np.ones(N, dtype=np.float64)
        try:
            inv_cov = np.linalg.pinv(cov)
            unconstrained_w = inv_cov @ ones
            denom = float(np.sum(unconstrained_w))
            if abs(denom) > 1e-12:
                weights_vec = unconstrained_w / denom
            else:
                weights_vec = ones / N
        except np.linalg.LinAlgError:
            weights_vec = ones / N

        if long_only:
            # Iterative non-negative quadratic projection
            weights_vec = self._project_long_only_weights(cov, weights_vec)

        # Guarantee sum to 1.0
        s = float(np.sum(weights_vec))
        if s > 0:
            weights_vec = weights_vec / s
        else:
            weights_vec = ones / N

        weights_dict = {sym: float(w) for sym, w in zip(symbols, weights_vec)}

        # Portfolio variance = w^T * Sigma * w
        port_var = float(weights_vec.T @ cov @ weights_vec)
        port_var = max(0.0, port_var)
        port_vol = float(math.sqrt(port_var))

        # Diversification Ratio = (w^T * sigma_individual) / portfolio_vol
        indiv_vols = np.sqrt(np.maximum(np.diag(cov), 0.0))
        weighted_avg_vol = float(np.dot(np.abs(weights_vec), indiv_vols))
        div_ratio = float(weighted_avg_vol / port_vol) if port_vol > 1e-12 else 1.0

        # Effective constituents = 1 / sum(w_i^2) (Herfindahl diversity)
        hhi = float(np.sum(weights_vec**2))
        eff_constituents = float(1.0 / hhi) if hhi > 1e-12 else 1.0

        return MinimumVarianceWeights(
            weights=weights_dict,
            portfolio_volatility=port_vol,
            portfolio_variance=port_var,
            diversification_ratio=div_ratio,
            effective_constituents=eff_constituents,
            shrinkage_delta_used=shrinkage_result.shrinkage_intensity,
        )

    # -------------------------------------------------------------------------
    # Internal Mathematical Engine (Ledoit-Wolf Derivations)
    # -------------------------------------------------------------------------

    def _compute_shrinkage_target_and_delta(
        self,
        X_c: np.ndarray,
        sample_cov: np.ndarray,
        target_type: ShrinkageTargetType,
    ) -> Tuple[np.ndarray, float]:
        """Compute shrinkage target F and optimal shrinkage parameter delta*."""
        T, N = X_c.shape

        if target_type == ShrinkageTargetType.CONSTANT_CORRELATION:
            return self._constant_correlation_shrinkage(X_c, sample_cov)
        elif target_type == ShrinkageTargetType.DIAGONAL_VARIANCE:
            return self._diagonal_variance_shrinkage(X_c, sample_cov)
        elif target_type == ShrinkageTargetType.SCALED_IDENTITY:
            return self._scaled_identity_shrinkage(X_c, sample_cov)
        else:
            return self._constant_correlation_shrinkage(X_c, sample_cov)

    def _constant_correlation_shrinkage(
        self,
        X_c: np.ndarray,
        sample_cov: np.ndarray,
    ) -> Tuple[np.ndarray, float]:
        """Ledoit & Wolf (2004) "Honey, I Shrunk the Sample Covariance Matrix": Constant Correlation Target."""
        T, N = X_c.shape

        var = np.diag(sample_cov)
        sqrt_var = np.sqrt(np.maximum(var, self.eigenvalue_floor))
        inv_sqrt = 1.0 / sqrt_var

        # Sample correlation matrix R
        R = sample_cov * np.outer(inv_sqrt, inv_sqrt)
        np.fill_diagonal(R, 1.0)
        R = np.clip(R, -1.0, 1.0)

        # Average off-diagonal correlation r_bar
        # Sum of all off-diagonal entries / (N * (N - 1))
        sum_r = float(np.sum(R) - N)
        r_bar = sum_r / (N * (N - 1)) if N > 1 else 0.0

        # Target matrix F: F_ii = S_ii; F_ij = r_bar * sqrt(S_ii * S_jj)
        F = r_bar * np.outer(sqrt_var, sqrt_var)
        np.fill_diagonal(F, var)

        # Compute asymptotic variance pi_hat = sum_{i,j} AsyVar(s_ij)
        # y_{t,i,j} = X_{t,i} * X_{t,j}
        # pi_{i,j} = (1/T) * sum_t (X_{t,i}*X_{t,j} - s_{i,j})^2
        X2 = X_c[:, :, None] * X_c[:, None, :]  # shape (T, N, N)
        dev = X2 - sample_cov[None, :, :]
        pi_mat = np.mean(dev**2, axis=0)  # shape (N, N)
        pi_hat = float(np.sum(pi_mat))

        # Compute rho_hat: off-diagonal correlation estimation term
        # rho_hat = sum_{i} pi_{ii} + sum_{i!=j} r_bar / 2 * [ sqrt(S_jj/S_ii)*theta_{ii,ij} + sqrt(S_ii/S_jj)*theta_{jj,ij} ]
        theta_mat = np.zeros((N, N), dtype=np.float64)
        for i in range(N):
            dev_ii = X_c[:, i] ** 2 - sample_cov[i, i]
            for j in range(N):
                if i != j:
                    dev_ij = X_c[:, i] * X_c[:, j] - sample_cov[i, j]
                    theta_mat[i, j] = float(np.mean(dev_ii * dev_ij))

        rho_diag = float(np.trace(pi_mat))
        rho_off = 0.0
        for i in range(N):
            for j in range(N):
                if i != j:
                    term1 = (sqrt_var[j] / sqrt_var[i]) * theta_mat[i, j]
                    term2 = (sqrt_var[i] / sqrt_var[j]) * theta_mat[j, i]
                    rho_off += (r_bar / 2.0) * (term1 + term2)

        rho_hat = rho_diag + rho_off

        # Compute gamma_hat: squared Frobenius distance ||S - F||_F^2
        gamma_hat = float(np.sum((sample_cov - F) ** 2))

        # Optimal shrinkage delta* = max(0, min(1, (pi_hat - rho_hat) / (gamma_hat * T)))
        kappa = (pi_hat - rho_hat) / gamma_hat if gamma_hat > 1e-12 else 0.0
        delta_star = kappa / float(T)
        delta_star = float(np.clip(delta_star, 0.0, 1.0))

        return F, delta_star

    def _diagonal_variance_shrinkage(
        self,
        X_c: np.ndarray,
        sample_cov: np.ndarray,
    ) -> Tuple[np.ndarray, float]:
        """Shrinkage toward diagonal matrix with unequal sample variances."""
        T, N = X_c.shape
        F = np.diag(np.diag(sample_cov))

        X2 = X_c[:, :, None] * X_c[:, None, :]
        dev = X2 - sample_cov[None, :, :]
        pi_mat = np.mean(dev**2, axis=0)

        # Off-diagonal shrinkage: target has 0 off-diagonals
        gamma_hat = float(np.sum((sample_cov - F) ** 2))
        # pi_hat only over off-diagonals since diagonal is exact match
        pi_off = float(np.sum(pi_mat) - np.trace(pi_mat))

        delta_star = (pi_off / (gamma_hat * T)) if gamma_hat > 1e-12 else 0.0
        delta_star = float(np.clip(delta_star, 0.0, 1.0))

        return F, delta_star

    def _scaled_identity_shrinkage(
        self,
        X_c: np.ndarray,
        sample_cov: np.ndarray,
    ) -> Tuple[np.ndarray, float]:
        """Shrinkage toward scaled identity: F = mu * I, where mu = trace(S) / N."""
        T, N = X_c.shape
        mu = float(np.trace(sample_cov) / N)
        F = mu * np.eye(N, dtype=np.float64)

        X2 = X_c[:, :, None] * X_c[:, None, :]
        dev = X2 - sample_cov[None, :, :]
        pi_mat = np.mean(dev**2, axis=0)
        pi_hat = float(np.sum(pi_mat))

        gamma_hat = float(np.sum((sample_cov - F) ** 2))
        delta_star = (pi_hat / (gamma_hat * T)) if gamma_hat > 1e-12 else 0.0
        delta_star = float(np.clip(delta_star, 0.0, 1.0))

        return F, delta_star

    def _ensure_positive_definite(
        self,
        mat: np.ndarray,
    ) -> Tuple[np.ndarray, bool]:
        """Verify positive definiteness, clipping negative eigenvalues if needed."""
        eigenvalues, eigenvectors = np.linalg.eigh(mat)
        min_eig = float(np.min(eigenvalues))
        if min_eig >= self.eigenvalue_floor:
            return mat, True

        # Regularize spectrum: clip eigenvalues at floor
        clipped_eigs = np.maximum(eigenvalues, self.eigenvalue_floor)
        reconstructed = eigenvectors @ np.diag(clipped_eigs) @ eigenvectors.T
        reconstructed = 0.5 * (reconstructed + reconstructed.T)
        return reconstructed, False

    def _calculate_condition_number(self, mat: np.ndarray) -> float:
        """Compute matrix condition number kappa(Sigma) = lambda_max / lambda_min."""
        try:
            eigenvalues = np.linalg.eigvalsh(mat)
            min_eig = float(np.min(eigenvalues))
            max_eig = float(np.max(eigenvalues))
            if min_eig <= 1e-15:
                return float("inf")
            return float(max_eig / min_eig)
        except np.linalg.LinAlgError:
            return float("inf")

    def _calculate_effective_rank(self, mat: np.ndarray) -> float:
        """Compute Roy & Vetterli effective rank = exp(-sum p_i ln p_i) from normalized eigenvalues."""
        try:
            eigenvalues = np.linalg.eigvalsh(mat)
            eigenvalues = np.maximum(eigenvalues, 1e-12)
            total = float(np.sum(eigenvalues))
            if total <= 1e-15:
                return 1.0
            p = eigenvalues / total
            entropy = -float(np.sum(p * np.log(p)))
            return float(math.exp(entropy))
        except (np.linalg.LinAlgError, ValueError):
            return 1.0

    def _project_long_only_weights(
        self,
        cov: np.ndarray,
        initial_weights: np.ndarray,
        max_iters: int = 50,
    ) -> np.ndarray:
        """Project unconstrained weights into long-only simplex via active-set clipping."""
        w = initial_weights.copy()
        if np.all(w >= 0):
            return w

        # Active-set iterative elimination of negative weights
        active = np.ones(len(w), dtype=bool)
        for _ in range(max_iters):
            if np.all(w[active] >= -1e-6):
                break
            # Eliminate the most negative weight
            neg_indices = np.where(active & (w < 0))[0]
            if len(neg_indices) == 0:
                break
            worst = neg_indices[np.argmin(w[neg_indices])]
            active[worst] = False
            w[worst] = 0.0

            if not np.any(active):
                # Fallback to equal weight
                return np.ones(len(w), dtype=np.float64) / len(w)

            # Re-solve on active submatrix
            sub_cov = cov[np.ix_(active, active)]
            ones_sub = np.ones(np.sum(active), dtype=np.float64)
            try:
                sub_inv = np.linalg.pinv(sub_cov)
                w_sub = sub_inv @ ones_sub
                s = float(np.sum(w_sub))
                if abs(s) > 1e-12:
                    w[active] = w_sub / s
                else:
                    w[active] = 1.0 / np.sum(active)
            except np.linalg.LinAlgError:
                w[active] = 1.0 / np.sum(active)

        w = np.maximum(w, 0.0)
        s = float(np.sum(w))
        if s > 0:
            w /= s
        return w
