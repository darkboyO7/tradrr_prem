"""Trading Intelligence Platform - Statistical Arbitrage & Dynamic Cointegration Engine.

Provides online recursive Kalman Filter hedge ratio tracking, Engle-Granger cointegration testing,
Ornstein-Uhlenbeck mean-reversion half-life estimation, rolling Z-score spread tracking, and
CUSUM structural break detection for multi-asset pairs trading.
"""

from datetime import datetime, timezone
import math
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field
from scipy import stats

from core.logging import get_logger

logger = get_logger("stat_arb")


class CointegrationResult(BaseModel):
    """Statistical summary of cointegration and mean-reversion properties."""

    pair_name: str
    asset_y: str
    asset_x: str
    is_cointegrated: bool
    adf_t_statistic: float
    critical_values: Dict[str, float]
    p_value_approx: float
    static_hedge_ratio: float
    intercept: float
    half_life_bars: float
    mean_reversion_speed: float
    r_squared: float
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class PairTradingSignal(BaseModel):
    """Actionable statistical arbitrage trading signal for an asset pair."""

    pair_name: str
    action: str  # "LONG_SPREAD", "SHORT_SPREAD", "UNWIND_SPREAD", "STOP_LOSS_EXIT", "HOLD"
    z_score: float
    current_spread: float
    hedge_ratio: float
    asset_y: str
    asset_x: str
    weight_y: float
    weight_x: float
    half_life_bars: float
    is_structurally_broken: bool
    rationale: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class KalmanFilterHedgeRatio:
    """Online 2-state Kalman Filter estimating dynamic hedge ratio beta and intercept alpha."""

    def __init__(
        self,
        delta: float = 1e-4,     # State transition covariance factor
        r_noise: float = 1e-3,   # Measurement noise variance
        initial_beta: float = 1.0,
        initial_alpha: float = 0.0,
    ) -> None:
        self.delta = delta
        self.r_noise = r_noise
        # State vector: [alpha, beta]^T
        self.state = np.array([initial_alpha, initial_beta], dtype=np.float64)
        # State estimation covariance matrix
        self.p_matrix = np.eye(2, dtype=np.float64) * 1.0
        # Process noise covariance matrix
        self.q_matrix = np.eye(2, dtype=np.float64) * (delta / (1.0 - delta))

    def update(self, price_y: float, price_x: float) -> Tuple[float, float, float]:
        """Update Kalman state with a new pair observation (y, x).
        
        Returns:
            Tuple of (alpha_t, beta_t, spread_t)
        """
        # Observation vector: H = [1.0, price_x]
        h = np.array([1.0, price_x], dtype=np.float64)

        # 1. State prediction (Random walk model: state_prior = state_post_{t-1})
        state_prior = self.state
        p_prior = self.p_matrix + self.q_matrix

        # 2. Measurement prediction & innovation
        y_hat = float(np.dot(h, state_prior))
        error = price_y - y_hat  # Measurement innovation

        # 3. Innovation covariance
        f_scalar = float(np.dot(h, np.dot(p_prior, h)) + self.r_noise)

        # 4. Kalman Gain
        k_gain = np.dot(p_prior, h) / f_scalar

        # 5. Posterior state update
        self.state = state_prior + (k_gain * error)
        # Joseph form or standard update for covariance
        self.p_matrix = p_prior - np.outer(k_gain, h).dot(p_prior)

        alpha, beta = float(self.state[0]), float(self.state[1])
        spread = error  # One-step-ahead forecast innovation spread

        return alpha, beta, spread

    def filter_series(
        self,
        series_y: pd.Series,
        series_x: pd.Series,
    ) -> pd.DataFrame:
        """Run batch Kalman Filter over chronological historical series."""
        n = min(len(series_y), len(series_x))
        alphas = np.zeros(n)
        betas = np.zeros(n)
        spreads = np.zeros(n)

        y_vals = series_y.values[:n]
        x_vals = series_x.values[:n]

        for i in range(n):
            a, b, s = self.update(float(y_vals[i]), float(x_vals[i]))
            alphas[i] = a
            betas[i] = b
            spreads[i] = s

        return pd.DataFrame({
            "alpha": alphas,
            "beta": betas,
            "spread": spreads,
        }, index=series_y.index[:n])


class CointegrationAnalyzer:
    """Performs Engle-Granger cointegration testing and Ornstein-Uhlenbeck parameter estimation."""

    # Standard MacKinnon ADF critical values for model with constant, no trend (sample N >= 100)
    ADF_CRITICAL_VALUES = {
        "1%": -3.498,
        "5%": -2.891,
        "10%": -2.582,
    }

    def _estimate_half_life(self, spread: np.ndarray) -> Tuple[float, float]:
        """Estimate mean-reversion rate theta and half-life using Ornstein-Uhlenbeck discretization.
        
        Delta S_t = -theta * (S_{t-1} - mu) * dt + sigma * dW_t
        """
        if len(spread) < 10:
            return 0.0, 999.0

        spread_lag = spread[:-1]
        delta_spread = spread[1:] - spread_lag

        # Linear regression: delta_spread = a + b * spread_lag
        x_mat = np.column_stack([np.ones_like(spread_lag), spread_lag])
        params, _, _, _ = np.linalg.lstsq(x_mat, delta_spread, rcond=None)
        b = params[1]

        if b >= 0.0:
            # Non-mean-reverting process
            return 0.0, 999.0

        theta = -b
        half_life = math.log(2.0) / theta
        return float(theta), float(max(1.0, min(half_life, 999.0)))

    def _adf_test_residuals(self, residuals: np.ndarray) -> Tuple[float, float]:
        """Compute Augmented Dickey-Fuller t-statistic on residuals."""
        if len(residuals) < 15:
            return 0.0, 1.0

        res_lag = residuals[:-1]
        delta_res = residuals[1:] - res_lag

        # Regress delta_res on res_lag
        x = res_lag
        y = delta_res
        x_mean = np.mean(x)
        y_mean = np.mean(y)

        # Slope estimate
        numerator = np.sum((x - x_mean) * (y - y_mean))
        denominator = np.sum((x - x_mean) ** 2)

        if denominator < 1e-12:
            return 0.0, 1.0

        beta = numerator / denominator
        alpha = y_mean - beta * x_mean

        # Residuals of ADF equation
        y_hat = alpha + beta * x
        eps = y - y_hat
        n = len(y)
        dof = n - 2
        s_err = np.sqrt(np.sum(eps ** 2) / max(dof, 1))
        se_beta = s_err / np.sqrt(denominator)

        if se_beta < 1e-12:
            return 0.0, 1.0

        t_stat = beta / se_beta

        # Approximate p-value based on MacKinnon response surface approximation
        if t_stat <= self.ADF_CRITICAL_VALUES["1%"]:
            p_val = 0.01
        elif t_stat <= self.ADF_CRITICAL_VALUES["5%"]:
            p_val = 0.05
        elif t_stat <= self.ADF_CRITICAL_VALUES["10%"]:
            p_val = 0.10
        else:
            p_val = min(1.0, 0.15 + max(0.0, (t_stat - self.ADF_CRITICAL_VALUES["10%"]) * 0.2))

        return float(t_stat), float(p_val)

    def test_cointegration(
        self,
        series_y: pd.Series,
        series_x: pd.Series,
        asset_y: str = "ETH-USD",
        asset_x: str = "BTC-USD",
    ) -> CointegrationResult:
        """Run Engle-Granger cointegration test and calculate mean-reversion metrics."""
        y = series_y.values.astype(np.float64)
        x = series_x.values.astype(np.float64)
        n = min(len(y), len(x))
        y = y[:n]
        x = x[:n]

        if n < 20:
            raise ValueError(f"Insufficient data for cointegration test (min 20 bars, got {n})")

        # 1. Step 1: Static OLS y = alpha + beta * x
        x_mat = np.column_stack([np.ones(n), x])
        params, residuals_sum, _, _ = np.linalg.lstsq(x_mat, y, rcond=None)
        alpha, beta = float(params[0]), float(params[1])

        # Residual spread
        spread = y - (alpha + beta * x)

        # R-squared
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        ss_res = np.sum(spread ** 2)
        r_squared = float(max(0.0, 1.0 - (ss_res / max(ss_tot, 1e-12))))

        # 2. Step 2: ADF test on spread residuals
        t_stat, p_val = self._adf_test_residuals(spread)

        # Check if null of unit root is rejected at 5% significance
        is_cointegrated = t_stat < self.ADF_CRITICAL_VALUES["5%"]

        # 3. Estimate Ornstein-Uhlenbeck half-life
        theta, half_life = self._estimate_half_life(spread)

        pair_name = f"{asset_y}/{asset_x}"

        logger.info(
            "Cointegration test completed",
            pair=pair_name,
            cointegrated=is_cointegrated,
            t_stat=round(t_stat, 3),
            p_val=round(p_val, 3),
            hedge_ratio=round(beta, 4),
            half_life_bars=round(half_life, 1),
        )

        return CointegrationResult(
            pair_name=pair_name,
            asset_y=asset_y,
            asset_x=asset_x,
            is_cointegrated=is_cointegrated,
            adf_t_statistic=round(t_stat, 4),
            critical_values=self.ADF_CRITICAL_VALUES,
            p_value_approx=round(p_val, 4),
            static_hedge_ratio=round(beta, 6),
            intercept=round(alpha, 4),
            half_life_bars=round(half_life, 2),
            mean_reversion_speed=round(theta, 4),
            r_squared=round(r_squared, 4),
        )


class PairsSpreadMonitor:
    """Real-time pairs spread Z-score monitoring, signal generation, and CUSUM break detection."""

    def __init__(
        self,
        pair_name: str,
        asset_y: str,
        asset_x: str,
        z_entry: float = 2.0,
        z_exit: float = 0.5,
        z_stop: float = 3.5,
        cusum_threshold: float = 5.0,
        window: int = 60,
    ) -> None:
        self.pair_name = pair_name
        self.asset_y = asset_y
        self.asset_x = asset_x
        self.z_entry = z_entry
        self.z_exit = z_exit
        self.z_stop = z_stop
        self.cusum_threshold = cusum_threshold
        self.window = window

        self.kalman = KalmanFilterHedgeRatio()
        self.spread_history: List[float] = []
        self._cusum_pos = 0.0
        self._cusum_neg = 0.0

    def process_observation(
        self,
        price_y: float,
        price_x: float,
        half_life_bars: float = 15.0,
    ) -> PairTradingSignal:
        """Ingest new tick/candle pair prices, update dynamic Kalman state, and generate signal."""
        alpha, beta, current_spread = self.kalman.update(price_y, price_x)
        self.spread_history.append(current_spread)

        if len(self.spread_history) > self.window * 2:
            self.spread_history.pop(0)

        # Rolling mean and standard deviation
        active_spreads = self.spread_history[-self.window:]
        mean_spread = float(np.mean(active_spreads))
        std_spread = float(np.std(active_spreads))

        if std_spread < 1e-6:
            z_score = 0.0
        else:
            z_score = (current_spread - mean_spread) / std_spread

        # CUSUM structural break detection
        self._cusum_pos = max(0.0, self._cusum_pos + (z_score - 0.5))
        self._cusum_neg = max(0.0, self._cusum_neg - (z_score + 0.5))
        is_structurally_broken = (self._cusum_pos > self.cusum_threshold) or (self._cusum_neg > self.cusum_threshold)

        action = "HOLD"
        rationale = "Spread within normal equilibrium band"

        if is_structurally_broken:
            action = "STOP_LOSS_EXIT"
            rationale = f"Structural break detected via CUSUM (C+={self._cusum_pos:.1f}, C-={self._cusum_neg:.1f})"
        elif abs(z_score) >= self.z_stop:
            action = "STOP_LOSS_EXIT"
            rationale = f"Divergence stop-loss breached (|Z|={abs(z_score):.2f} >= {self.z_stop})"
        elif z_score >= self.z_entry:
            action = "SHORT_SPREAD"
            rationale = f"Spread elevated (+{z_score:.2f}Z): Short {self.asset_y}, Long {self.asset_x}"
        elif z_score <= -self.z_entry:
            action = "LONG_SPREAD"
            rationale = f"Spread depressed ({z_score:.2f}Z): Long {self.asset_y}, Short {self.asset_x}"
        elif abs(z_score) <= self.z_exit:
            action = "UNWIND_SPREAD"
            rationale = f"Spread converged to mean (|Z|={abs(z_score):.2f} <= {self.z_exit})"

        # Portfolio weights normalized for beta neutrality
        # Dollar neutrality: weight_y = 1.0, weight_x = -beta * (price_x / price_y)
        weight_y = 1.0
        weight_x = float(round(-beta * (price_x / max(price_y, 1e-6)), 4))

        return PairTradingSignal(
            pair_name=self.pair_name,
            action=action,
            z_score=round(z_score, 3),
            current_spread=round(current_spread, 4),
            hedge_ratio=round(beta, 4),
            asset_y=self.asset_y,
            asset_x=self.asset_x,
            weight_y=weight_y,
            weight_x=weight_x,
            half_life_bars=round(half_life_bars, 1),
            is_structurally_broken=is_structurally_broken,
            rationale=rationale,
        )
