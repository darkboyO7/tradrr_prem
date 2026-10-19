"""Trading Intelligence Platform - Institutional Derivatives Pricing & Greeks Surface Engine.

Provides analytical Black-Scholes-Merton and Merton Jump-Diffusion models,
high-precision Implied Volatility (IV) solving via Newton-Raphson/Brent,
volatility surface interpolation, and portfolio-level Greek risk aggregation.
"""

from datetime import datetime, timezone
from enum import Enum
import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from pydantic import BaseModel, Field
import scipy.optimize as opt
import scipy.stats as st

from core.logging import get_logger

logger = get_logger("derivatives")


class OptionType(str, Enum):
    """Option payoff specification."""

    CALL = "CALL"
    PUT = "PUT"


class ExerciseStyle(str, Enum):
    """Exercise style specification."""

    EUROPEAN = "EUROPEAN"
    AMERICAN = "AMERICAN"


class OptionGreeks(BaseModel):
    """Comprehensive analytical Greeks and valuation metrics."""

    price: float
    intrinsic_value: float
    time_value: float
    delta: float
    gamma: float
    vega_1pct: float
    vega_raw: float
    theta_daily: float
    theta_annualized: float
    rho_1pct: float
    rho_raw: float
    vanna: float
    volga: float
    implied_volatility: Optional[float] = None


class BlackScholesEngine:
    """Institutional Black-Scholes-Merton (1973/1976) analytical options pricing engine."""

    @staticmethod
    def calculate_greeks(
        spot: float,
        strike: float,
        time_to_expiry: float,
        volatility: float,
        risk_free_rate: float = 0.0,
        dividend_yield: float = 0.0,
        option_type: OptionType = OptionType.CALL,
    ) -> OptionGreeks:
        """Compute closed-form BSM option price and complete 1st & 2nd order Greeks."""
        if spot <= 0 or strike <= 0:
            raise ValueError(f"Spot ({spot}) and Strike ({strike}) must be strictly positive.")

        is_call = option_type == OptionType.CALL
        intrinsic = max(0.0, (spot - strike) if is_call else (strike - spot))

        # Handle expired / zero time to maturity
        if time_to_expiry <= 1e-7:
            delta = (1.0 if spot > strike else 0.0) if is_call else (-1.0 if spot < strike else 0.0)
            return OptionGreeks(
                price=intrinsic,
                intrinsic_value=intrinsic,
                time_value=0.0,
                delta=delta,
                gamma=0.0,
                vega_1pct=0.0,
                vega_raw=0.0,
                theta_daily=0.0,
                theta_annualized=0.0,
                rho_1pct=0.0,
                rho_raw=0.0,
                vanna=0.0,
                volga=0.0,
                implied_volatility=volatility,
            )

        # Handle zero or near-zero volatility
        if volatility <= 1e-7:
            disc_s = spot * math.exp(-dividend_yield * time_to_expiry)
            disc_k = strike * math.exp(-risk_free_rate * time_to_expiry)
            forward_diff = disc_s - disc_k
            price = max(0.0, forward_diff if is_call else -forward_diff)
            delta = (math.exp(-dividend_yield * time_to_expiry) if forward_diff > 0 else 0.0) if is_call else (
                -math.exp(-dividend_yield * time_to_expiry) if forward_diff < 0 else 0.0
            )
            return OptionGreeks(
                price=price,
                intrinsic_value=intrinsic,
                time_value=max(0.0, price - intrinsic),
                delta=delta,
                gamma=0.0,
                vega_1pct=0.0,
                vega_raw=0.0,
                theta_daily=0.0,
                theta_annualized=0.0,
                rho_1pct=0.0,
                rho_raw=0.0,
                vanna=0.0,
                volga=0.0,
                implied_volatility=volatility,
            )

        sqrt_t = math.sqrt(time_to_expiry)
        sigma_sqrt_t = volatility * sqrt_t

        d1 = (
            math.log(spot / strike)
            + (risk_free_rate - dividend_yield + 0.5 * volatility * volatility) * time_to_expiry
        ) / sigma_sqrt_t
        d2 = d1 - sigma_sqrt_t

        nd1 = float(st.norm.cdf(d1))
        nd2 = float(st.norm.cdf(d2))
        n_prime_d1 = float(st.norm.pdf(d1))

        disc_s = spot * math.exp(-dividend_yield * time_to_expiry)
        disc_k = strike * math.exp(-risk_free_rate * time_to_expiry)

        if is_call:
            price = disc_s * nd1 - disc_k * nd2
            delta = math.exp(-dividend_yield * time_to_expiry) * nd1
            theta_ann = (
                -(disc_s * n_prime_d1 * volatility) / (2.0 * sqrt_t)
                - risk_free_rate * disc_k * nd2
                + dividend_yield * disc_s * nd1
            )
            rho_raw = disc_k * time_to_expiry * nd2
        else:
            n_neg_d1 = float(st.norm.cdf(-d1))
            n_neg_d2 = float(st.norm.cdf(-d2))
            price = disc_k * n_neg_d2 - disc_s * n_neg_d1
            delta = -math.exp(-dividend_yield * time_to_expiry) * n_neg_d1
            theta_ann = (
                -(disc_s * n_prime_d1 * volatility) / (2.0 * sqrt_t)
                + risk_free_rate * disc_k * n_neg_d2
                - dividend_yield * disc_s * n_neg_d1
            )
            rho_raw = -disc_k * time_to_expiry * n_neg_d2

        gamma = (math.exp(-dividend_yield * time_to_expiry) * n_prime_d1) / (spot * sigma_sqrt_t)
        vega_raw = disc_s * sqrt_t * n_prime_d1
        vega_1pct = vega_raw * 0.01
        theta_daily = theta_ann / 365.0
        rho_1pct = rho_raw * 0.01

        # Higher-order Greeks
        vanna = -math.exp(-dividend_yield * time_to_expiry) * n_prime_d1 * (d2 / volatility)
        volga = vega_raw * (d1 * d2 / volatility)

        time_val = max(0.0, price - intrinsic)

        return OptionGreeks(
            price=max(0.0, price),
            intrinsic_value=intrinsic,
            time_value=time_val,
            delta=delta,
            gamma=gamma,
            vega_1pct=vega_1pct,
            vega_raw=vega_raw,
            theta_daily=theta_daily,
            theta_annualized=theta_ann,
            rho_1pct=rho_1pct,
            rho_raw=rho_raw,
            vanna=vanna,
            volga=volga,
            implied_volatility=volatility,
        )


class MertonJumpDiffusionEngine:
    """Merton (1976) Jump-Diffusion analytical options pricing engine.

    Models underlying dynamics as a continuous geometric Brownian motion with
    compound Poisson jumps having log-normal jump magnitudes.
    """

    @staticmethod
    def calculate_price(
        spot: float,
        strike: float,
        time_to_expiry: float,
        volatility: float,
        jump_intensity: float = 1.0,     # lambda: expected jumps per year
        jump_mean: float = -0.05,         # mu_J: mean log jump size
        jump_volatility: float = 0.15,    # delta_J: vol of log jump size
        risk_free_rate: float = 0.0,
        dividend_yield: float = 0.0,
        option_type: OptionType = OptionType.CALL,
        max_iterations: int = 35,
    ) -> float:
        """Compute Merton jump-diffusion European option price via Poisson series expansion."""
        if time_to_expiry <= 1e-7:
            intrinsic = max(0.0, (spot - strike) if option_type == OptionType.CALL else (strike - spot))
            return intrinsic

        kappa = math.exp(jump_mean + 0.5 * jump_volatility * jump_volatility) - 1.0
        lambda_prime = jump_intensity * (1.0 + kappa)

        total_price = 0.0
        log_fact = 0.0

        for k in range(max_iterations):
            if k > 0:
                log_fact += math.log(k)

            # Weight w_k = exp(-lambda' * T) * (lambda' * T)^k / k!
            log_weight = -lambda_prime * time_to_expiry + k * math.log(max(1e-12, lambda_prime * time_to_expiry)) - log_fact
            weight = math.exp(log_weight)

            vol_k = math.sqrt(volatility * volatility + (k * jump_volatility * jump_volatility) / time_to_expiry)
            r_k = risk_free_rate - jump_intensity * kappa + (k * (jump_mean + 0.5 * jump_volatility * jump_volatility)) / time_to_expiry

            greeks_k = BlackScholesEngine.calculate_greeks(
                spot=spot,
                strike=strike,
                time_to_expiry=time_to_expiry,
                volatility=vol_k,
                risk_free_rate=r_k,
                dividend_yield=dividend_yield,
                option_type=option_type,
            )

            total_price += weight * greeks_k.price

            # Stop when weights become negligible
            if weight < 1e-12 and k > 5:
                break

        return max(0.0, total_price)


class ImpliedVolatilitySolver:
    """High-performance Implied Volatility (IV) solver combining Newton-Raphson and Brent's method."""

    @staticmethod
    def solve_iv(
        market_price: float,
        spot: float,
        strike: float,
        time_to_expiry: float,
        risk_free_rate: float = 0.0,
        dividend_yield: float = 0.0,
        option_type: OptionType = OptionType.CALL,
        tolerance: float = 1e-6,
        max_iterations: int = 30,
    ) -> Optional[float]:
        """Solve for implied volatility matching market price."""
        if market_price <= 0.0 or time_to_expiry <= 1e-7 or spot <= 0.0 or strike <= 0.0:
            return None

        is_call = option_type == OptionType.CALL
        disc_s = spot * math.exp(-dividend_yield * time_to_expiry)
        disc_k = strike * math.exp(-risk_free_rate * time_to_expiry)
        lower_bound = max(0.0, (disc_s - disc_k) if is_call else (disc_k - disc_s))
        upper_bound = disc_s if is_call else disc_k

        if market_price < lower_bound - 1e-5 or market_price > upper_bound + 1e-5:
            # Arbitrage breach; no valid implied volatility
            return None

        # Closed-form initial guess via Corrado-Miller / Brenner-Subrahmanyam
        sigma = math.sqrt(2.0 * math.pi / time_to_expiry) * (market_price / spot)
        sigma = max(0.05, min(2.5, sigma))

        # Phase 1: Newton-Raphson with analytical Vega
        for _ in range(max_iterations):
            greeks = BlackScholesEngine.calculate_greeks(
                spot=spot,
                strike=strike,
                time_to_expiry=time_to_expiry,
                volatility=sigma,
                risk_free_rate=risk_free_rate,
                dividend_yield=dividend_yield,
                option_type=option_type,
            )
            diff = greeks.price - market_price

            if abs(diff) < tolerance:
                return float(sigma)

            vega = greeks.vega_raw
            if vega > 1e-8:
                sigma_next = sigma - diff / vega
                if 0.001 < sigma_next < 5.0:
                    sigma = sigma_next
                    continue

            # If vega too small or step goes out of bounds, break to Brent's method
            break

        # Phase 2: Brent's root-finding fallback
        def obj_func(vol: float) -> float:
            p = BlackScholesEngine.calculate_greeks(
                spot=spot,
                strike=strike,
                time_to_expiry=time_to_expiry,
                volatility=vol,
                risk_free_rate=risk_free_rate,
                dividend_yield=dividend_yield,
                option_type=option_type,
            ).price
            return p - market_price

        try:
            f_low = obj_func(1e-4)
            f_high = obj_func(6.0)
            if f_low * f_high > 0:
                return None
            iv_root = opt.brentq(obj_func, 1e-4, 6.0, xtol=tolerance, maxiter=50)
            return float(iv_root)
        except Exception as exc:
            logger.debug(f"Brent IV solve failed: {exc}")
            return None


class VolatilityPoint(BaseModel):
    """Observed market implied volatility data point."""

    strike: float
    time_to_expiry: float
    implied_vol: float


class VolatilitySurface:
    """Calibrated volatility smile and surface representation with interpolation."""

    def __init__(self, spot: float, quotes: Optional[List[VolatilityPoint]] = None):
        self.spot = spot
        self._quotes: List[VolatilityPoint] = quotes or []

    def add_point(self, strike: float, time_to_expiry: float, implied_vol: float) -> None:
        """Register an observed market IV point."""
        self._quotes.append(
            VolatilityPoint(strike=strike, time_to_expiry=time_to_expiry, implied_vol=implied_vol)
        )

    def get_volatility(self, strike: float, time_to_expiry: float) -> float:
        """Interpolate implied volatility using moneyness and total variance weighting."""
        if not self._quotes:
            return 0.30  # Default institutional prior

        if len(self._quotes) == 1:
            return self._quotes[0].implied_vol

        # Group by expiry or interpolate across (moneyness, expiry)
        moneyness = strike / self.spot
        pts = [(q.strike / self.spot, q.time_to_expiry, q.implied_vol) for q in self._quotes]

        # Inverse distance weighting on normalized log-moneyness and time coordinates
        log_m = math.log(max(1e-3, moneyness))
        t = max(1e-3, time_to_expiry)

        weights = []
        vols = []
        for q_m, q_t, q_v in pts:
            q_log_m = math.log(max(1e-3, q_m))
            dist_sq = (log_m - q_log_m) ** 2 + 0.5 * ((t - q_t) / max(0.1, t)) ** 2
            if dist_sq < 1e-7:
                return q_v
            w = 1.0 / dist_sq
            weights.append(w)
            vols.append(q_v)

        total_weight = sum(weights)
        if total_weight <= 0:
            return 0.30

        interp_vol = sum(w * v for w, v in zip(weights, vols)) / total_weight
        return float(interp_vol)

    def compute_skew_metrics(self, time_to_expiry: float = 30.0 / 365.0) -> Dict[str, float]:
        """Compute 25-delta put-call skew and ATM smile curvature."""
        atm_vol = self.get_volatility(self.spot, time_to_expiry)
        put_strike = self.spot * 0.90   # Proxy for ~25-delta put
        call_strike = self.spot * 1.10  # Proxy for ~25-delta call

        put_vol = self.get_volatility(put_strike, time_to_expiry)
        call_vol = self.get_volatility(call_strike, time_to_expiry)

        skew_25d = put_vol - call_vol
        convexity = (put_vol + call_vol) / 2.0 - atm_vol

        return {
            "atm_volatility": atm_vol,
            "put_volatility_90pct": put_vol,
            "call_volatility_110pct": call_vol,
            "volatility_skew_25d": skew_25d,
            "smile_convexity": convexity,
        }


class DerivativePosition(BaseModel):
    """Institutional option derivative contract position."""

    contract_id: str
    symbol: str
    option_type: OptionType
    strike: float
    time_to_expiry: float
    quantity: float
    mark_price: float
    underlying_price: float
    implied_volatility: float
    risk_free_rate: float = 0.03
    dividend_yield: float = 0.0


class DerivativePortfolioReport(BaseModel):
    """Aggregated portfolio-level Greeks and delta-neutral hedging guidance."""

    total_market_value_usd: float
    net_delta: float
    dollar_delta_usd: float
    net_gamma: float
    dollar_gamma_usd: float
    net_vega_1pct_usd: float
    net_theta_daily_usd: float
    net_rho_1pct_usd: float
    net_vanna_usd: float
    net_volga_usd: float
    delta_neutral_hedge_shares: float
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class DerivativesRiskEngine:
    """Aggregates derivative books into portfolio Greeks and hedging requirements."""

    def evaluate_portfolio_greeks(
        self, positions: List[DerivativePosition]
    ) -> DerivativePortfolioReport:
        """Calculate holistic Greek exposures and spot hedging requirements."""
        total_market_value = 0.0
        net_delta = 0.0
        dollar_delta = 0.0
        net_gamma = 0.0
        dollar_gamma = 0.0
        net_vega_1pct = 0.0
        net_theta_daily = 0.0
        net_rho_1pct = 0.0
        net_vanna = 0.0
        net_volga = 0.0

        for pos in positions:
            greeks = BlackScholesEngine.calculate_greeks(
                spot=pos.underlying_price,
                strike=pos.strike,
                time_to_expiry=pos.time_to_expiry,
                volatility=pos.implied_volatility,
                risk_free_rate=pos.risk_free_rate,
                dividend_yield=pos.dividend_yield,
                option_type=pos.option_type,
            )

            qty = pos.quantity
            pos_val = qty * pos.mark_price
            total_market_value += pos_val

            net_delta += qty * greeks.delta
            dollar_delta += qty * greeks.delta * pos.underlying_price
            net_gamma += qty * greeks.gamma
            dollar_gamma += 0.5 * qty * greeks.gamma * (pos.underlying_price ** 2) * 0.01  # 1% move dollar impact
            net_vega_1pct += qty * greeks.vega_1pct
            net_theta_daily += qty * greeks.theta_daily
            net_rho_1pct += qty * greeks.rho_1pct
            net_vanna += qty * greeks.vanna
            net_volga += qty * greeks.volga

        hedge_shares = -net_delta

        return DerivativePortfolioReport(
            total_market_value_usd=round(total_market_value, 2),
            net_delta=round(net_delta, 4),
            dollar_delta_usd=round(dollar_delta, 2),
            net_gamma=round(net_gamma, 6),
            dollar_gamma_usd=round(dollar_gamma, 2),
            net_vega_1pct_usd=round(net_vega_1pct, 2),
            net_theta_daily_usd=round(net_theta_daily, 2),
            net_rho_1pct_usd=round(net_rho_1pct, 2),
            net_vanna_usd=round(net_vanna, 4),
            net_volga_usd=round(net_volga, 4),
            delta_neutral_hedge_shares=round(hedge_shares, 4),
        )
