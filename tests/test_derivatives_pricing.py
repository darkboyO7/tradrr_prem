"""Tests for Derivatives Pricing, Greeks, Implied Volatility & Surface Engine."""

import math
import pytest

from analysis.derivatives import (
    BlackScholesEngine,
    DerivativePosition,
    DerivativePortfolioReport,
    DerivativesRiskEngine,
    ImpliedVolatilitySolver,
    MertonJumpDiffusionEngine,
    OptionGreeks,
    OptionType,
    VolatilityPoint,
    VolatilitySurface,
)


def test_black_scholes_call_and_put_parity():
    """Verify standard Put-Call parity relationship: C - P = S*e^(-qT) - K*e^(-rT)."""
    spot = 65000.0
    strike = 64000.0
    time_to_expiry = 0.5  # 6 months
    vol = 0.40
    rate = 0.05
    div = 0.01

    call_greeks = BlackScholesEngine.calculate_greeks(
        spot=spot,
        strike=strike,
        time_to_expiry=time_to_expiry,
        volatility=vol,
        risk_free_rate=rate,
        dividend_yield=div,
        option_type=OptionType.CALL,
    )

    put_greeks = BlackScholesEngine.calculate_greeks(
        spot=spot,
        strike=strike,
        time_to_expiry=time_to_expiry,
        volatility=vol,
        risk_free_rate=rate,
        dividend_yield=div,
        option_type=OptionType.PUT,
    )

    lhs = call_greeks.price - put_greeks.price
    rhs = spot * math.exp(-div * time_to_expiry) - strike * math.exp(-rate * time_to_expiry)

    assert math.isclose(lhs, rhs, rel_tol=1e-5)
    assert call_greeks.delta > 0.0 and call_greeks.delta < 1.0
    assert put_greeks.delta < 0.0 and put_greeks.delta > -1.0
    assert math.isclose(call_greeks.gamma, put_greeks.gamma, rel_tol=1e-6)
    assert math.isclose(call_greeks.vega_raw, put_greeks.vega_raw, rel_tol=1e-6)


def test_black_scholes_greeks_finite_difference_consistency():
    """Verify analytical Greeks against numerical central finite differences."""
    spot = 60000.0
    strike = 60000.0
    t = 0.25
    vol = 0.50
    rate = 0.03
    div = 0.0

    base = BlackScholesEngine.calculate_greeks(
        spot=spot, strike=strike, time_to_expiry=t, volatility=vol, risk_free_rate=rate, dividend_yield=div
    )

    eps_s = 0.50
    up_s = BlackScholesEngine.calculate_greeks(
        spot=spot + eps_s, strike=strike, time_to_expiry=t, volatility=vol, risk_free_rate=rate, dividend_yield=div
    )
    down_s = BlackScholesEngine.calculate_greeks(
        spot=spot - eps_s, strike=strike, time_to_expiry=t, volatility=vol, risk_free_rate=rate, dividend_yield=div
    )

    num_delta = (up_s.price - down_s.price) / (2.0 * eps_s)
    num_gamma = (up_s.price - 2.0 * base.price + down_s.price) / (eps_s ** 2)

    assert math.isclose(base.delta, num_delta, rel_tol=1e-3)
    assert math.isclose(base.gamma, num_gamma, rel_tol=1e-3)

    eps_vol = 0.001
    up_vol = BlackScholesEngine.calculate_greeks(
        spot=spot, strike=strike, time_to_expiry=t, volatility=vol + eps_vol, risk_free_rate=rate, dividend_yield=div
    )
    down_vol = BlackScholesEngine.calculate_greeks(
        spot=spot, strike=strike, time_to_expiry=t, volatility=vol - eps_vol, risk_free_rate=rate, dividend_yield=div
    )

    num_vega_raw = (up_vol.price - down_vol.price) / (2.0 * eps_vol)
    assert math.isclose(base.vega_raw, num_vega_raw, rel_tol=1e-3)


def test_black_scholes_boundary_conditions():
    """Verify expired options and near-zero volatility convergence."""
    # Expired Call
    exp_call = BlackScholesEngine.calculate_greeks(
        spot=65000.0, strike=60000.0, time_to_expiry=0.0, volatility=0.5
    )
    assert exp_call.price == 5000.0
    assert exp_call.delta == 1.0
    assert exp_call.gamma == 0.0

    # Expired OTM Put
    exp_put = BlackScholesEngine.calculate_greeks(
        spot=65000.0, strike=60000.0, time_to_expiry=0.0, volatility=0.5, option_type=OptionType.PUT
    )
    assert exp_put.price == 0.0
    assert exp_put.delta == 0.0


def test_merton_jump_diffusion_behavior():
    """Verify Merton jump-diffusion matches BSM when jump intensity is zero."""
    spot = 50000.0
    strike = 50000.0
    t = 0.5
    vol = 0.35
    rate = 0.02

    bsm_price = BlackScholesEngine.calculate_greeks(
        spot=spot, strike=strike, time_to_expiry=t, volatility=vol, risk_free_rate=rate
    ).price

    merton_zero_jump = MertonJumpDiffusionEngine.calculate_price(
        spot=spot,
        strike=strike,
        time_to_expiry=t,
        volatility=vol,
        jump_intensity=0.0,
        risk_free_rate=rate,
    )

    assert math.isclose(bsm_price, merton_zero_jump, rel_tol=1e-4)

    # With jump intensity, tail risk increases out-of-the-money option prices
    otm_bsm = BlackScholesEngine.calculate_greeks(
        spot=spot, strike=65000.0, time_to_expiry=t, volatility=vol, risk_free_rate=rate
    ).price
    otm_merton = MertonJumpDiffusionEngine.calculate_price(
        spot=spot,
        strike=65000.0,
        time_to_expiry=t,
        volatility=vol,
        jump_intensity=2.0,
        jump_mean=-0.05,
        jump_volatility=0.20,
        risk_free_rate=rate,
    )
    assert otm_merton > otm_bsm


def test_implied_volatility_solver_round_trip():
    """Verify IV solver accurately recovers known volatility for Call and Put."""
    spot = 62000.0
    strike = 60000.0
    t = 0.33
    true_vol = 0.62
    rate = 0.04

    # Call IV recovery
    call_price = BlackScholesEngine.calculate_greeks(
        spot=spot, strike=strike, time_to_expiry=t, volatility=true_vol, risk_free_rate=rate, option_type=OptionType.CALL
    ).price

    solved_iv_call = ImpliedVolatilitySolver.solve_iv(
        market_price=call_price,
        spot=spot,
        strike=strike,
        time_to_expiry=t,
        risk_free_rate=rate,
        option_type=OptionType.CALL,
    )
    assert solved_iv_call is not None
    assert math.isclose(solved_iv_call, true_vol, rel_tol=1e-4)

    # Put IV recovery
    put_price = BlackScholesEngine.calculate_greeks(
        spot=spot, strike=strike, time_to_expiry=t, volatility=true_vol, risk_free_rate=rate, option_type=OptionType.PUT
    ).price

    solved_iv_put = ImpliedVolatilitySolver.solve_iv(
        market_price=put_price,
        spot=spot,
        strike=strike,
        time_to_expiry=t,
        risk_free_rate=rate,
        option_type=OptionType.PUT,
    )
    assert solved_iv_put is not None
    assert math.isclose(solved_iv_put, true_vol, rel_tol=1e-4)

    # Arbitrage breach should return None
    invalid_iv = ImpliedVolatilitySolver.solve_iv(
        market_price=spot * 2.0,  # Impossible option price
        spot=spot,
        strike=strike,
        time_to_expiry=t,
    )
    assert invalid_iv is None


def test_volatility_surface_and_skew():
    """Verify calibrated surface interpolation and skew computation."""
    spot = 65000.0
    surface = VolatilitySurface(spot=spot)

    # Add smile points (higher vol on out-of-the-money puts)
    surface.add_point(strike=55000.0, time_to_expiry=0.082, implied_vol=0.68)
    surface.add_point(strike=65000.0, time_to_expiry=0.082, implied_vol=0.52)
    surface.add_point(strike=75000.0, time_to_expiry=0.082, implied_vol=0.56)

    # Interpolate ATM
    atm_vol = surface.get_volatility(strike=65000.0, time_to_expiry=0.082)
    assert math.isclose(atm_vol, 0.52, rel_tol=1e-3)

    metrics = surface.compute_skew_metrics(time_to_expiry=0.082)
    assert "atm_volatility" in metrics
    assert "volatility_skew_25d" in metrics
    assert "smile_convexity" in metrics
    assert metrics["smile_convexity"] > 0.0  # Smile curvature is positive


def test_derivatives_risk_portfolio_aggregation():
    """Verify portfolio Greek aggregation and delta-neutral hedging."""
    engine = DerivativesRiskEngine()
    positions = [
        DerivativePosition(
            contract_id="BTC-260327-70000-C",
            symbol="BTC-USD",
            option_type=OptionType.CALL,
            strike=70000.0,
            time_to_expiry=0.25,
            quantity=10.0,
            mark_price=4200.0,
            underlying_price=68000.0,
            implied_volatility=0.55,
        ),
        DerivativePosition(
            contract_id="BTC-260327-60000-P",
            symbol="BTC-USD",
            option_type=OptionType.PUT,
            strike=60000.0,
            time_to_expiry=0.25,
            quantity=5.0,
            mark_price=1800.0,
            underlying_price=68000.0,
            implied_volatility=0.60,
        ),
    ]

    report = engine.evaluate_portfolio_greeks(positions)
    assert isinstance(report, DerivativePortfolioReport)
    assert report.total_market_value_usd == round(10.0 * 4200.0 + 5.0 * 1800.0, 2)
    assert report.net_gamma > 0.0  # Net long options -> positive gamma
    assert report.net_vega_1pct_usd > 0.0  # Net long options -> positive vega
    assert report.net_theta_daily_usd < 0.0  # Net long options -> negative daily theta decay
    # Delta-neutral hedge shares is the exact negative of net delta
    assert math.isclose(report.delta_neutral_hedge_shares, -report.net_delta, abs_tol=1e-4)
