"""Tests for Cross-Asset Dynamic Yield Curve & Term Structure Calibrator."""

import math
import numpy as np
import pytest

from analysis.yield_curve import (
    CalibratedCurveParameters,
    CalibratedYieldCurveReport,
    CurveModelType,
    TermStructureCalibrator,
    TermStructureRollDownReport,
    YieldTenorPoint,
)


@pytest.fixture
def sample_yield_curve_data():
    """Observed standard crypto funding & basis term structure points (1m to 5y)."""
    # Upward sloping normal crypto basis curve: 4% at 1m to 9% at 5y
    return [
        YieldTenorPoint(maturity_years=1.0 / 12.0, yield_rate=0.040),  # 1m
        YieldTenorPoint(maturity_years=3.0 / 12.0, yield_rate=0.048),  # 3m
        YieldTenorPoint(maturity_years=6.0 / 12.0, yield_rate=0.055),  # 6m
        YieldTenorPoint(maturity_years=1.0, yield_rate=0.065),         # 1y
        YieldTenorPoint(maturity_years=2.0, yield_rate=0.075),         # 2y
        YieldTenorPoint(maturity_years=3.0, yield_rate=0.082),         # 3y
        YieldTenorPoint(maturity_years=5.0, yield_rate=0.090),         # 5y
    ]


def test_nelson_siegel_calibration(sample_yield_curve_data):
    """Test 4-parameter Nelson-Siegel calibration on term structure."""
    calibrator = TermStructureCalibrator()
    params = calibrator.fit_curve(sample_yield_curve_data, model_type=CurveModelType.NELSON_SIEGEL)

    assert isinstance(params, CalibratedCurveParameters)
    assert params.model_type == CurveModelType.NELSON_SIEGEL
    assert params.beta_3 == 0.0
    assert params.tau_1 > 0.0
    assert params.rmse_bps < 50.0  # Good fit within 50 bps

    # Verify spot rates at 1y and 5y interpolate smoothly
    y_1y = calibrator.evaluate_spot_rate(params, 1.0)
    y_5y = calibrator.evaluate_spot_rate(params, 5.0)
    assert 0.05 <= y_1y <= 0.08
    assert y_5y > y_1y  # Upward sloping

    # Discount factor D(m) must be strictly between 0 and 1 and monotonically decreasing
    d_1y = calibrator.evaluate_discount_factor(params, 1.0)
    d_5y = calibrator.evaluate_discount_factor(params, 5.0)
    assert 0.0 < d_5y < d_1y < 1.0


def test_nelson_siegel_svensson_calibration(sample_yield_curve_data):
    """Test 6-parameter Svensson extension capturing curvature humps."""
    calibrator = TermStructureCalibrator()
    params = calibrator.fit_curve(sample_yield_curve_data, model_type=CurveModelType.NELSON_SIEGEL_SVENSSON)

    assert params.model_type == CurveModelType.NELSON_SIEGEL_SVENSSON
    assert params.tau_1 > 0.0
    assert params.tau_2 > 0.0
    assert params.rmse_bps <= 35.0  # Typically lower RMSE than standard NS


def test_full_curve_report_and_roll_down(sample_yield_curve_data):
    """Test comprehensive term structure report with carry and roll-down decomposition."""
    calibrator = TermStructureCalibrator()
    params = calibrator.fit_curve(sample_yield_curve_data)

    report = calibrator.generate_full_curve_report(underlying="BTC", params=params)
    assert isinstance(report, CalibratedYieldCurveReport)
    assert report.underlying == "BTC"
    assert report.curve_shape_summary in ("NORMAL_UPWARD", "HUMPED", "FLAT")
    assert len(report.tenor_points_evaluated) >= 6

    # Verify roll-down mechanics: on an upward sloping curve, roll-down gain should be positive
    t_2y = next(t for t in report.tenor_points_evaluated if math.isclose(t.tenor_years, 2.0, rel_tol=1e-2))
    assert t_2y.modified_duration > 0.0
    assert t_2y.convexity > 0.0
    assert t_2y.carry_yield_bps > 0.0
    # Positive roll down gain from declining yield as maturity approaches
    assert t_2y.roll_down_gain_bps >= 0.0


def test_instantaneous_forward_rate_consistency(sample_yield_curve_data):
    """Verify mathematical relationship: f(0) == y(0) == beta0 + beta1."""
    calibrator = TermStructureCalibrator()
    params = calibrator.fit_curve(sample_yield_curve_data)

    # At short end limit m -> 0: spot and forward should converge to beta_0 + beta_1
    spot_zero = calibrator.evaluate_spot_rate(params, 0.0001)
    fwd_zero = calibrator.evaluate_instantaneous_forward(params, 0.0001)
    theoretical_zero = params.beta_0 + params.beta_1

    assert spot_zero == pytest.approx(theoretical_zero, rel=1e-2)
    assert fwd_zero == pytest.approx(theoretical_zero, rel=1e-2)


def test_input_validation_and_serialization(sample_yield_curve_data):
    """Test input errors and dictionary serialization."""
    calibrator = TermStructureCalibrator()

    with pytest.raises(ValueError, match="At least 3"):
        calibrator.fit_curve(sample_yield_curve_data[:2])

    with pytest.raises(ValueError, match="strictly positive"):
        calibrator.fit_curve([
            YieldTenorPoint(maturity_years=-0.5, yield_rate=0.05),
            YieldTenorPoint(maturity_years=1.0, yield_rate=0.06),
            YieldTenorPoint(maturity_years=2.0, yield_rate=0.07),
        ])

    params = calibrator.fit_curve(sample_yield_curve_data)
    rep = calibrator.generate_full_curve_report("ETH", params)
    d = rep.to_dict()
    assert d["underlying"] == "ETH"
    assert "parameters" in d
    assert "tenor_points" in d
    assert d["parameters"]["rmse_bps"] >= 0.0
