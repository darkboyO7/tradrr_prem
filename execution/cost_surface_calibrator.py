"""Trading Intelligence Platform - Real-Time Slippage & Liquidity Cost Surface Calibration Engine.

Dynamic empirical recalibration of Almgren-Chriss temporary (eta) and permanent (gamma)
market impact coefficients from post-trade execution telemetry, multi-dimensional
liquidity cost surface modeling, and OFI-conditioned slippage forecasting.
"""

from datetime import datetime, timezone
import math
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from pydantic import BaseModel, Field

from core.logging import get_logger
from execution.tca import TradeExecutionRecord

logger = get_logger("cost_surface_calibrator")


class CalibratedCostParameters(BaseModel):
    """Calibrated empirical Almgren-Chriss impact and liquidity elasticity parameters."""

    symbol: str
    venue: str
    calibrated_eta: float = 0.142               # Temporary impact coefficient
    calibrated_gamma: float = 0.314             # Permanent impact coefficient
    alpha_exponent: float = 0.50                # Permanent impact power law exponent
    beta_exponent: float = 0.50                 # Temporary impact power law exponent
    ofi_sensitivity_kappa: float = 0.35         # Asymmetric OFI impact multiplier
    r_squared: float = 0.0                      # Goodness-of-fit of empirical calibration
    sample_trades_count: int = 0
    calibrated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class CostSurfacePoint(BaseModel):
    """Specific coordinate on the multi-dimensional liquidity cost surface."""

    participation_rate: float
    volatility_annualized: float
    expected_slippage_bps: float
    expected_market_impact_usd: float


class CostSurfaceGrid(BaseModel):
    """Discrete 2D/3D cost surface matrix parameterized across participation rates and volatility."""

    symbol: str
    venue: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    participation_bins: List[float] = []        # e.g. [0.01, 0.05, 0.10, 0.20, 0.35, 0.50]
    volatility_bins: List[float] = []           # e.g. [0.20, 0.40, 0.60, 0.80, 1.00, 1.20]
    surface_matrix_bps: List[List[float]] = []  # [vol_idx][part_idx] in bps
    parameters: CalibratedCostParameters


class SlippagePredictionRequest(BaseModel):
    """Input parameters for pre-trade slippage and market impact prediction."""

    symbol: str
    venue: str = "coinbase"
    side: str = "BUY"  # "BUY" or "SELL"
    order_quantity: float
    order_duration_seconds: float = 60.0
    reference_price: float
    market_volume_rate_per_sec: float = 2.0  # Prevailing market volume rate (units/sec)
    realized_volatility: float = 0.50        # Annualized volatility (e.g. 50%)
    order_flow_imbalance: float = 0.0        # Normalized OFI in [-1.0, 1.0]
    half_spread_bps: float = 2.0             # Quoted half spread in basis points


class SlippagePredictionResult(BaseModel):
    """Pre-trade execution cost forecast with impact breakdown and recommended urgency."""

    symbol: str
    venue: str
    side: str
    participation_rate: float
    expected_temporary_impact_bps: float
    expected_permanent_impact_bps: float
    expected_half_spread_bps: float
    expected_total_slippage_bps: float
    expected_total_slippage_usd: float
    confidence_score: float
    recommended_urgency: str  # "PASSIVE", "NORMAL", "AGGRESSIVE"
    parameters_used: CalibratedCostParameters


class CostSurfaceCalibrator:
    """Continuous statistical calibrator for institutional market impact and slippage surfaces."""

    DEFAULT_PARTICIPATION_BINS = [0.01, 0.025, 0.05, 0.10, 0.15, 0.25, 0.35, 0.50]
    DEFAULT_VOLATILITY_BINS = [0.20, 0.35, 0.50, 0.70, 0.90, 1.20]

    def __init__(
        self,
        default_eta: float = 0.142,
        default_gamma: float = 0.314,
        min_samples_for_recal: int = 5,
    ) -> None:
        self.default_eta = default_eta
        self.default_gamma = default_gamma
        self.min_samples_for_recal = min_samples_for_recal

        # Internal trade observation storage: (symbol, venue) -> list of trade observation tuples
        # Tuple: (participation_rate, volatility, realized_temp_impact_bps, realized_perm_impact_bps, notional)
        self._trade_records: Dict[Tuple[str, str], List[Tuple[float, float, float, float, float]]] = {}
        # Active calibrated parameters cache: (symbol, venue) -> CalibratedCostParameters
        self._calibrated_params: Dict[Tuple[str, str], CalibratedCostParameters] = {}

    def ingest_trade_execution(
        self,
        trade: TradeExecutionRecord,
        realized_volatility: float = 0.50,
        market_interval_volume: Optional[float] = None,
    ) -> None:
        """Extract empirical impact observations from a completed post-trade execution record."""
        key = (trade.symbol.upper(), trade.venue.lower())
        if key not in self._trade_records:
            self._trade_records[key] = []

        # Compute participation rate
        interval_vol = market_interval_volume or max(trade.filled_quantity * 5.0, 10.0)
        participation = min(0.99, max(0.001, trade.filled_quantity / interval_vol))

        # Realized impact in bps
        arrival = trade.arrival_price
        fill_px = trade.execution_price
        terminal = trade.terminal_price if trade.terminal_price is not None else fill_px

        direction = 1.0 if trade.side.upper() == "BUY" else -1.0
        # Temporary impact: difference between execution price and terminal price
        temp_diff = (fill_px - terminal) * direction
        temp_bps = max(0.0, (temp_diff / arrival) * 10000.0)

        # Permanent impact: shift in midpoint from arrival to terminal
        perm_diff = (terminal - arrival) * direction
        perm_bps = max(0.0, (perm_diff / arrival) * 10000.0)

        notional = trade.filled_quantity * trade.execution_price
        self._trade_records[key].append((participation, realized_volatility, temp_bps, perm_bps, notional))

        logger.info(
            f"Ingested trade observation for {key}: part={participation:.3f}, "
            f"temp_bps={temp_bps:.1f}, perm_bps={perm_bps:.1f}, notional=${notional:,.0f}"
        )

    def calibrate(self, symbol: str, venue: str = "coinbase") -> CalibratedCostParameters:
        """Empirically fit eta and gamma coefficients using OLS power-law regression."""
        key = (symbol.upper(), venue.lower())
        records = self._trade_records.get(key, [])

        if len(records) < self.min_samples_for_recal:
            # Insufficient trade observations: return prior default parameters
            params = CalibratedCostParameters(
                symbol=symbol,
                venue=venue,
                calibrated_eta=self.default_eta,
                calibrated_gamma=self.default_gamma,
                r_squared=0.0,
                sample_trades_count=len(records),
            )
            self._calibrated_params[key] = params
            return params

        # Log-linear regression: ln(Impact / (sigma * 10000)) = ln(coeff) + beta * ln(participation)
        part_arr = np.array([r[0] for r in records])
        vol_arr = np.array([r[1] for r in records])
        temp_bps_arr = np.array([max(0.1, r[2]) for r in records])
        perm_bps_arr = np.array([max(0.1, r[3]) for r in records])

        # Regression for Temporary Impact (eta)
        log_x = np.log(part_arr)
        norm_temp = temp_bps_arr / np.maximum(0.01, vol_arr * 100.0)
        log_y_temp = np.log(norm_temp)

        # OLS slope and intercept
        A = np.vstack([log_x, np.ones(len(log_x))]).T
        slope_temp, log_eta = np.linalg.lstsq(A, log_y_temp, rcond=None)[0]
        calibrated_eta = float(np.clip(np.exp(log_eta), 0.02, 1.50))
        beta_exp = float(np.clip(slope_temp, 0.20, 0.90))

        # Regression for Permanent Impact (gamma)
        norm_perm = perm_bps_arr / np.maximum(0.01, vol_arr * 100.0)
        log_y_perm = np.log(norm_perm)
        slope_perm, log_gamma = np.linalg.lstsq(A, log_y_perm, rcond=None)[0]
        calibrated_gamma = float(np.clip(np.exp(log_gamma), 0.05, 2.00))
        alpha_exp = float(np.clip(slope_perm, 0.20, 0.90))

        # Compute R-squared on temporary fit
        y_pred = slope_temp * log_x + log_eta
        ss_res = np.sum((log_y_temp - y_pred) ** 2)
        ss_tot = np.sum((log_y_temp - np.mean(log_y_temp)) ** 2)
        r2 = float(max(0.0, 1.0 - (ss_res / max(1e-6, ss_tot))))

        params = CalibratedCostParameters(
            symbol=symbol,
            venue=venue,
            calibrated_eta=round(calibrated_eta, 4),
            calibrated_gamma=round(calibrated_gamma, 4),
            alpha_exponent=round(alpha_exp, 3),
            beta_exponent=round(beta_exp, 3),
            ofi_sensitivity_kappa=0.35,
            r_squared=round(r2, 3),
            sample_trades_count=len(records),
            calibrated_at=datetime.now(timezone.utc),
        )
        self._calibrated_params[key] = params
        logger.info(
            f"Calibrated Almgren-Chriss parameters for {key}: eta={calibrated_eta:.4f}, "
            f"gamma={calibrated_gamma:.4f}, R2={r2:.2f}, samples={len(records)}"
        )
        return params

    def get_parameters(self, symbol: str, venue: str = "coinbase") -> CalibratedCostParameters:
        """Retrieve active parameters for symbol/venue, triggering calibration if uncalibrated."""
        key = (symbol.upper(), venue.lower())
        if key not in self._calibrated_params:
            return self.calibrate(symbol, venue)
        return self._calibrated_params[key]

    def predict_slippage(self, req: SlippagePredictionRequest) -> SlippagePredictionResult:
        """Compute expected basis point and dollar slippage breakdown for a proposed order."""
        params = self.get_parameters(req.symbol, req.venue)

        # Slicing rate (units / sec)
        order_rate = req.order_quantity / max(1.0, req.order_duration_seconds)
        participation = min(0.99, max(0.001, order_rate / max(0.01, req.market_volume_rate_per_sec)))

        # Almgren-Chriss formulation:
        # Temp impact (bps) = eta * sigma * (pov ** beta) * 100
        # Perm impact (bps) = gamma * sigma * (pov ** alpha) * 100
        vol = max(0.05, req.realized_volatility)
        raw_temp_bps = params.calibrated_eta * vol * 100.0 * (participation ** params.beta_exponent)
        raw_perm_bps = params.calibrated_gamma * vol * 100.0 * (participation ** params.alpha_exponent)

        # Microstructure OFI asymmetry adjustment:
        # If buying and OFI > 0 (demand pressure), slippage is magnified. If OFI < 0, liquidity is abundant.
        side_mult = 1.0 if req.side.upper() == "BUY" else -1.0
        ofi_factor = 1.0 + params.ofi_sensitivity_kappa * side_mult * req.order_flow_imbalance
        ofi_factor = max(0.50, min(2.0, ofi_factor))

        adj_temp_bps = raw_temp_bps * ofi_factor
        adj_perm_bps = raw_perm_bps * ofi_factor
        half_spread_bps = max(0.5, req.half_spread_bps)

        total_slippage_bps = half_spread_bps + adj_temp_bps + 0.5 * adj_perm_bps
        notional_usd = req.reference_price * req.order_quantity
        total_slippage_usd = (total_slippage_bps / 10000.0) * notional_usd

        # Recommended execution urgency
        if total_slippage_bps < 10.0:
            urgency = "AGGRESSIVE"  # Low market impact, safe to take liquidity rapidly
        elif total_slippage_bps < 35.0:
            urgency = "NORMAL"
        else:
            urgency = "PASSIVE"     # High market impact, slice conservatively with TWAP/VWAP

        # Confidence based on historical sample size
        n_samples = params.sample_trades_count
        conf = min(0.95, 0.40 + 0.10 * min(5, n_samples) + 0.15 * params.r_squared)

        return SlippagePredictionResult(
            symbol=req.symbol,
            venue=req.venue,
            side=req.side.upper(),
            participation_rate=round(participation, 4),
            expected_temporary_impact_bps=round(adj_temp_bps, 2),
            expected_permanent_impact_bps=round(adj_perm_bps, 2),
            expected_half_spread_bps=round(half_spread_bps, 2),
            expected_total_slippage_bps=round(total_slippage_bps, 2),
            expected_total_slippage_usd=round(total_slippage_usd, 2),
            confidence_score=round(conf, 3),
            recommended_urgency=urgency,
            parameters_used=params,
        )

    def generate_cost_surface(
        self,
        symbol: str,
        venue: str = "coinbase",
        participation_bins: Optional[List[float]] = None,
        volatility_bins: Optional[List[float]] = None,
    ) -> CostSurfaceGrid:
        """Synthesize discrete 2D matrix of expected slippage across participation and volatility coordinates."""
        params = self.get_parameters(symbol, venue)
        p_bins = participation_bins or self.DEFAULT_PARTICIPATION_BINS
        v_bins = volatility_bins or self.DEFAULT_VOLATILITY_BINS

        matrix: List[List[float]] = []
        for vol in v_bins:
            row: List[float] = []
            for pov in p_bins:
                temp_bps = params.calibrated_eta * vol * 100.0 * (pov ** params.beta_exponent)
                perm_bps = params.calibrated_gamma * vol * 100.0 * (pov ** params.alpha_exponent)
                half_spread = 2.0
                total_bps = half_spread + temp_bps + 0.5 * perm_bps
                row.append(round(total_bps, 2))
            matrix.append(row)

        return CostSurfaceGrid(
            symbol=symbol,
            venue=venue,
            participation_bins=p_bins,
            volatility_bins=v_bins,
            surface_matrix_bps=matrix,
            parameters=params,
        )
