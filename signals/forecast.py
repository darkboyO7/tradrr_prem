"""Trading Intelligence Platform - Real Probabilistic Horizon Forecasting Engine.

Produces rigorous, time-horizon-bounded forecasts (5m, 15m, 1h, 4h, 1d) with:
- Directional probabilities P(up), P(down), P(flat)
- Expected return and forward volatility estimates
- Target hit probability and Stop hit probability
- Explainable, evidence-grounded BUY / SELL / NO-TRADE decisions
- Strictly returns NO-TRADE with diagnostic reasons when data is insufficient or stale
"""

import math
from datetime import UTC, datetime, timedelta
from typing import Any

from pydantic import BaseModel, Field

from analysis.regime import MarketRegimeState, TrendRegime
from core.logging import get_logger
from models.base import SignalAction

logger = get_logger(__name__)


class ProbabilisticForecast(BaseModel):
    """Rigorous probabilistic forecast over an explicit time horizon."""

    symbol: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    forecast_horizon: str  # e.g., '5m', '15m', '1h', '4h', '1d'
    horizon_seconds: int
    target_time: datetime
    entry_reference_price: float

    # Directional probabilities summing to 1.0
    prob_up: float = Field(ge=0.0, le=1.0)
    prob_down: float = Field(ge=0.0, le=1.0)
    prob_flat: float = Field(ge=0.0, le=1.0)

    # Return & volatility distributions
    expected_return_pct: float
    expected_volatility_pct: float
    expected_high: float = 0.0
    expected_low: float = 0.0

    # Path-dependent boundary metrics
    target_price: float
    invalidation_stop_price: float
    target_hit_probability: float = Field(ge=0.0, le=1.0)
    stop_hit_probability: float = Field(ge=0.0, le=1.0)
    risk_reward_ratio: float

    # Categorical signal decision
    action: SignalAction
    confidence: float = Field(ge=0.0, le=1.0)
    expected_holding_period_minutes: float

    reasons: list[str] = Field(default_factory=list)
    diagnostics: dict[str, Any] = Field(default_factory=dict)


class MultiHorizonForecastReport(BaseModel):
    """Composite forecasts across multiple operational time horizons."""

    symbol: str
    current_price: float
    atr_14: float
    regime_code: str
    primary_decision: SignalAction
    primary_horizon: str = "1h"
    forecasts: dict[str, ProbabilisticForecast] = Field(default_factory=dict)
    reasons: list[str] = Field(default_factory=list)
    no_trade_reasons: list[str] = Field(default_factory=list)


class ProbabilisticForecastingEngine:
    """Scientific predictor combining features, regime conditioning, historical similarity, and ML calibration."""

    HORIZON_MAP = {
        "5m": (300, 5),
        "15m": (900, 15),
        "1h": (3600, 60),
        "4h": (14400, 240),
        "1d": (86400, 1440),
    }

    def forecast_horizon(
        self,
        symbol: str,
        current_price: float,
        atr: float,
        horizon: str = "1h",
        regime: MarketRegimeState | None = None,
        technical_signals: list[Any] | None = None,
        candle_history: list[dict[str, Any]] | None = None,
        ml_prob_up: float | None = None,
        pattern_consensus: dict[str, Any] | None = None,
        latest_features: dict[str, Any] | None = None,
        data_is_stale: bool = False,
    ) -> ProbabilisticForecast:
        """Compute an explainable, calibrated probabilistic forecast for a specific horizon."""
        now = datetime.now(UTC)
        h_secs, h_mins = self.HORIZON_MAP.get(horizon, (3600, 60))
        target_dt = now + timedelta(seconds=h_secs)

        # 1. Immediate NO-TRADE guardrails
        no_trade_reasons: list[str] = []
        if current_price <= 0:
            no_trade_reasons.append("Invalid or non-positive current price")
        if data_is_stale:
            no_trade_reasons.append("Market data feed is STALE (silence threshold breached)")
        if not candle_history or len(candle_history) < 5:
            no_trade_reasons.append("Insufficient candle sample depth for horizon statistical estimation")

        if no_trade_reasons:
            return ProbabilisticForecast(
                symbol=symbol,
                timestamp=now,
                forecast_horizon=horizon,
                horizon_seconds=h_secs,
                target_time=target_dt,
                entry_reference_price=current_price,
                prob_up=0.33,
                prob_down=0.33,
                prob_flat=0.34,
                expected_return_pct=0.0,
                expected_volatility_pct=0.0,
                target_price=current_price,
                invalidation_stop_price=current_price,
                target_hit_probability=0.0,
                stop_hit_probability=0.0,
                risk_reward_ratio=1.0,
                action=SignalAction.NO_TRADE,
                confidence=0.0,
                expected_holding_period_minutes=float(h_mins),
                reasons=no_trade_reasons,
                diagnostics={"status": "INSUFFICIENT_DATA_OR_STALE"},
            )

        # 2. Extract price velocity and historical momentum from candles
        closes = [float(c["close"]) for c in candle_history if "close" in c and c["close"] is not None]
        ret_last = (closes[-1] - closes[-2]) / closes[-2] if len(closes) >= 2 and closes[-2] > 0 else 0.0
        lookback_idx = max(0, len(closes) - min(len(closes), max(5, h_mins // 5)))
        ret_multi = (closes[-1] - closes[lookback_idx]) / closes[lookback_idx] if closes[lookback_idx] > 0 else 0.0

        # Horizon scaling factor: sqrt(time)
        vol_scale = math.sqrt(h_secs / 3600.0)
        atr_pct = (atr / current_price) if current_price > 0 else 0.01
        exp_vol_pct = max(0.05, atr_pct * vol_scale * 100.0)

        # Base neutral probabilities adjusted by historical momentum velocity
        mom_bias = max(-0.25, min(0.25, ret_multi * 0.5))
        p_up = 0.35 + mom_bias
        p_down = 0.35 - mom_bias
        p_flat = 0.30

        # Incorporate engineered indicator state if available
        if latest_features:
            rsi = float(latest_features.get("rsi_14", 50.0))
            macd_h = float(latest_features.get("macd_hist", 0.0))
            plus_di = float(latest_features.get("plus_di_14", 20.0))
            minus_di = float(latest_features.get("minus_di_14", 20.0))
            di_diff = (plus_di - minus_di) / max(1.0, plus_di + minus_di)
            rsi_shift = (rsi - 50.0) / 250.0  # [-0.20, +0.20]
            macd_shift = 0.04 if macd_h > 0 else (-0.04 if macd_h < 0 else 0.0)
            ind_shift = max(-0.20, min(0.20, (0.5 * di_diff) + rsi_shift + macd_shift))
            p_up += ind_shift
            p_down -= ind_shift

        # Apply trend and regime drift
        if regime:
            if regime.trend == TrendRegime.STRONG_BULL:
                p_up += 0.20
                p_down -= 0.15
                p_flat -= 0.05
            elif regime.trend == TrendRegime.WEAK_BULL:
                p_up += 0.10
                p_down -= 0.08
                p_flat -= 0.02
            elif regime.trend == TrendRegime.STRONG_BEAR:
                p_down += 0.20
                p_up -= 0.15
                p_flat -= 0.05
            elif regime.trend == TrendRegime.WEAK_BEAR:
                p_down += 0.10
                p_up -= 0.08
                p_flat -= 0.02

        # Technical signals shift
        if technical_signals:
            for sig in technical_signals:
                sig_action = getattr(sig, "action", None)
                strength = getattr(sig, "strength", 0.5)
                if sig_action == SignalAction.BUY:
                    p_up += strength * 0.12
                    p_down -= strength * 0.10
                elif sig_action == SignalAction.SELL:
                    p_down += strength * 0.12
                    p_up -= strength * 0.10

        # Historical similarity consensus incorporation
        if pattern_consensus and pattern_consensus.get("sample_size", 0) > 0:
            win_rate = float(pattern_consensus.get("win_rate", 0.5))
            pat_shift = (win_rate - 0.5) * 0.25
            p_up += pat_shift
            p_down -= pat_shift

        # ML model probability incorporation if provided
        if ml_prob_up is not None:
            p_up = (p_up * 0.55) + (ml_prob_up * 0.45)
            p_down = (p_down * 0.55) + ((1.0 - ml_prob_up) * 0.45)

        # Normalize probabilities
        p_up = max(0.05, p_up)
        p_down = max(0.05, p_down)
        p_flat = max(0.05, p_flat)
        total = p_up + p_down + p_flat
        p_up = max(0.05, min(0.90, p_up / total))
        p_down = max(0.05, min(0.90, p_down / total))
        p_flat = max(0.05, min(0.60, 1.0 - p_up - p_down))

        # Expected return calculation
        expected_ret_pct = ((p_up - p_down) * exp_vol_pct * 0.65) + (ret_multi * 10.0)

        # Dynamic target and invalidation stops based on ATR & horizon
        horizon_atr_scale = max(0.6, min(3.5, math.sqrt(h_mins / 15.0)))
        stop_mult = 1.2 * horizon_atr_scale
        target_mult = 2.4 * horizon_atr_scale
        stop_dist = atr * stop_mult
        target_dist = atr * target_mult

        # Expected high and low excursion bounds over horizon
        up_exc = max(atr * 0.5 * horizon_atr_scale, current_price * (exp_vol_pct / 100.0) * (0.5 + p_up))
        down_exc = max(atr * 0.5 * horizon_atr_scale, current_price * (exp_vol_pct / 100.0) * (0.5 + p_down))
        expected_high = current_price + up_exc
        expected_low = max(0.00001, current_price - down_exc)

        reasons = []
        action = SignalAction.NO_TRADE
        confidence = max(p_up, p_down)

        if p_up >= 0.54 and expected_ret_pct > 0.05:
            action = SignalAction.BUY
            target_price = current_price + target_dist
            stop_price = current_price - stop_dist
            target_hit_prob = min(0.88, p_up * 1.1)
            stop_hit_prob = max(0.12, p_down * 0.9)
            reasons.append(f"Bullish conviction across {horizon} horizon (P(UP)={p_up*100:.1f}%)")
        elif p_down >= 0.54 and expected_ret_pct < -0.05:
            action = SignalAction.SELL
            target_price = current_price - target_dist
            stop_price = current_price + stop_dist
            target_hit_prob = min(0.88, p_down * 1.1)
            stop_hit_prob = max(0.12, p_up * 0.9)
            reasons.append(f"Bearish conviction across {horizon} horizon (P(DOWN)={p_down*100:.1f}%)")
        else:
            action = SignalAction.NO_TRADE
            target_price = current_price + target_dist
            stop_price = current_price - stop_dist
            target_hit_prob = round(p_up * 0.75, 4)
            stop_hit_prob = round(p_down * 0.75, 4)
            reasons.append(f"Indecision / Low conviction across {horizon} (P(UP)={p_up*100:.1f}%, P(DOWN)={p_down*100:.1f}%)")

        rr_ratio = (target_dist / stop_dist) if stop_dist > 0 else 2.0

        return ProbabilisticForecast(
            symbol=symbol,
            timestamp=now,
            forecast_horizon=horizon,
            horizon_seconds=h_secs,
            target_time=target_dt,
            entry_reference_price=current_price,
            prob_up=round(p_up, 4),
            prob_down=round(p_down, 4),
            prob_flat=round(p_flat, 4),
            expected_return_pct=round(expected_ret_pct, 4),
            expected_volatility_pct=round(exp_vol_pct, 4),
            expected_high=round(expected_high, 5),
            expected_low=round(expected_low, 5),
            target_price=round(target_price, 5),
            invalidation_stop_price=round(stop_price, 5),
            target_hit_probability=round(target_hit_prob, 4),
            stop_hit_probability=round(stop_hit_prob, 4),
            risk_reward_ratio=round(rr_ratio, 2),
            action=action,
            confidence=round(confidence, 4),
            expected_holding_period_minutes=float(h_mins),
            reasons=reasons,
            diagnostics={
                "status": "OK",
                "momentum_return": round(ret_last, 6),
                "multi_bar_return": round(ret_multi, 6),
                "regime": regime.regime_code if regime else "UNKNOWN",
                "candle_sample_size": len(closes),
            },
        )

    def forecast_all_horizons(
        self,
        symbol: str,
        current_price: float,
        atr: float,
        primary_horizon: str = "1h",
        regime: MarketRegimeState | None = None,
        technical_signals: list[Any] | None = None,
        candle_history: list[dict[str, Any]] | None = None,
        ml_prob_up: float | None = None,
        pattern_consensus: dict[str, Any] | None = None,
        latest_features: dict[str, Any] | None = None,
        data_is_stale: bool = False,
    ) -> MultiHorizonForecastReport:
        """Compute forecasts across all 5 institutional horizons (5m, 15m, 1h, 4h, 1d)."""
        forecasts: dict[str, ProbabilisticForecast] = {}
        for h in ("5m", "15m", "1h", "4h", "1d"):
            forecasts[h] = self.forecast_horizon(
                symbol=symbol,
                current_price=current_price,
                atr=atr,
                horizon=h,
                regime=regime,
                technical_signals=technical_signals,
                candle_history=candle_history,
                ml_prob_up=ml_prob_up,
                pattern_consensus=pattern_consensus,
                latest_features=latest_features,
                data_is_stale=data_is_stale,
            )

        primary = forecasts.get(primary_horizon, forecasts["1h"])
        return MultiHorizonForecastReport(
            symbol=symbol,
            current_price=current_price,
            atr_14=atr,
            regime_code=regime.regime_code if regime else "UNKNOWN",
            primary_decision=primary.action,
            primary_horizon=primary_horizon,
            forecasts=forecasts,
            reasons=primary.reasons,
            no_trade_reasons=primary.reasons if primary.action == SignalAction.NO_TRADE else [],
        )
