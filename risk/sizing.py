"""Trading Intelligence Platform - Position Sizing Engine.

Implements Fixed Fractional, Volatility-Adjusted, and Fractional Kelly sizing algorithms.
"""



import math


class PositionSizer:
    """Calculates safe, risk-managed order quantities."""

    @staticmethod
    def calculate_fixed_fractional_size(
        equity: float,
        entry_price: float,
        stop_loss: float,
        risk_per_trade_pct: float = 0.015,
        max_position_pct: float = 0.25,
    ) -> float:
        """Calculate order quantity such that hitting the stop loss loses exactly `risk_per_trade_pct` of equity.

        Args:
            equity: Current portfolio equity in base currency (e.g., USD).
            entry_price: Anticipated execution price.
            stop_loss: Validated stop loss level.
            risk_per_trade_pct: Fraction of equity to risk (e.g. 0.015 = 1.5%).
            max_position_pct: Maximum position notional value as fraction of equity.

        Returns:
            Approved asset quantity (e.g. BTC or ETH units).
        """
        if (
            math.isnan(equity) or math.isinf(equity) or
            math.isnan(entry_price) or math.isinf(entry_price) or
            math.isnan(stop_loss) or math.isinf(stop_loss) or
            math.isnan(risk_per_trade_pct) or math.isinf(risk_per_trade_pct) or
            math.isnan(max_position_pct) or math.isinf(max_position_pct) or
            equity <= 0 or entry_price <= 0 or stop_loss <= 0 or
            risk_per_trade_pct <= 0 or max_position_pct <= 0
        ):
            return 0.0

        risk_dollars = equity * risk_per_trade_pct
        per_unit_risk = abs(entry_price - stop_loss)

        if per_unit_risk <= 0:
            return 0.0

        # Raw risk-based size
        raw_quantity = risk_dollars / per_unit_risk

        # Cap by maximum allowable position notional value
        max_notional = equity * max_position_pct
        max_quantity = max_notional / entry_price

        final_quantity = min(raw_quantity, max_quantity)
        return round(float(final_quantity), 6)

    @staticmethod
    def calculate_volatility_adjusted_size(
        equity: float,
        entry_price: float,
        atr: float,
        target_vol_dollars: float,
        atr_multiplier: float = 1.5,
    ) -> float:
        """Calculate quantity scaling inversely with recent market ATR."""
        if (
            math.isnan(atr) or math.isinf(atr) or
            math.isnan(entry_price) or math.isinf(entry_price) or
            math.isnan(equity) or math.isinf(equity) or
            math.isnan(target_vol_dollars) or math.isinf(target_vol_dollars) or
            atr <= 0 or entry_price <= 0 or equity <= 0 or target_vol_dollars <= 0
        ):
            return 0.0

        dollar_risk_per_unit = atr * atr_multiplier
        if dollar_risk_per_unit <= 0 or math.isnan(dollar_risk_per_unit) or math.isinf(dollar_risk_per_unit):
            return 0.0
        quantity = target_vol_dollars / dollar_risk_per_unit
        return round(float(quantity), 6)

    @staticmethod
    def calculate_fractional_kelly(
        win_rate: float,
        payoff_ratio: float,
        fraction: float = 0.25,
    ) -> float:
        """Compute cautious fractional Kelly criterion (typically quarter-Kelly).

        Kelly formula: f* = (p * b - q) / b
        where p = win rate, q = 1 - p, b = payoff ratio (win/loss).
        """
        if (
            math.isnan(win_rate) or math.isinf(win_rate) or
            math.isnan(payoff_ratio) or math.isinf(payoff_ratio) or
            math.isnan(fraction) or math.isinf(fraction) or
            payoff_ratio <= 0 or win_rate <= 0 or win_rate >= 1.0 or fraction <= 0
        ):
            return 0.0

        q = 1.0 - win_rate
        full_kelly = (win_rate * payoff_ratio - q) / payoff_ratio

        if full_kelly <= 0:
            return 0.0

        # Apply fraction e.g. 0.25 for quarter-kelly safety buffer
        safe_kelly = full_kelly * fraction
        # Hard ceiling at 5% of equity
        return round(min(0.05, safe_kelly), 4)
