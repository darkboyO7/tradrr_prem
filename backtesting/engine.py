"""Trading Intelligence Platform - Chronological Event-Driven Backtesting Engine.

Simulates trading strategies across historical market data with realistic fee deductions,
slippage models, dynamic position sizing, and stop-loss / take-profit order execution.
"""

from typing import Any

import pandas as pd

from analysis.regime import MarketRegimeDetector
from backtesting.metrics import PerformanceMetricsCalculator
from models.base import SignalAction
from paper_trading.models import OrderSide, PaperOrder, PaperPosition
from paper_trading.simulator import ExecutionSimulator
from risk.engine import RiskDecisionStatus, RiskEngine
from risk.limits import RiskLimits
from signals.fusion import SignalFusionEngine
from signals.technical import TechnicalSignalGenerator


class BacktestEngine:
    """Historical trading simulation coordinator."""

    def __init__(
        self,
        initial_capital: float = 100000.0,
        limits: RiskLimits | None = None,
        simulator: ExecutionSimulator | None = None,
    ) -> None:
        self.initial_capital = initial_capital
        self.limits = limits or RiskLimits()
        self.simulator = simulator or ExecutionSimulator()

    def run(
        self,
        df: pd.DataFrame,
        symbol: str = "BTC-USD",
        timeframe: str = "1h",
        ml_model: Any | None = None,
    ) -> dict[str, Any]:
        """Execute a full chronological backtest over an OHLCV dataset.

        Returns:
            Dictionary containing:
                - 'equity_curve': pd.Series
                - 'trades': List of completed trades
                - 'metrics': PerformanceReport
        """
        if df.empty or len(df) < 20:
            raise ValueError("Dataset has insufficient bars for backtesting (minimum 20 bars required).")

        # Initialize engines
        regime_detector = MarketRegimeDetector()
        tech_generator = TechnicalSignalGenerator()
        fusion_engine = SignalFusionEngine()
        risk_engine = RiskEngine(initial_capital=self.initial_capital, limits=self.limits)

        # Precompute features and regimes
        classified_df = regime_detector.classify_dataframe(df)
        vol_map = {"COMPRESSED": 0.0, "NORMAL": 1.0, "ELEVATED": 2.0, "EXTREME": 3.0}
        if "volatility_regime" in classified_df.columns:
            classified_df_ml = classified_df.copy()
            classified_df_ml["volatility_regime"] = classified_df_ml["volatility_regime"].map(
                lambda v: vol_map.get(str(v), 1.0)
            )
        else:
            classified_df_ml = classified_df

        ml_preds_by_idx: dict[Any, Any] = {}
        if ml_model is not None and getattr(ml_model, "is_fitted", False):
            try:
                preds_list = ml_model.predict(classified_df_ml)
                for idx_val, pred_obj in zip(classified_df.index, preds_list):
                    ml_preds_by_idx[idx_val] = pred_obj
            except Exception:
                pass

        cash = self.initial_capital
        equity_records = []
        completed_trades: list[dict[str, Any]] = []
        active_position: PaperPosition | None = None

        for idx, row in classified_df.iterrows():
            current_price = float(row["close"])
            high_price = float(row["high"])
            low_price = float(row["low"])
            atr = float(row.get("atr_14", current_price * 0.01))

            # 1. Manage existing open position: check Stop Loss & Take Profit
            if active_position is not None:
                active_position.update_price(current_price)

                exit_price: float | None = None
                exit_reason = ""

                if active_position.side == OrderSide.BUY:
                    if low_price <= active_position.stop_loss:
                        exit_price = active_position.stop_loss
                        exit_reason = "STOP_LOSS_HIT"
                    elif high_price >= active_position.take_profit:
                        exit_price = active_position.take_profit
                        exit_reason = "TAKE_PROFIT_HIT"
                elif active_position.side == OrderSide.SELL:
                    if high_price >= active_position.stop_loss:
                        exit_price = active_position.stop_loss
                        exit_reason = "STOP_LOSS_HIT"
                    elif low_price <= active_position.take_profit:
                        exit_price = active_position.take_profit
                        exit_reason = "TAKE_PROFIT_HIT"

                if exit_price is not None:
                    # Execute exit
                    order = PaperOrder(
                        order_id=f"exit_{idx}",
                        symbol=symbol,
                        side=OrderSide.SELL if active_position.side == OrderSide.BUY else OrderSide.BUY,
                        quantity=active_position.quantity,
                    )
                    fill_p, fee, _ = self.simulator.simulate_fill(order, exit_price)
                    cash -= fee

                    if active_position.side == OrderSide.BUY:
                        pnl = (fill_p - active_position.entry_price) * active_position.quantity
                    else:
                        pnl = (active_position.entry_price - fill_p) * active_position.quantity

                    cash += pnl
                    completed_trades.append({
                        "symbol": symbol,
                        "side": active_position.side.value,
                        "entry_price": active_position.entry_price,
                        "exit_price": fill_p,
                        "quantity": active_position.quantity,
                        "pnl": round(pnl, 2),
                        "return_pct": round((pnl / (active_position.entry_price * active_position.quantity)) * 100.0, 3),
                        "reason": exit_reason,
                        "entry_bar": active_position.opened_at,
                        "exit_bar": idx,
                        "duration_bars": max(1, int(idx) - int(active_position.opened_at)) if isinstance(idx, int) else 1,
                    })
                    active_position = None

            # 2. Check for new entry if no active position
            current_equity = cash + (active_position.unrealized_pnl if active_position else 0.0)
            open_pos_map = {symbol: active_position.quantity * current_price} if active_position else {}
            risk_engine.update_portfolio_state(current_equity, open_pos_map)

            if active_position is None:
                # Classify regime
                regime_state = regime_detector.classify_bar(row)
                tech_signals = tech_generator.evaluate_bar(row)

                # Fuse signals
                decision = fusion_engine.fuse(
                    symbol=symbol,
                    current_price=current_price,
                    atr=atr,
                    regime=regime_state,
                    technical_signals=tech_signals,
                    ml_prediction=ml_preds_by_idx.get(idx),
                )

                if decision.action in (SignalAction.BUY, SignalAction.SELL):
                    risk_dec = risk_engine.evaluate_trade(decision)
                    if risk_dec.status in (RiskDecisionStatus.APPROVED, RiskDecisionStatus.RESIZED):
                        # Execute entry
                        side = OrderSide.BUY if decision.action == SignalAction.BUY else OrderSide.SELL
                        entry_order = PaperOrder(
                            order_id=f"entry_{idx}",
                            symbol=symbol,
                            side=side,
                            quantity=risk_dec.approved_quantity,
                            stop_loss=risk_dec.stop_loss,
                            take_profit=risk_dec.take_profit,
                        )
                        fill_p, fee, _ = self.simulator.simulate_fill(entry_order, current_price)
                        cash -= fee

                        active_position = PaperPosition(
                            position_id=f"pos_{idx}",
                            symbol=symbol,
                            side=side,
                            quantity=risk_dec.approved_quantity,
                            entry_price=fill_p,
                            current_price=fill_p,
                            stop_loss=risk_dec.stop_loss,
                            take_profit=risk_dec.take_profit,
                        )
                        active_position.opened_at = idx

            # Record periodic equity
            unrealized = active_position.unrealized_pnl if active_position else 0.0
            equity_records.append(cash + unrealized)

        # Build equity curve Series
        ts_index = df["timestamp"] if "timestamp" in df.columns else df.index
        equity_curve = pd.Series(equity_records, index=ts_index, name="equity")

        # Calculate metrics
        periods_year = 252 * 24 if timeframe == "1h" else (252 if timeframe == "1d" else 252 * 24 * 12)
        metrics = PerformanceMetricsCalculator.calculate_metrics(
            equity_curve=equity_curve,
            trades=completed_trades,
            periods_per_year=periods_year,
        )

        return {
            "equity_curve": equity_curve,
            "trades": completed_trades,
            "metrics": metrics,
        }

    def run_binary_options(
        self,
        df: pd.DataFrame,
        symbol: str = "QUOTEX:BTCUSD_QX",
        timeframe: str = "5m",
        expiry_bars: int = 3,
        payout_pct: float = 0.85,
        stake_amount: float = 100.0,
        ml_model: Any | None = None,
    ) -> dict[str, Any]:
        """Execute chronological binary options backtest adhering to Quotex payout & settlement rules.
        
        Rules:
        - Signal evaluated at bar t close.
        - Entry executes at bar t+1 open (zero lookahead).
        - Settlement occurs at bar t+1+expiry_bars close.
        - Win payout = +stake * payout_pct.
        - Loss = -stake.
        - Tie (price equal to strike) = 0.0 (stake returned).
        - Break-even win rate is 1 / (1 + payout_pct) = ~54.05% for payout=0.85.
        """
        if df.empty or len(df) < (20 + expiry_bars):
            raise ValueError(f"Insufficient bars for binary backtest (minimum {20 + expiry_bars} required).")

        regime_detector = MarketRegimeDetector()
        tech_generator = TechnicalSignalGenerator()
        fusion_engine = SignalFusionEngine()

        classified_df = regime_detector.classify_dataframe(df)
        cash = self.initial_capital
        equity_records = []
        completed_trades: list[dict[str, Any]] = []

        n_rows = len(classified_df)
        i = 0
        while i < n_rows - expiry_bars - 1:
            row = classified_df.iloc[i]
            current_price = float(row["close"])
            atr = float(row.get("atr_14", current_price * 0.01))

            regime_state = regime_detector.classify_bar(row)
            tech_signals = tech_generator.evaluate_bar(row)

            ml_pred = None
            if ml_model is not None and getattr(ml_model, "is_fitted", False):
                try:
                    p_list = ml_model.predict(classified_df.iloc[[i]])
                    if p_list:
                        ml_pred = p_list[0]
                except Exception:
                    pass

            decision = fusion_engine.fuse(
                symbol=symbol,
                current_price=current_price,
                atr=atr,
                regime=regime_state,
                technical_signals=tech_signals,
                ml_prediction=ml_pred,
            )

            # Record current equity before possible trade
            equity_records.append(cash)

            # Check if decision clears hurdle
            if decision.action in (SignalAction.BUY, SignalAction.SELL):
                # Entry at t+1 open
                entry_row = classified_df.iloc[i + 1]
                entry_price = float(entry_row["open"])

                # Expiry settlement at t + 1 + expiry_bars close
                exit_idx = i + 1 + expiry_bars
                exit_row = classified_df.iloc[exit_idx]
                exit_price = float(exit_row["close"])

                # Evaluate win / loss / tie
                if decision.action == SignalAction.BUY:
                    if exit_price > entry_price:
                        pnl = stake_amount * payout_pct
                        outcome = "WIN"
                    elif exit_price < entry_price:
                        pnl = -stake_amount
                        outcome = "LOSS"
                    else:
                        pnl = 0.0
                        outcome = "TIE"
                else:  # SELL / PUT
                    if exit_price < entry_price:
                        pnl = stake_amount * payout_pct
                        outcome = "WIN"
                    elif exit_price > entry_price:
                        pnl = -stake_amount
                        outcome = "LOSS"
                    else:
                        pnl = 0.0
                        outcome = "TIE"

                cash += pnl
                completed_trades.append({
                    "symbol": symbol,
                    "action": decision.action.value,
                    "entry_price": entry_price,
                    "exit_price": exit_price,
                    "stake": stake_amount,
                    "pnl": round(pnl, 2),
                    "return_pct": round((pnl / stake_amount) * 100.0, 2),
                    "outcome": outcome,
                    "payout_pct": payout_pct,
                    "entry_bar": i + 1,
                    "exit_bar": exit_idx,
                    "duration_bars": expiry_bars,
                })
                # Advance past the option's duration
                for _ in range(expiry_bars):
                    equity_records.append(cash)
                i += expiry_bars + 1
            else:
                i += 1

        # Pad remaining bars
        while len(equity_records) < len(classified_df):
            equity_records.append(cash)

        ts_index = df["timestamp"] if "timestamp" in df.columns else df.index
        equity_curve = pd.Series(equity_records[:len(ts_index)], index=ts_index[:len(equity_records)], name="equity")

        periods_year = 252 * 24 * 12  # 5-minute periods per year
        metrics = PerformanceMetricsCalculator.calculate_metrics(
            equity_curve=equity_curve,
            trades=completed_trades,
            periods_per_year=periods_year,
        )

        return {
            "equity_curve": equity_curve,
            "trades": completed_trades,
            "metrics": metrics,
            "break_even_win_rate_pct": round(1.0 / (1.0 + payout_pct) * 100.0, 2),
        }
