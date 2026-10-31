"""Paper-Trading Prediction and Outcome Store.

Provides an append-only, crash-safe, and idempotent store for prospective paper-trading
predictions and their subsequent outcome resolutions.

Strictly separated from production ledgers and backtest experiments.
Guarantees restart recovery without duplicates and enforces explicit missing-data handling
without fabricating market prices.
"""

import json
import logging
import os
import threading
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger("paper_trading.prediction_store")

DEFAULT_PAPER_STORE_DIR = Path("data/paper_trading")


@dataclass(frozen=True)
class PaperPredictionRecord:
    prediction_id: str
    instrument: str
    timeframe: str
    forecast_horizon: str
    candle_timestamp: str
    prediction_timestamp: str
    model_version: str
    model_sha256: str
    config_hash: str
    prob_sell_raw: float
    prob_neutral_raw: float
    prob_buy_raw: float
    prob_sell_calibrated: float
    prob_neutral_calibrated: float
    prob_buy_calibrated: float
    decision: str  # BUY, SELL, NO_TRADE
    decision_margin: float
    no_trade_reason: str
    reference_price: float
    executable_entry_price: float
    intended_entry_rule: str
    scheduled_outcome_timestamp: str
    data_source_id: str
    freshness_status: str  # FRESH, STALE, DELAYED
    status: str = "PENDING"


@dataclass(frozen=True)
class PaperOutcomeRecord:
    prediction_id: str
    instrument: str
    timeframe: str
    forecast_horizon: str
    candle_timestamp: str
    prediction_timestamp: str
    model_version: str
    model_sha256: str
    config_hash: str
    decision: str
    reference_price: float
    executable_entry_price: float
    observed_exit_price: float | None
    resolution_timestamp: str
    scheduled_outcome_timestamp: str
    gross_return: float
    spread_cost: float
    commission_cost: float
    slippage_cost: float
    total_transaction_costs: float
    net_return: float
    outcome: str  # WIN, LOSS, NEUTRAL, EXPIRED_WITHOUT_DATA, INVALID_DATA
    data_source_status: str  # VALID_OBSERVATION, MISSING_DATA_DEFERRED, EXPIRED_WITHOUT_DATA, INVALID_DATA
    data_source_id: str
    notes: str = ""


class PaperTradingPredictionStore:
    """Append-only, crash-safe store for prospective paper-trading predictions and outcomes."""

    def __init__(self, store_dir: Path | str = DEFAULT_PAPER_STORE_DIR):
        self.store_dir = Path(store_dir)
        self.store_dir.mkdir(parents=True, exist_ok=True)
        self.predictions_file = self.store_dir / "paper_predictions.jsonl"
        self.outcomes_file = self.store_dir / "paper_outcomes.jsonl"
        self._lock = threading.RLock()
        self._pending_ids: set[str] = set()
        self._resolved_ids: set[str] = set()
        self._load_existing_state()

    def _load_existing_state(self) -> None:
        """Loads existing state from append-only JSONL files upon startup."""
        with self._lock:
            self._resolved_ids.clear()
            self._pending_ids.clear()

            # 1. Load resolved IDs
            if self.outcomes_file.exists():
                with open(self.outcomes_file, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            try:
                                record = json.loads(line)
                                pid = record.get("prediction_id")
                                if pid:
                                    self._resolved_ids.add(pid)
                            except json.JSONDecodeError as e:
                                logger.error(f"Malformed JSON in outcomes ledger: {e}")

            # 2. Load predictions to identify pending
            if self.predictions_file.exists():
                with open(self.predictions_file, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            try:
                                record = json.loads(line)
                                pid = record.get("prediction_id")
                                if pid and pid not in self._resolved_ids:
                                    self._pending_ids.add(pid)
                            except json.JSONDecodeError as e:
                                logger.error(f"Malformed JSON in predictions ledger: {e}")

            logger.info(
                f"PaperTradingStore initialized at {self.store_dir}: "
                f"{len(self._pending_ids)} pending, {len(self._resolved_ids)} resolved."
            )

    def record_prediction(self, prediction: PaperPredictionRecord) -> bool:
        """Records a prospective prediction before outcome is known.

        Enforces:
        - Atomic and crash-safe write via flush + fsync.
        - Idempotency: Rejects duplicate prediction_id.
        - Immutable: Never alters past records.
        """
        with self._lock:
            pid = prediction.prediction_id
            if pid in self._pending_ids or pid in self._resolved_ids:
                logger.warning(f"Paper prediction {pid} already exists. Ignoring duplicate.")
                return False

            data = asdict(prediction)
            with open(self.predictions_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(data) + "\n")
                f.flush()
                os.fsync(f.fileno())

            self._pending_ids.add(pid)
            return True

    def get_pending_predictions(self) -> list[dict[str, Any]]:
        """Returns all pending predictions that have not been resolved yet."""
        with self._lock:
            pending_list = []
            if not self.predictions_file.exists():
                return []

            with open(self.predictions_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        record = json.loads(line)
                        pid = record.get("prediction_id")
                        if pid in self._pending_ids:
                            pending_list.append(record)
            return pending_list

    def resolve_prediction(
        self,
        prediction_id: str,
        observed_exit_price: float | None,
        resolution_timestamp: str,
        data_source_status: str = "VALID_OBSERVATION",
        data_source_id: str = "TRADINGVIEW_FX_LIVE",
        spread_cost: float = 0.000075,      # 0.75 bps half-spread round-trip
        commission_cost: float = 0.000050,  # 0.50 bps commission
        slippage_cost: float = 0.000025,    # 0.25 bps slippage
        spread_embedded_in_price: bool = False,
        notes: str = "",
    ) -> PaperOutcomeRecord | None:
        """Resolves a pending prediction and appends outcome to paper_outcomes.jsonl.

        Enforces:
        - Separately itemized costs (spread, commission, slippage).
        - Zero double-counting: If spread and slippage are already embedded in the executable fill prices,
          only external commission/financing is subtracted from gross_return.
        - Atomic crash-safe append (flush + fsync).
        - Idempotency: Rejects already-resolved prediction_id.
        - Missing-data handling: No fabricated prices.
        """
        with self._lock:
            if prediction_id in self._resolved_ids:
                logger.warning(f"Prediction {prediction_id} already resolved. Skipping.")
                return None

            # Retrieve original prediction
            target_pred = None
            if self.predictions_file.exists():
                with open(self.predictions_file, "r", encoding="utf-8") as f:
                    for line in f:
                        record = json.loads(line)
                        if record.get("prediction_id") == prediction_id:
                            target_pred = record
                            break

            if not target_pred:
                logger.error(f"Cannot resolve unknown prediction {prediction_id}")
                return None

            decision = target_pred["decision"]
            entry_px = float(target_pred["executable_entry_price"])

            # Evaluate missing data or expiration
            if data_source_status in ("EXPIRED_WITHOUT_DATA", "INVALID_DATA") or observed_exit_price is None:
                outcome_status = "EXPIRED_WITHOUT_DATA" if data_source_status != "INVALID_DATA" else "INVALID_DATA"
                gross_ret = 0.0
                tot_costs = 0.0
                net_ret = 0.0
                exit_px = None
                sp_cost = 0.0
                cm_cost = 0.0
                sl_cost = 0.0
            else:
                exit_px = float(observed_exit_price)
                if decision == "BUY":
                    direction = 1.0
                elif decision == "SELL":
                    direction = -1.0
                else:
                    direction = 0.0

                if direction != 0.0:
                    gross_ret = direction * (exit_px - entry_px) / entry_px
                    sp_cost = spread_cost
                    cm_cost = commission_cost
                    sl_cost = slippage_cost
                    tot_costs = sp_cost + cm_cost + sl_cost
                    # If spread and slippage are already embedded in entry_px and exit_px, do not subtract twice!
                    if spread_embedded_in_price:
                        net_ret = gross_ret - cm_cost
                    else:
                        net_ret = gross_ret - tot_costs
                    outcome_status = "WIN" if net_ret > 0.0 else ("LOSS" if net_ret < 0.0 else "NEUTRAL")
                else:
                    gross_ret = 0.0
                    sp_cost = 0.0
                    cm_cost = 0.0
                    sl_cost = 0.0
                    tot_costs = 0.0
                    net_ret = 0.0
                    outcome_status = "NEUTRAL"
                    outcome_status = "NEUTRAL"

            outcome_record = PaperOutcomeRecord(
                prediction_id=prediction_id,
                instrument=target_pred["instrument"],
                timeframe=target_pred["timeframe"],
                forecast_horizon=target_pred["forecast_horizon"],
                candle_timestamp=target_pred["candle_timestamp"],
                prediction_timestamp=target_pred["prediction_timestamp"],
                model_version=target_pred["model_version"],
                model_sha256=target_pred["model_sha256"],
                config_hash=target_pred.get("config_hash", ""),
                decision=decision,
                reference_price=float(target_pred["reference_price"]),
                executable_entry_price=entry_px,
                observed_exit_price=exit_px,
                resolution_timestamp=resolution_timestamp,
                scheduled_outcome_timestamp=target_pred["scheduled_outcome_timestamp"],
                gross_return=round(gross_ret, 6),
                spread_cost=round(sp_cost, 6),
                commission_cost=round(cm_cost, 6),
                slippage_cost=round(sl_cost, 6),
                total_transaction_costs=round(tot_costs, 6),
                net_return=round(net_ret, 6),
                outcome=outcome_status,
                data_source_status=data_source_status,
                data_source_id=data_source_id,
                notes=notes,
            )

            with open(self.outcomes_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(asdict(outcome_record)) + "\n")
                f.flush()
                os.fsync(f.fileno())

            self._pending_ids.discard(prediction_id)
            self._resolved_ids.add(prediction_id)
            return outcome_record

    def handle_missing_data(
        self,
        prediction_id: str,
        current_timestamp: str,
        max_deferral_seconds: float = 86400.0 * 3,
    ) -> str:
        """Determines whether to DEFER resolution or EXPIRE due to missing data."""
        with self._lock:
            pending = [p for p in self.get_pending_predictions() if p["prediction_id"] == prediction_id]
            if not pending:
                return "NOT_PENDING"

            p = pending[0]
            sched_ts = datetime.fromisoformat(p["scheduled_outcome_timestamp"])
            now_ts = datetime.fromisoformat(current_timestamp)

            elapsed = (now_ts - sched_ts).total_seconds()
            if elapsed > max_deferral_seconds:
                self.resolve_prediction(
                    prediction_id=prediction_id,
                    observed_exit_price=None,
                    resolution_timestamp=current_timestamp,
                    data_source_status="EXPIRED_WITHOUT_DATA",
                    notes=f"Missing data exceeded max deferral of {max_deferral_seconds}s.",
                )
                return "EXPIRED_WITHOUT_DATA"
            else:
                return "MISSING_DATA_DEFERRED"

    def get_summary_metrics(self) -> dict[str, Any]:
        """Calculates paper trading metrics, properly handling missing/expired predictions.

        Missing-data deferral and expired outcomes are tracked separately and excluded from
        active win-rate denominators.
        """
        with self._lock:
            total_preds = 0
            if self.predictions_file.exists():
                with open(self.predictions_file, "r", encoding="utf-8") as f:
                    total_preds = sum(1 for line in f if line.strip())

            outcomes: list[dict[str, Any]] = []
            if self.outcomes_file.exists():
                with open(self.outcomes_file, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            outcomes.append(json.loads(line))

            pending_cnt = len(self._pending_ids)
            resolved_cnt = len(outcomes)

            # Categorize outcomes
            valid_active = [o for o in outcomes if o["outcome"] in ("WIN", "LOSS")]
            neutrals = [o for o in outcomes if o["outcome"] == "NEUTRAL"]
            expired = [o for o in outcomes if o["outcome"] == "EXPIRED_WITHOUT_DATA"]
            invalid = [o for o in outcomes if o["outcome"] == "INVALID_DATA"]

            wins = [o for o in valid_active if o["outcome"] == "WIN"]
            losses = [o for o in valid_active if o["outcome"] == "LOSS"]

            win_rate = (len(wins) / len(valid_active) * 100.0) if valid_active else 0.0

            # Cumulative net return over resolved trades
            eq = 1.0
            for o in outcomes:
                eq *= (1.0 + o["net_return"])
            net_ret = (eq - 1.0) * 100.0

            return {
                "total_predictions": total_preds,
                "pending_predictions": pending_cnt,
                "resolved_predictions": resolved_cnt,
                "valid_active_trades": len(valid_active),
                "neutral_trades": len(neutrals),
                "expired_without_data": len(expired),
                "invalid_data_count": len(invalid),
                "wins": len(wins),
                "losses": len(losses),
                "net_win_rate_pct": round(win_rate, 2),
                "net_cumulative_return_pct": round(net_ret, 4),
            }
