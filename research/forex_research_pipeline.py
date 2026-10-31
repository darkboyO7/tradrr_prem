"""Forex Research Pipeline and Candidate Development Framework.

Strictly isolated from production state (data/ledger/, models_store/).
Adheres to POL-GOV-2026-001 and FX-DATA-CONTRACT-v1.0.0.

Components:
1. ResearchDatasetManager: Isolated research store with provenance and gap logging.
2. ScaleInvariantForexFeatureExtractor: Scale-invariant technical features (NATR, Log-Returns, %B).
   Solves the unnormalized ATR split threshold bug identified in Phase 6.
3. ForexLabelingEngine: Triple-barrier / horizon labeling with transaction cost thresholds.
4. PurgedWalkForwardSplitter: Chronological train/val/test splits with embargo windows.
5. ForexRealisticCostEvaluator: Asymmetric execution fills, slippage, commission, and financing.
6. BenchmarkBaselineEvaluator: Majority-class, class-prior, momentum, and no-trade baselines.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from data.contracts.forex_contract import (
    APPROVED_FX_SYMBOLS,
    PERMANENT_BLOCKLIST,
    PriceBasis,
    ForexCandleContractV1,
    validate_forex_candle_contract,
    validate_forex_sequence_contract,
)

RESEARCH_BASE_DIR = Path("data/research")
RESEARCH_MODELS_DIR = Path("data/research/models")
RESEARCH_DATASETS_DIR = Path("data/research/datasets")
RESEARCH_AUDIT_LOG = Path("data/research/research_audit_ledger.jsonl")


@dataclass(frozen=True)
class ForexCostStructure:
    """Instrument-specific transaction cost structure.

    Units:
    - spread: Price units in quote currency (e.g. 0.00015 = 1.5 pips for EURUSD)
    - slippage: Price units in quote currency per side (e.g. 0.000020 = 0.20 pip)
    - commission_rate: Fractional commission per side on notional value (e.g. 0.000030 = 0.30 bps)
    - margin_rate: Additional required directional profit margin above round-trip friction (e.g. 0.000100 = 1.0 bps)
    """

    spread: float
    slippage: float
    commission_rate: float
    margin_rate: float = 0.000100  # 1.0 bps default alpha buffer

    def round_trip_frictional_cost(self, reference_price: float | pd.Series) -> float | pd.Series:
        """Calculates total round-trip friction in return fraction (spread + 2*slippage + 2*commission)."""
        if isinstance(reference_price, (int, float)):
            if reference_price <= 0.0:
                raise ValueError(f"Reference price must be positive, got {reference_price}")
        spread_ret = self.spread / reference_price
        slip_ret = (2.0 * self.slippage) / reference_price
        comm_ret = 2.0 * self.commission_rate
        return spread_ret + slip_ret + comm_ret

    def directional_hurdle(self, reference_price: float | pd.Series) -> float | pd.Series:
        """Minimum expected return required to trigger BUY/SELL label: friction + margin."""
        return self.round_trip_frictional_cost(reference_price) + self.margin_rate


DEFAULT_FX_COSTS = {
    "EURUSD": ForexCostStructure(spread=0.00015, slippage=0.000020, commission_rate=0.000030, margin_rate=0.000100),
    "EURCHF": ForexCostStructure(spread=0.00020, slippage=0.000025, commission_rate=0.000030, margin_rate=0.000100),
}


@dataclass
class ResearchDatasetMetadata:
    dataset_id: str
    symbol: str
    native_interval: str
    price_basis: str
    start_time_utc: str
    end_time_utc: str
    observation_count: int
    detected_gaps_count: int
    missing_bars_count: int
    provider: str
    retrieval_timestamp: str
    notes: str
    feature_pipeline_version: str = "FX-FEAT-v1.0.0"
    feature_names: list[str] = field(default_factory=list)


class ResearchDatasetManager:
    """Manages isolated research datasets outside production ledgers."""

    def __init__(self, base_dir: Path | str = RESEARCH_BASE_DIR):
        self.base_dir = Path(base_dir)
        self.datasets_dir = self.base_dir / "datasets"
        self.models_dir = self.base_dir / "models"
        self.audit_log = self.base_dir / "research_audit_ledger.jsonl"
        self._ensure_dirs()

    def _ensure_dirs(self) -> None:
        self.datasets_dir.mkdir(parents=True, exist_ok=True)
        self.models_dir.mkdir(parents=True, exist_ok=True)

    def save_research_dataset(
        self,
        dataset_id: str,
        symbol: str,
        candles: list[ForexCandleContractV1],
        expected_interval: str,
        provider: str,
        notes: str = "",
    ) -> tuple[bool, str, ResearchDatasetMetadata | None]:
        """Validates and stores an isolated research dataset with full provenance."""
        if symbol in PERMANENT_BLOCKLIST:
            return False, f"Symbol {symbol} is permanently blocked by governance.", None
        if symbol not in APPROVED_FX_SYMBOLS:
            return False, f"Symbol {symbol} is not approved under governance.", None

        valid, msg, gaps = validate_forex_sequence_contract(candles, expected_interval=expected_interval)
        if not valid:
            return False, f"Dataset sequence validation failed: {msg}", None

        now_dt = datetime.now(UTC)
        total_missing_bars = sum(g.get("missing_bars", 0) for g in gaps)
        meta = ResearchDatasetMetadata(
            dataset_id=dataset_id,
            symbol=symbol,
            native_interval=expected_interval,
            price_basis=candles[0].price_basis.value if candles else "UNKNOWN",
            start_time_utc=candles[0].timestamp_utc if candles else "",
            end_time_utc=candles[-1].timestamp_utc if candles else "",
            observation_count=len(candles),
            detected_gaps_count=len(gaps),
            missing_bars_count=total_missing_bars,
            provider=provider,
            retrieval_timestamp=now_dt.isoformat(),
            notes=notes,
            feature_pipeline_version=ScaleInvariantForexFeatureExtractor.FEATURE_PIPELINE_VERSION,
            feature_names=list(ScaleInvariantForexFeatureExtractor.FEATURE_NAMES),
        )

        # Write dataset file
        data_payload = {
            "metadata": asdict(meta),
            "gaps": gaps,
            "candles": [c.to_dict() for c in candles],
        }
        file_path = self.datasets_dir / f"{dataset_id}.json"
        tmp_path = file_path.with_suffix(".tmp")
        tmp_path.write_text(json.dumps(data_payload, indent=2), encoding="utf-8")
        tmp_path.replace(file_path)

        # Append to research audit log
        log_entry = {
            "timestamp": now_dt.isoformat(),
            "event": "DATASET_SAVED",
            "dataset_id": dataset_id,
            "symbol": symbol,
            "interval": expected_interval,
            "count": len(candles),
            "file": str(file_path),
            "feature_pipeline_version": meta.feature_pipeline_version,
        }
        with open(self.audit_log, "a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry) + "\n")

        return True, f"Saved {len(candles)} candles to {file_path}", meta

    def load_research_dataset(self, dataset_id: str) -> dict[str, Any]:
        """Loads an isolated research dataset and verifies schema version."""
        file_path = self.datasets_dir / f"{dataset_id}.json"
        if not file_path.exists():
            raise FileNotFoundError(f"Research dataset {dataset_id} not found at {file_path}")
        data = json.loads(file_path.read_text(encoding="utf-8"))
        meta = data.get("metadata", {})
        ver = meta.get("feature_pipeline_version")
        if ver and ver != ScaleInvariantForexFeatureExtractor.FEATURE_PIPELINE_VERSION:
            raise ValueError(
                f"Incompatible feature pipeline version in dataset {dataset_id}: "
                f"expected {ScaleInvariantForexFeatureExtractor.FEATURE_PIPELINE_VERSION}, found {ver}."
            )
        return data


class ScaleInvariantForexFeatureExtractor:
    """Extracts strictly scale-invariant, session-gap-aware features for Forex prediction.

    Guarantees:
    - Zero absolute nominal prices (O, H, L, C, EMA).
    - Normalized ATR (NATR = ATR / Close) to eliminate nominal split threshold distortion.
    - Symmetrical log-returns.
    - Session-gap-aware handling: weekend closures and missing candle gaps (>300s) do not
      distort returns or true range indicators.
    - Zero future lookahead leakage.
    """

    FEATURE_PIPELINE_VERSION: str = "FX-FEAT-v1.0.0"

    FEATURE_NAMES = [
        "log_return_1",
        "log_return_5",
        "log_return_12",
        "natr_14",
        "candle_body_ratio",
        "candle_range_pct",
        "rsi_14",
        "norm_macd_hist",
        "bb_pct_b_20",
        "adx_14",
        "plus_di_14",
        "minus_di_14",
    ]

    def extract_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extracts scale-invariant feature DataFrame from OHLC DataFrame."""
        if len(df) < 30:
            return pd.DataFrame()

        close = df["close"].astype(float)
        high = df["high"].astype(float)
        low = df["low"].astype(float)
        open_p = df["open"].astype(float)

        # Detect session gaps (> 300 seconds for 5m interval)
        if "timestamp" in df.columns or "timestamp_utc" in df.columns:
            ts_col = "timestamp" if "timestamp" in df.columns else "timestamp_utc"
            ts = pd.to_datetime(df[ts_col], utc=True)
            deltas = ts.diff().dt.total_seconds()
            is_gap = deltas > 300.0
        elif isinstance(df.index, pd.DatetimeIndex):
            deltas = df.index.to_series().diff().dt.total_seconds()
            is_gap = deltas > 300.0
        else:
            is_gap = pd.Series(False, index=df.index)

        feat = pd.DataFrame(index=df.index)

        # 1. Log returns with session-gap protection
        # For normal bars: log(close / close.shift(1))
        # Across a gap: use intra-bar return log(close / open) to prevent weekend jump leakage
        raw_log_ret_1 = np.log(close / close.shift(1)).fillna(0.0)
        intrabar_ret_1 = np.log(close / open_p).fillna(0.0)
        feat["log_return_1"] = np.where(is_gap, intrabar_ret_1, raw_log_ret_1)

        # Multi-bar returns within contiguous sessions
        session_group = is_gap.cumsum()
        df_session = pd.DataFrame({"close": close, "session_group": session_group})
        feat["log_return_5"] = df_session.groupby("session_group", group_keys=False)["close"].apply(
            lambda s: np.log(s / s.shift(5)).fillna(0.0)
        )
        feat["log_return_12"] = df_session.groupby("session_group", group_keys=False)["close"].apply(
            lambda s: np.log(s / s.shift(12)).fillna(0.0)
        )

        # 2. True Range & Normalized ATR (NATR) with session gap filtering
        prev_close = close.shift(1)
        tr1 = high - low
        tr2 = np.where(is_gap, 0.0, (high - prev_close).abs())
        tr3 = np.where(is_gap, 0.0, (low - prev_close).abs())
        tr = pd.concat([pd.Series(tr1, index=df.index), pd.Series(tr2, index=df.index), pd.Series(tr3, index=df.index)], axis=1).max(axis=1)
        atr_14 = tr.rolling(window=14, min_periods=1).mean()
        feat["natr_14"] = (atr_14 / close).fillna(0.0)

        # 3. Candle Geometry
        candle_range = high - low
        feat["candle_range_pct"] = (candle_range / close).fillna(0.0)
        feat["candle_body_ratio"] = ((close - open_p).abs() / (candle_range + 1e-7)).clip(0.0, 1.0).fillna(0.0)

        # 4. RSI-14 with gap-filtered delta
        gap_adjusted_delta = np.where(is_gap, close - open_p, close.diff())
        gain = pd.Series(gap_adjusted_delta, index=df.index).clip(lower=0.0)
        loss = -pd.Series(gap_adjusted_delta, index=df.index).clip(upper=0.0)
        avg_gain = gain.rolling(window=14, min_periods=1).mean()
        avg_loss = loss.rolling(window=14, min_periods=1).mean()
        rs = avg_gain / (avg_loss + 1e-9)
        feat["rsi_14"] = (100.0 - (100.0 / (1.0 + rs))).fillna(50.0)

        # 5. Normalized MACD Histogram
        ema_12 = close.ewm(span=12, adjust=False).mean()
        ema_26 = close.ewm(span=26, adjust=False).mean()
        macd_line = ema_12 - ema_26
        signal_line = macd_line.ewm(span=9, adjust=False).mean()
        macd_hist = macd_line - signal_line
        feat["norm_macd_hist"] = (macd_hist / close).fillna(0.0)

        # 6. Bollinger Bands %B (20, 2)
        sma_20 = close.rolling(window=20, min_periods=1).mean()
        std_20 = close.rolling(window=20, min_periods=1).std().fillna(1e-6)
        upper_bb = sma_20 + 2.0 * std_20
        lower_bb = sma_20 - 2.0 * std_20
        feat["bb_pct_b_20"] = ((close - lower_bb) / (upper_bb - lower_bb + 1e-9)).clip(-1.0, 2.0).fillna(0.5)

        # 7. Directional Movement (ADX-14, +DI, -DI) with gap protection
        up_move = np.where(is_gap, 0.0, high.diff())
        down_move = np.where(is_gap, 0.0, -low.diff())
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
        plus_di = 100.0 * (pd.Series(plus_dm, index=df.index).rolling(14, min_periods=1).mean() / (atr_14 + 1e-9))
        minus_di = 100.0 * (pd.Series(minus_dm, index=df.index).rolling(14, min_periods=1).mean() / (atr_14 + 1e-9))
        dx = 100.0 * ((plus_di - minus_di).abs() / (plus_di + minus_di + 1e-9))
        feat["adx_14"] = dx.rolling(14, min_periods=1).mean().fillna(20.0)
        feat["plus_di_14"] = plus_di.fillna(20.0)
        feat["minus_di_14"] = minus_di.fillna(20.0)

        return feat[self.FEATURE_NAMES].fillna(0.0)


class ForexLabelingEngine:
    """Computes three-class directional labels with realistic cost barrier and execution alignment."""

    def __init__(
        self,
        horizon_bars: int = 12,
        cost_threshold: float | None = None,
        cost_structure: ForexCostStructure | None = None,
        symbol: str = "EURUSD",
        entry_mode: str = "next_open",
    ):
        self.horizon_bars = horizon_bars
        self.symbol = symbol
        self.entry_mode = entry_mode
        self.cost_structure = cost_structure or (DEFAULT_FX_COSTS.get(symbol) if cost_threshold is None else None)
        self.fixed_cost_threshold = cost_threshold

    def create_labels(self, df: pd.DataFrame) -> pd.Series:
        """Constructs target labels y in {0: SELL, 1: NO_TRADE, 2: BUY}.

        Timing Conventions:
        - Prediction is made at bar t (completed bar).
        - If entry_mode == 'next_open' and 'open' in df:
            Executable entry is bar t+1 open (no lookahead; prediction precedes open).
            Exit price is bar t+1+horizon_bars open.
        - If entry_mode == 'same_close' or 'open' not in df:
            Executable entry is bar t close.
            Exit price is bar t+horizon_bars close.
        """
        if self.entry_mode == "next_open" and "open" in df.columns:
            entry_price = df["open"].astype(float).shift(-1)
            exit_price = df["open"].astype(float).shift(-(1 + self.horizon_bars))
        else:
            entry_price = df["close"].astype(float)
            exit_price = df["close"].astype(float).shift(-self.horizon_bars)

        forward_return = (exit_price - entry_price) / entry_price

        if self.fixed_cost_threshold is not None:
            theta = self.fixed_cost_threshold
        elif self.cost_structure is not None:
            theta = self.cost_structure.directional_hurdle(entry_price)
        else:
            theta = DEFAULT_FX_COSTS["EURUSD"].directional_hurdle(entry_price)

        labels = pd.Series(index=df.index, dtype=float)
        # Mutually exclusive and collectively exhaustive over finite forward returns
        labels[forward_return >= theta] = 2.0   # BUY
        labels[forward_return <= -theta] = 0.0  # SELL
        labels[(forward_return > -theta) & (forward_return < theta)] = 1.0  # NO_TRADE
        # When forward_return is NaN (incomplete horizon), label remains NaN

        return labels


class PurgedWalkForwardSplitter:
    """Partitions time series chronologically with purging and embargo."""

    def __init__(self, train_pct: float = 0.60, val_pct: float = 0.20, embargo_bars: int = 24):
        self.train_pct = train_pct
        self.val_pct = val_pct
        self.embargo_bars = embargo_bars

    def split(self, n_samples: int) -> dict[str, tuple[int, int]]:
        """Returns index ranges for train, validation, and holdout test."""
        train_end = int(n_samples * self.train_pct)
        val_start = train_end + self.embargo_bars
        val_end = val_start + int(n_samples * self.val_pct)
        test_start = val_end + self.embargo_bars
        test_end = n_samples

        if val_start >= val_end or test_start >= test_end:
            raise ValueError(f"Insufficient samples ({n_samples}) for purged walk-forward splitting.")

        return {
            "train": (0, train_end),
            "val": (val_start, val_end),
            "test": (test_start, test_end),
        }


class ForexRealisticCostEvaluator:
    """Evaluates prediction returns under realistic asymmetric execution costs."""

    def __init__(
        self,
        cost_structure: ForexCostStructure | None = None,
        spread: float = 0.00020,     # 2 pips
        slippage: float = 0.000025,  # 0.25 pip price addendum
        commission: float = 0.000050,# 0.50 bps fraction
    ):
        if cost_structure is not None:
            self.spread = cost_structure.spread
            self.slippage = cost_structure.slippage
            self.commission = cost_structure.commission_rate
            self.cost_structure = cost_structure
        else:
            self.spread = spread
            self.slippage = slippage
            self.commission = commission
            self.cost_structure = ForexCostStructure(
                spread=spread,
                slippage=slippage,
                commission_rate=commission,
                margin_rate=0.0,
            )
        self.half_spread = self.spread / 2.0

    def evaluate_trades(
        self,
        predictions: np.ndarray,      # 0=SELL, 1=NO_TRADE, 2=BUY
        entry_mids: np.ndarray,
        exit_mids: np.ndarray,
    ) -> dict[str, Any]:
        """Calculates net returns with Ask/Bid execution, slippage, and commissions."""
        n = len(predictions)
        trade_returns = []
        actions_taken = 0

        for act, p_entry, p_exit in zip(predictions, entry_mids, exit_mids):
            if act == 2:  # BUY
                # Enter at Ask, Exit at Bid
                exec_entry = p_entry + self.half_spread + self.slippage
                exec_exit = p_exit - self.half_spread - self.slippage
                gross_ret = (exec_exit - exec_entry) / exec_entry
                net_ret = gross_ret - (2 * self.commission)
                trade_returns.append(net_ret)
                actions_taken += 1
            elif act == 0:  # SELL
                # Enter at Bid, Exit at Ask
                exec_entry = p_entry - self.half_spread - self.slippage
                exec_exit = p_exit + self.half_spread + self.slippage
                gross_ret = (exec_entry - exec_exit) / exec_entry
                net_ret = gross_ret - (2 * self.commission)
                trade_returns.append(net_ret)
                actions_taken += 1
            else:  # NO_TRADE
                trade_returns.append(0.0)

        returns_arr = np.array(trade_returns)
        total_net_return = float(np.sum(returns_arr))
        mean_trade_return = float(np.mean(returns_arr[returns_arr != 0.0])) if actions_taken > 0 else 0.0
        win_rate = float(np.sum(returns_arr > 0.0) / actions_taken) if actions_taken > 0 else 0.0

        # Maximum drawdown
        cum_ret = np.cumsum(returns_arr)
        peak = np.maximum.accumulate(cum_ret)
        dd = peak - cum_ret
        max_dd = float(np.max(dd)) if len(dd) > 0 else 0.0

        return {
            "total_samples": n,
            "actions_taken": actions_taken,
            "coverage_pct": round(100.0 * actions_taken / max(1, n), 2),
            "total_net_return": round(total_net_return, 6),
            "mean_trade_return": round(mean_trade_return, 6),
            "win_rate_pct": round(100.0 * win_rate, 2),
            "max_drawdown": round(max_dd, 6),
        }


class BenchmarkBaselineEvaluator:
    """Calculates benchmark baselines for comparative evaluation."""

    @staticmethod
    def evaluate_majority_baseline(y_train: np.ndarray, y_test: np.ndarray) -> float:
        """Returns accuracy of majority class from training set applied to test set."""
        classes, counts = np.unique(y_train[~np.isnan(y_train)], return_counts=True)
        majority_class = classes[np.argmax(counts)]
        return float(np.mean(y_test == majority_class))

    @staticmethod
    def evaluate_class_prior_baseline(y_train: np.ndarray, y_test: np.ndarray) -> dict[str, float]:
        """Calculates multiclass Brier score of class priors."""
        classes, counts = np.unique(y_train[~np.isnan(y_train)], return_counts=True)
        total = float(np.sum(counts))
        priors = np.zeros(3)
        for c, count in zip(classes, counts):
            priors[int(c)] = count / total

        brier_sum = 0.0
        for yt in y_test:
            if np.isnan(yt):
                continue
            one_hot = np.zeros(3)
            one_hot[int(yt)] = 1.0
            brier_sum += np.sum((priors - one_hot) ** 2)

        return {
            "priors": priors.tolist(),
            "brier_score": float(brier_sum / len(y_test)),
        }

    @staticmethod
    def evaluate_momentum_baseline(
        close_prices: np.ndarray,
        y_test: np.ndarray,
        test_indices: np.ndarray | list[int] | None = None,
        hurdle: float = 0.0,
    ) -> float:
        """Evaluates 1-bar momentum direction baseline with explicit index alignment.

        Three-class policy:
        - If past return >= +hurdle: predict BUY (2)
        - If past return <= -hurdle: predict SELL (0)
        - If -hurdle < past return < +hurdle: predict NO_TRADE (1)

        If hurdle == 0.0: classic binary momentum predicts BUY if past return > 0 else SELL.
        """
        close = np.asarray(close_prices, dtype=float)
        y_true = np.asarray(y_test, dtype=float)
        n = len(close)

        past_returns = np.zeros(n, dtype=float)
        past_returns[1:] = (close[1:] - close[:-1]) / (close[:-1] + 1e-12)

        if hurdle > 0.0:
            preds = np.ones(n, dtype=float)  # default NO_TRADE (1)
            preds[past_returns >= hurdle] = 2.0   # BUY
            preds[past_returns <= -hurdle] = 0.0  # SELL
        else:
            preds = np.where(past_returns > 0, 2.0, 0.0)

        if test_indices is not None:
            idx = np.asarray(test_indices, dtype=int)
            y_pred_eval = preds[idx]
            y_true_eval = y_true
        else:
            y_pred_eval = preds[-len(y_true):]
            y_true_eval = y_true

        valid = ~np.isnan(y_true_eval)
        if not np.any(valid):
            return 0.0
        return float(np.mean(y_pred_eval[valid] == y_true_eval[valid]))
