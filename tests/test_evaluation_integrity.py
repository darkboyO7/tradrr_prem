"""Automated Integrity and Regression Tests for Platform Evaluation Suite.

Verifies:
- Walk-forward fold counts, window bounds, and embargo enforcement.
- Absence of timestamp leakage or duplicates.
- Holdout overlap and contamination detection.
- Trade ledger schema completeness and execution pricing integrity.
- Cost sensitivity monotonicity.
- Confusion matrix and metric reconciliation.
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

FX_CACHE = Path("data/universe/fx_candles_cache.json")
OUTPUT_DIR = Path("C:/Users/Dark_Boy_/.gemini/antigravity/brain/b6abc64c-efb5-4d29-9019-2edf65681920/scratch")
WF_LEDGER_CSV = OUTPUT_DIR / "walk_forward_trade_ledger_519.csv"
WF_REPORT_JSON = OUTPUT_DIR / "walk_forward_reconciled_audit_report.json"


@pytest.fixture(scope="module")
def fx_data():
    assert FX_CACHE.exists(), f"FX cache missing at {FX_CACHE}"
    with open(FX_CACHE, "r", encoding="utf-8") as f:
        return json.load(f)


def test_walk_forward_fold_count_and_embargo(fx_data):
    """Verifies expanding walk-forward fold definitions, window counts, and embargo."""
    pairs = ["EURAUD", "EURCAD", "EURCHF"]
    fold_defs = [
        ("Fold 1", 0, 180, 182, 240, 58),
        ("Fold 2", 0, 240, 242, 300, 58),
        ("Fold 3", 0, 300, 302, 359, 57),
    ]

    total_oos = 0
    for pair in pairs:
        raw = fx_data[pair]
        df = pd.DataFrame(raw)
        df["timestamp"] = pd.to_datetime(df["timestamp"], format="ISO8601")
        df = df.sort_values("timestamp").reset_index(drop=True)
        fwd_ret = (df["close"].shift(-1) - df["close"]) / df["close"]
        valid = ~fwd_ret.isna()
        df_valid = df[valid].reset_index(drop=True)

        assert len(df_valid) == 359, f"Pair {pair} should have 359 valid bars, got {len(df_valid)}"
        pair_oos = 0

        for name, tr_s, tr_e, te_s, te_e, expected_len in fold_defs:
            tr_df = df_valid.iloc[tr_s:tr_e]
            emb_df = df_valid.iloc[tr_e:te_s]
            te_df = df_valid.iloc[te_s:te_e]

            # Embargo validation
            assert len(emb_df) == 2, f"Embargo must be exactly 2 bars in {pair} {name}"
            assert tr_df["timestamp"].iloc[-1] < emb_df["timestamp"].iloc[0]
            assert emb_df["timestamp"].iloc[-1] < te_df["timestamp"].iloc[0]

            # Test slice count
            assert len(te_df) == expected_len, f"Fold {name} expected {expected_len} bars, got {len(te_df)}"
            pair_oos += len(te_df)

        assert pair_oos == 173, f"Pair {pair} expected 173 total test bars, got {pair_oos}"
        total_oos += pair_oos

    assert total_oos == 519, f"Total OOS bars across 3 pairs must be exactly 519, got {total_oos}"


def test_holdout_contamination_and_overlap(fx_data):
    """Formally verifies the timestamp overlap between Phase 3, Phase 4, and Phase 5 holdouts."""
    raw = fx_data["EURUSD"]
    df = pd.DataFrame(raw)
    df["timestamp"] = pd.to_datetime(df["timestamp"], format="ISO8601")
    df = df.sort_values("timestamp").reset_index(drop=True)
    fwd_ret = (df["close"].shift(-1) - df["close"]) / df["close"]
    valid = ~fwd_ret.isna()
    df_valid = df[valid].reset_index(drop=True)

    n = len(df_valid)
    assert n == 359

    # Phase 3: 60/20/20 split -> train 215, val 71, emb 4, test 69
    p3_test = df_valid.iloc[215 + 4 + 71 :].reset_index(drop=True)
    # Phase 5: 50/25/25 split -> train 179, val 89, emb 2, test 87
    p5_test = df_valid.iloc[int(n * 0.50) + 2 + int(n * 0.25) + 2 :].reset_index(drop=True)

    assert len(p3_test) == 69
    assert len(p5_test) == 87

    p3_ts_set = set(p3_test["timestamp"])
    p5_ts_set = set(p5_test["timestamp"])

    overlap = p3_ts_set.intersection(p5_ts_set)
    assert len(overlap) == 69, "Phase 3 holdout must be 100% contained inside Phase 5 test window!"
    overlap_ratio = len(overlap) / len(p5_test)
    assert overlap_ratio > 0.79, f"Phase 5 has {overlap_ratio*100:.1f}% date overlap with Phase 3!"


def test_trade_ledger_schema_and_execution_integrity():
    """Verifies that the trade ledger has full schema, valid probabilities, and proper pricing."""
    assert WF_LEDGER_CSV.exists(), f"Missing trade ledger at {WF_LEDGER_CSV}"
    df = pd.read_csv(WF_LEDGER_CSV)

    assert len(df) == 519, f"Expected 519 ledger entries, got {len(df)}"

    required_columns = [
        "instrument", "fold", "bar_idx", "signal_timestamp", "latest_candle_timestamp",
        "decision", "prob_sell", "prob_neutral", "prob_buy", "true_class",
        "predicted_class", "entry_timestamp", "executable_entry_price",
        "exit_timestamp", "executable_exit_price", "forecast_horizon",
        "trade_taken", "direction", "gross_return", "spread_cost",
        "commission_cost", "slippage_cost", "net_return_baseline",
        "net_return_realistic", "net_return_adverse", "position_size", "reason"
    ]
    for col in required_columns:
        assert col in df.columns, f"Required column '{col}' missing from trade ledger!"

    # Probabilities row sum
    prob_sums = df["prob_sell"] + df["prob_neutral"] + df["prob_buy"]
    assert np.allclose(prob_sums, 1.0, atol=1e-5), "Probabilities do not sum to 1.0!"

    # Executable prices must be strictly positive
    assert (df["executable_entry_price"] > 0).all()
    assert (df["executable_exit_price"] > 0).all()

    # Direction and Trade Taken consistency
    active = df["trade_taken"]
    assert ((df["decision"] == "BUY") == (active & (df["direction"] == 1.0))).all()
    assert ((df["decision"] == "SELL") == (active & (df["direction"] == -1.0))).all()
    assert ((df["decision"] == "NO_TRADE") == (~active & (df["direction"] == 0.0))).all()


def test_cost_sensitivity_monotonicity():
    """Verifies that net returns strictly decrease as costs increase."""
    assert WF_LEDGER_CSV.exists()
    df = pd.read_csv(WF_LEDGER_CSV)
    active = df[df["trade_taken"]]

    # Under active trades: baseline (1.0 bps) >= realistic (1.5 bps) >= adverse (3.0 bps)
    assert (active["net_return_baseline"] >= active["net_return_realistic"]).all()
    assert (active["net_return_realistic"] >= active["net_return_adverse"]).all()

    # Cumulative compounding returns
    def cum_ret(col):
        eq = 1.0
        for r in df[col]:
            eq *= (1.0 + r)
        return eq - 1.0

    ret_base = cum_ret("net_return_baseline")
    ret_real = cum_ret("net_return_realistic")
    ret_adv = cum_ret("net_return_adverse")

    assert ret_base > ret_real > ret_adv, f"Cost monotonicity failed: {ret_base} > {ret_real} > {ret_adv}"


def test_confusion_matrix_reconciliation():
    """Verifies that confusion matrix totals and class supports reconcile perfectly."""
    assert WF_REPORT_JSON.exists(), f"Missing report at {WF_REPORT_JSON}"
    with open(WF_REPORT_JSON, "r", encoding="utf-8") as f:
        rep = json.load(f)

    cm = np.array(rep["confusion_matrix"]["matrix"])
    row_sums = np.array(rep["confusion_matrix"]["row_sums_support"])
    col_sums = np.array(rep["confusion_matrix"]["col_sums_predicted"])

    assert cm.sum() == 519
    assert row_sums.sum() == 519
    assert col_sums.sum() == 519
    assert (cm.sum(axis=1) == row_sums).all()
    assert (cm.sum(axis=0) == col_sums).all()


def test_strategy_comparison_ledger_and_persistence_discrepancy():
    """Verifies baseline comparison ledger totals, active trade counts, and the persistence discrepancy."""
    ledger_path = OUTPUT_DIR / "strategy_comparison_ledger_519.csv"
    assert ledger_path.exists(), f"Missing comparison ledger at {ledger_path}"
    df = pd.read_csv(ledger_path)

    assert len(df) == 519

    # Reconcile active trades
    lgbm_active = (df["lgbm_pred"] != 1).sum()
    assert lgbm_active == 220, f"Expected 220 LGBM active trades, got {lgbm_active}"

    persist_active = (df["persist_pred_aligned"] != 1).sum()
    assert persist_active == 431, f"Expected 431 Persistence active trades, got {persist_active}"

    majority_active = (df["majority_pred"] == 2).sum()
    assert majority_active == 519, f"Expected 519 Majority active trades, got {majority_active}"

    notrade_active = (df["notrade_pred"] != 1).sum()
    assert notrade_active == 0, f"Expected 0 No-Trade active trades, got {notrade_active}"

    # Properly aligned persistence accuracy
    persist_acc = (df["persist_pred_aligned"] == df["true_class"]).mean()
    assert round(persist_acc * 100.0, 2) == 38.34

    # Baseline audit report assertions
    report_path = OUTPUT_DIR / "baseline_audit_report.json"
    assert report_path.exists()
    with open(report_path, "r", encoding="utf-8") as f:
        rep = json.load(f)

    assert rep["persistence_discrepancy_audit"]["misaligned_method_A"]["accuracy_pct"] == 62.04
    assert rep["persistence_discrepancy_audit"]["properly_aligned_method_B"]["accuracy_pct"] == 38.34
    assert rep["strategies_summary"]["lgbm_candidate"]["active_trades"] == 220


def test_promotion_policy_pre_registered():
    """Asserts that the formal promotion policy document exists and contains core gates."""
    policy_path = Path("policies/MODEL_EVALUATION_AND_PROMOTION_POLICY.md")
    assert policy_path.exists(), f"Policy document missing at {policy_path}"
    content = policy_path.read_text(encoding="utf-8")

    assert "POL-GOV-2026-001" in content
    assert "Prospective Observation Requirements" in content
    assert "Maximum Allowable Drawdown" in content
    assert "Block-Bootstrapped Confidence Intervals" in content


def test_feature_pipeline_zero_lookahead(fx_data):
    """Verifies that future candles cannot leak into completed candle features."""
    from features.pipeline import FeaturePipeline
    fp = FeaturePipeline()

    raw = fx_data["EURUSD"]
    df = pd.DataFrame(raw[:200])
    df["timestamp"] = pd.to_datetime(df["timestamp"], format="ISO8601")

    # Baseline features up to bar 150
    feat_base = fp.compute_all_features(df.iloc[:151].copy())
    row_150_base = feat_base.iloc[150].to_dict()

    # Create mutated future scenario where bars 151..199 have extreme shocks
    df_mutated = df.copy()
    df_mutated.loc[151:, "close"] *= 5.0
    df_mutated.loc[151:, "high"] *= 5.0
    df_mutated.loc[151:, "low"] *= 5.0
    df_mutated.loc[151:, "volume"] *= 100.0

    feat_mutated = fp.compute_all_features(df_mutated)
    row_150_mutated = feat_mutated.iloc[150].to_dict()

    # Compare all numeric features at bar 150
    for col in feat_base.columns:
        val_base = row_150_base[col]
        val_mut = row_150_mutated[col]
        if isinstance(val_base, (int, float)) and not np.isnan(val_base):
            assert np.isclose(val_base, val_mut, atol=1e-7), (
                f"Lookahead leak detected in feature '{col}' at bar 150! "
                f"Base={val_base}, Mutated={val_mut}"
            )


def test_execution_fill_timing_and_spread_penalty():
    """Verifies that trades cannot fill before signal generation and enforce spread penalty."""
    assert WF_LEDGER_CSV.exists()
    df = pd.read_csv(WF_LEDGER_CSV)

    for _, row in df.iterrows():
        sig_ts = pd.to_datetime(row["signal_timestamp"])
        candle_ts = pd.to_datetime(row["latest_candle_timestamp"])
        entry_ts = pd.to_datetime(row["entry_timestamp"])
        exit_ts = pd.to_datetime(row["exit_timestamp"])

        # Signal cannot exist before the candle closes
        assert sig_ts >= candle_ts, f"Signal timestamp {sig_ts} precedes candle {candle_ts}"
        # Execution cannot occur before the signal is generated
        assert entry_ts >= sig_ts, f"Entry {entry_ts} precedes signal {sig_ts}"
        # Exit must occur strictly after entry
        assert exit_ts > entry_ts, f"Exit {exit_ts} does not follow entry {entry_ts}"

        if row["trade_taken"]:
            # Mathematical consistency of gross and net return
            dir_t = row["direction"]
            entry_px = row["executable_entry_price"]
            exit_px = row["executable_exit_price"]
            expected_gross = dir_t * (exit_px - entry_px) / entry_px
            assert np.isclose(row["gross_return"], expected_gross, atol=1e-5)

            # Realistic cost deduction
            expected_net = expected_gross - 0.00015
            assert np.isclose(row["net_return_realistic"], expected_net, atol=1e-5)


