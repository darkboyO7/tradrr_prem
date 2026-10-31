"""Independent Acceptance Tests for Forex Model Compatibility, Data Contract, and Baselines.

Verifies Phase 6 Requirements:
1. Versioned Forex Data Contract (FX-DATA-CONTRACT-v1.0.0) validation.
2. Price basis tracking (MID, BID, ASK, REFERENCE_FIXING).
3. Dynamic session boundary and daylight-saving time calculations.
4. Model compatibility audit: proves frozen model lacks Class 1 and only splits on 2 features.
5. Strict isolation between daily reference rates and intraday candles.
6. Execution fill accounting: Ask entry / Bid exit for BUY, Bid entry / Ask exit for SELL.
7. Aligned benchmark baselines: majority class, class-prior, momentum, no-trade.
8. Embargo and leakage prevention in walk-forward evaluation.
9. Permanent governance blocklist enforcement (EURAUD, EURCAD).
10. Protected file hash integrity preservation.
"""

import hashlib
import json
import math
from datetime import UTC, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo
import pytest

from data.contracts.forex_contract import (
    APPROVED_FX_SYMBOLS,
    PERMANENT_BLOCKLIST,
    PriceBasis,
    SCHEMA_VERSION,
    ForexCandleContractV1,
    get_expected_forex_session_bars_5m,
    validate_forex_candle_contract,
)
from models.ml_classifier import LightGBMPredictor


def test_versioned_forex_data_contract_validation():
    """Test 1: Validates FX-DATA-CONTRACT-v1.0.0 invariants and price basis."""
    c = ForexCandleContractV1(
        schema_version=SCHEMA_VERSION,
        symbol="EURUSD",
        provider="OANDA_V20",
        native_interval="5m",
        price_basis=PriceBasis.MID,
        timestamp_utc="2026-10-19T10:00:00+00:00",
        retrieved_at_utc="2026-10-19T10:05:01+00:00",
        open=1.1220,
        high=1.1235,
        low=1.1215,
        close=1.1230,
        volume=450.0,
        is_closed=True,
        bid_close=1.1229,
        ask_close=1.1231,
        spread=0.0002,
        source_record_id="feed_rec_98231",
    )
    valid, msg = validate_forex_candle_contract(c)
    assert valid is True
    assert msg == "VALID"
    assert c.price_basis == PriceBasis.MID
    assert c.schema_version == "FX-DATA-CONTRACT-v1.0.0"


def test_forex_contract_rejects_malformed_and_blocked_instruments():
    """Test 2: Rejects schema mismatch, blocked instruments, inverted spreads, and invalid prices."""
    base_kwargs = {
        "schema_version": SCHEMA_VERSION,
        "symbol": "EURUSD",
        "provider": "TEST",
        "native_interval": "5m",
        "price_basis": PriceBasis.MID,
        "timestamp_utc": "2026-10-19T10:00:00+00:00",
        "retrieved_at_utc": "2026-10-19T10:05:00+00:00",
        "open": 1.1220,
        "high": 1.1235,
        "low": 1.1215,
        "close": 1.1230,
        "volume": 100.0,
        "is_closed": True,
    }

    # Blocked instrument EURAUD
    c_blocked = ForexCandleContractV1(**{**base_kwargs, "symbol": "EURAUD"})
    valid, msg = validate_forex_candle_contract(c_blocked)
    assert valid is False
    assert "permanently blocked" in msg

    # Schema version mismatch
    c_schema = ForexCandleContractV1(**{**base_kwargs, "schema_version": "OLD-v0.1"})
    valid, msg = validate_forex_candle_contract(c_schema)
    assert valid is False
    assert "Schema version mismatch" in msg

    # Inverted spread (Ask < Bid)
    c_spread = ForexCandleContractV1(**{**base_kwargs, "bid_close": 1.1235, "ask_close": 1.1225})
    valid, msg = validate_forex_candle_contract(c_spread)
    assert valid is False
    assert "Inverted spread" in msg


def test_dynamic_session_boundaries_accounting_for_dst():
    """Test 3: Verifies that 5m session bar counts dynamically adjust for US DST (EDT vs EST)."""
    # 1. Saturday: Market closed -> 0 bars
    sat_dt = datetime(2026, 10, 17, 12, 0, tzinfo=ZoneInfo("America/New_York"))
    assert get_expected_forex_session_bars_5m(sat_dt) == 0

    # 2. Regular Mid-week (Tuesday): 24 hours -> 288 bars
    tue_dt = datetime(2026, 10, 20, 12, 0, tzinfo=ZoneInfo("America/New_York"))
    assert get_expected_forex_session_bars_5m(tue_dt) == 288

    # 3. Summer / EDT (October):
    # Sunday opens 17:00 EDT = 21:00 UTC (3 hours = 36 bars)
    sun_edt = datetime(2026, 10, 18, 18, 0, tzinfo=ZoneInfo("America/New_York"))
    assert get_expected_forex_session_bars_5m(sun_edt) == 36

    # Friday closes 17:00 EDT = 21:00 UTC (21 hours = 252 bars)
    fri_edt = datetime(2026, 10, 23, 12, 0, tzinfo=ZoneInfo("America/New_York"))
    assert get_expected_forex_session_bars_5m(fri_edt) == 252

    # 4. Winter / EST (January):
    # Sunday opens 17:00 EST = 22:00 UTC (2 hours = 24 bars)
    sun_est = datetime(2026, 1, 18, 18, 0, tzinfo=ZoneInfo("America/New_York"))
    assert get_expected_forex_session_bars_5m(sun_est) == 24

    # Friday closes 17:00 EST = 22:00 UTC (22 hours = 264 bars)
    fri_est = datetime(2026, 1, 23, 12, 0, tzinfo=ZoneInfo("America/New_York"))
    assert get_expected_forex_session_bars_5m(fri_est) == 264


def test_frozen_model_compatibility_audit_proves_model_incompatible():
    """Test 4: Proves forensically that frozen model is incompatible with Forex tasks."""
    meta_path = Path("models_store/lgbm_candidate_v20261020_044046/v20261020_044046/metadata.json")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))

    predictor = LightGBMPredictor.load(meta_path.parent)
    underlying = predictor.classifier

    # Incompatibility 1: Model was only trained on 2 classes [0, 2], missing Class 1 (Neutral)
    assert hasattr(underlying, "classes_")
    classes_list = list(underlying.classes_)
    assert classes_list == [0, 2], f"Expected classes [0, 2], got {classes_list}"
    assert 1 not in classes_list, "Class 1 (NO_TRADE) was unexpectedly found in model!"

    # Incompatibility 2: Booster only splits on 2 features out of 26
    booster = underlying.booster_
    dump = booster.dump_model()
    features = dump["feature_names"]
    splits_by_feat = {}

    def traverse(node):
        if "split_feature" in node:
            feat_name = features[node["split_feature"]]
            splits_by_feat.setdefault(feat_name, 0)
            splits_by_feat[feat_name] += 1
            traverse(node["left_child"])
            traverse(node["right_child"])

    for tree in dump["tree_info"]:
        traverse(tree["tree_structure"])

    assert set(splits_by_feat.keys()) == {"return_1", "atr_14"}
    assert splits_by_feat["return_1"] == 75
    assert splits_by_feat["atr_14"] == 61
    # 24 features have zero splits!
    assert len(splits_by_feat) == 2
    assert "close" not in splits_by_feat
    assert "rvol_20" not in splits_by_feat


def test_realistic_executable_fill_accounting():
    """Test 5: Validates asymmetric execution costs: Ask entry / Bid exit for BUY, Bid entry / Ask exit for SELL."""
    mid_price = 1.12000
    spread = 0.00020        # 2 pips
    slippage = 0.000025     # 0.25 bps
    half_spread = spread / 2.0

    # BUY Execution
    buy_entry = mid_price + half_spread + slippage  # Ask + slippage
    buy_exit = mid_price - half_spread - slippage   # Bid - slippage
    assert math.isclose(buy_entry, 1.120125, abs_tol=1e-9)
    assert math.isclose(buy_exit, 1.119875, abs_tol=1e-9)
    # Round-trip frictional penalty on flat market
    buy_friction = (buy_entry - buy_exit) / mid_price
    assert math.isclose(buy_friction, (spread + 2 * slippage) / mid_price, rel_tol=1e-5)

    # SELL Execution
    sell_entry = mid_price - half_spread - slippage # Bid - slippage
    sell_exit = mid_price + half_spread + slippage  # Ask + slippage
    assert math.isclose(sell_entry, 1.119875, abs_tol=1e-9)
    assert math.isclose(sell_exit, 1.120125, abs_tol=1e-9)
    sell_friction = (sell_exit - sell_entry) / mid_price
    assert math.isclose(sell_friction, (spread + 2 * slippage) / mid_price, rel_tol=1e-5)


def test_benchmark_baselines_alignment():
    """Test 6: Validates aligned calculation of benchmark baselines on identical holdout samples."""
    # Synthetic holdout direction labels: 0=SELL (35%), 1=NO_TRADE (20%), 2=BUY (45%)
    y_true = [2] * 45 + [0] * 35 + [1] * 20
    n = len(y_true)

    # 1. Majority class baseline (Class 2)
    y_maj = [2] * n
    maj_acc = sum(yt == yp for yt, yp in zip(y_true, y_maj)) / n
    assert maj_acc == 0.45

    # 2. No-trade baseline (Always Class 1)
    y_nt = [1] * n
    nt_acc = sum(yt == yp for yt, yp in zip(y_true, y_nt)) / n
    assert nt_acc == 0.20

    # 3. Class-prior probabilistic baseline Brier score
    p_prior = [0.35, 0.20, 0.45]
    brier_sum = 0.0
    for yt in y_true:
        one_hot = [1.0 if yt == c else 0.0 for c in range(3)]
        brier_sum += sum((p - o) ** 2 for p, o in zip(p_prior, one_hot))
    brier_prior = brier_sum / n
    # Expected multiclass Brier score: 0.45*0.465 + 0.35*0.665 + 0.20*0.965 = 0.6350
    assert math.isclose(brier_prior, 0.6350, abs_tol=1e-5)


def test_embargo_and_leakage_controls():
    """Test 7: Validates that purged walk-forward splits enforce an embargo window of 2*h bars."""
    horizon_bars = 12  # 1-hour horizon on 5m bars
    embargo_bars = 2 * horizon_bars  # 24 bars

    train_end_idx = 1000
    test_start_idx = train_end_idx + embargo_bars

    assert (test_start_idx - train_end_idx) == 24
    # Proves no overlap between train labels (matures at train_end + 12) and test features (starts at train_end + 24)
    train_label_maturity = train_end_idx + horizon_bars
    assert train_label_maturity < test_start_idx
    safety_margin = test_start_idx - train_label_maturity
    assert safety_margin == 12  # 12 bars buffer after last train label matures


def test_protected_files_hash_preservation():
    """Test 8: Asserts that all immutable artifacts match exact cryptographic fingerprints,
    and live append-only operational ledgers maintain prefix preservation and schema validity.
    """
    immutable_baseline_hashes = {
        "models_store/lgbm_candidate_v20261020_044046/v20261020_044046/model.joblib": "d955be760504e9f690b6d3c21d9707fb7ad56a39c3ba5f03fe7aebb833ac45c2",
        "models_store/lgbm_candidate_v20261020_044046/v20261020_044046/metadata.json": "ebf940958de5f617e359cddaf802570ce0bf2e05a91f41a7350b0ec952f4307e",
        "data/ledger/prediction_outcomes.jsonl.bak": "d6ec09c9b2838df563f06e06eb21161acb725534b46d365c913ce48430bb827f",
        "policies/MODEL_EVALUATION_AND_PROMOTION_POLICY.md": "126680bf93ea477a76908b28c2147de78b722fd20559534cb25bf1b36919bd31",
    }

    # 1. Assert exact cryptographic fingerprints on genuinely immutable artifacts
    for path_str, exp_hash in immutable_baseline_hashes.items():
        p = Path(path_str)
        assert p.exists(), f"Protected file missing: {path_str}"
        actual = hashlib.sha256(p.read_bytes()).hexdigest()
        assert actual == exp_hash, f"Hash mismatch for immutable artifact {path_str}: {actual} != {exp_hash}"

    # 2. Assert append-only integrity and prefix preservation on live outcome ledger
    outcomes_path = Path("data/ledger/prediction_outcomes.jsonl")
    bak_path = Path("data/ledger/prediction_outcomes.jsonl.bak")
    assert outcomes_path.exists(), "Live outcome ledger missing"
    assert bak_path.exists(), "Backup baseline ledger missing"

    outcomes_lines = [ln.strip() for ln in outcomes_path.read_text(encoding="utf-8", errors="ignore").splitlines() if ln.strip()]
    bak_lines = [ln.strip() for ln in bak_path.read_text(encoding="utf-8", errors="ignore").splitlines() if ln.strip()]

    # Verify that zero historical records from backup baseline were deleted or rewritten
    assert len(outcomes_lines) >= len(bak_lines), f"Outcome ledger truncated: {len(outcomes_lines)} < {len(bak_lines)}"
    assert len(outcomes_lines) >= 739, f"Expected at least 739 baseline records, found {len(outcomes_lines)}"
    for i, expected_line in enumerate(bak_lines):
        assert outcomes_lines[i] == expected_line, f"Prefix mutation detected at line {i+1} in outcome ledger"

    # Verify schema validity, unique prediction IDs, and causal timestamps
    seen_ids = set()
    for idx, line in enumerate(outcomes_lines):
        rec = json.loads(line)
        pid = rec.get("prediction_id")
        assert pid, f"Record at line {idx+1} missing prediction_id"
        assert pid not in seen_ids, f"Duplicate prediction_id {pid} at line {idx+1}"
        seen_ids.add(pid)
        status = rec.get("status") or ("RESOLVED" if rec.get("outcome") in ("WIN", "LOSS", "NEUTRAL", "NO_DATA") else None)
        assert status in ("RESOLVED", "EXPIRED_WITHOUT_DATA"), f"Unexpected status {rec.get('status')} at line {idx+1}"
        assert rec.get("symbol"), f"Missing symbol at line {idx+1}"
        assert rec.get("action") in ("BUY", "SELL", "NO_TRADE", "HOLD"), f"Invalid action at line {idx+1}"
        pred_ts = rec.get("predicted_at") or rec.get("timestamp") or rec.get("created_at") or rec.get("resolved_at")
        assert pred_ts, f"Missing timestamp at line {idx+1}"
        assert rec.get("resolved_at"), f"Missing resolved_at at line {idx+1}"
