"""Independent Unit and Acceptance Tests for Forex Research Pipeline.

Verifies Phase 7 Requirements:
1. Research dataset isolation: strictly writes to data/research/, never touches data/ledger/.
2. Scale-invariant feature extraction: proves NATR, log-returns, and absence of nominal prices.
3. Three-class labeling engine with cost threshold.
4. Purged walk-forward splitter enforcing embargo windows.
5. Realistic asymmetric cost evaluator (Ask entry / Bid exit for BUY, Bid entry / Ask exit for SELL).
6. Benchmark baseline evaluator (majority class, class priors, momentum).
7. Governance blocklist enforcement (EURAUD, EURCAD).
8. Zero lookahead leakage in feature computation.
"""

import math
from datetime import UTC, datetime, timedelta
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from data.contracts.forex_contract import (
    ForexCandleContractV1,
    PriceBasis,
    SCHEMA_VERSION,
)
from research.forex_research_pipeline import (
    DEFAULT_FX_COSTS,
    BenchmarkBaselineEvaluator,
    ForexCostStructure,
    ForexLabelingEngine,
    ForexRealisticCostEvaluator,
    PurgedWalkForwardSplitter,
    ResearchDatasetManager,
    ScaleInvariantForexFeatureExtractor,
)


@pytest.fixture
def temp_research_dir(tmp_path):
    res_dir = tmp_path / "research_test"
    res_dir.mkdir(parents=True, exist_ok=True)
    return res_dir


def test_research_dataset_manager_isolation(temp_research_dir):
    """Test 1: Proves research datasets are isolated from production ledgers."""
    manager = ResearchDatasetManager(base_dir=temp_research_dir)

    base_dt = datetime(2026, 10, 19, 10, 0, tzinfo=UTC)
    candles = [
        ForexCandleContractV1(
            schema_version=SCHEMA_VERSION,
            symbol="EURUSD",
            provider="CLEAN_TEST",
            native_interval="5m",
            price_basis=PriceBasis.MID,
            timestamp_utc=(base_dt + timedelta(minutes=5 * i)).isoformat(),
            retrieved_at_utc=base_dt.isoformat(),
            open=1.1200 + i * 0.0001,
            high=1.1210 + i * 0.0001,
            low=1.1190 + i * 0.0001,
            close=1.1205 + i * 0.0001,
            volume=100.0,
            is_closed=True,
        )
        for i in range(10)
    ]

    success, msg, meta = manager.save_research_dataset(
        dataset_id="test_ds_001",
        symbol="EURUSD",
        candles=candles,
        expected_interval="5m",
        provider="CLEAN_TEST",
    )

    assert success is True
    assert "test_ds_001.json" in msg
    assert meta is not None
    assert meta.observation_count == 10
    assert (temp_research_dir / "datasets" / "test_ds_001.json").exists()
    assert (temp_research_dir / "research_audit_ledger.jsonl").exists()

    # Verify production path was not touched
    assert not Path("data/ledger/test_ds_001.json").exists()


def test_scale_invariant_feature_extractor():
    """Test 2: Proves features are scale-invariant, NATR is normalized by price, and nominal prices are absent."""
    extractor = ScaleInvariantForexFeatureExtractor()

    # Generate synthetic price series
    n = 100
    base_price = 1.1200
    prices = [base_price + 0.0010 * math.sin(i / 5.0) for i in range(n)]
    df = pd.DataFrame({
        "open": prices,
        "high": [p + 0.0005 for p in prices],
        "low": [p - 0.0005 for p in prices],
        "close": [p + 0.0002 for p in prices],
        "volume": [1000.0] * n,
    })

    feat = extractor.extract_features(df)
    assert not feat.empty
    assert len(feat) == n
    assert set(feat.columns) == set(extractor.FEATURE_NAMES)

    # 1. Nominal price columns must be absent
    assert "close" not in feat.columns
    assert "open" not in feat.columns
    assert "ema_9" not in feat.columns

    # 2. NATR must be normalized (ATR / Close), typically around 0.0005 - 0.0020 for FX, NOT nominal 45.0
    natr_vals = feat["natr_14"].values
    assert np.all(natr_vals >= 0.0)
    assert np.all(natr_vals < 0.05), "NATR exceeds plausible Forex relative volatility!"
    # Solves the 1e-35 unnormalized threshold defect: NATR is well above 1e-6
    assert np.median(natr_vals[30:]) > 1e-4

    # 3. All features are finite
    assert np.all(np.isfinite(feat.values))


def test_forex_labeling_engine_three_classes():
    """Test 3: Validates 3-class target labeling with cost threshold."""
    labeler = ForexLabelingEngine(horizon_bars=5, cost_threshold=0.00020)

    # Generate test DataFrame
    # 0..10: flat -> NO_TRADE (1)
    # 10..20: large rally -> BUY (2)
    # 20..30: large drop -> SELL (0)
    prices = [1.1200] * 10 + [1.1200 + 0.0010 * i for i in range(10)] + [1.1300 - 0.0010 * i for i in range(10)]
    df = pd.DataFrame({"close": prices})

    labels = labeler.create_labels(df)
    valid_labels = labels.dropna()

    assert set(valid_labels.unique()).issubset({0.0, 1.0, 2.0})
    assert 2.0 in valid_labels.values, "BUY class missing!"
    assert 0.0 in valid_labels.values, "SELL class missing!"
    assert 1.0 in valid_labels.values, "NO_TRADE class missing!"


def test_purged_walk_forward_splitter_embargo():
    """Test 4: Verifies walk-forward splitting enforces embargo window between train/val and val/test."""
    splitter = PurgedWalkForwardSplitter(train_pct=0.60, val_pct=0.20, embargo_bars=24)
    n = 1000
    splits = splitter.split(n)

    train_start, train_end = splits["train"]
    val_start, val_end = splits["val"]
    test_start, test_end = splits["test"]

    assert train_start == 0
    assert train_end == 600
    # Embargo between train and val
    assert val_start == train_end + 24
    assert (val_start - train_end) == 24

    # Embargo between val and test
    assert test_start == val_end + 24
    assert (test_start - val_end) == 24
    assert test_end == 1000


def test_realistic_cost_evaluator_asymmetric_fills():
    """Test 5: Validates asymmetric execution costs: Ask entry / Bid exit for BUY, Bid entry / Ask exit for SELL."""
    evaluator = ForexRealisticCostEvaluator(spread=0.00020, slippage=0.000025, commission=0.000050)

    preds = np.array([2, 0, 1])  # BUY, SELL, NO_TRADE
    entry_mids = np.array([1.1200, 1.1200, 1.1200])
    exit_mids = np.array([1.1200, 1.1200, 1.1200])  # Flat prices -> only costs incurred

    res = evaluator.evaluate_trades(preds, entry_mids, exit_mids)
    assert res["total_samples"] == 3
    assert res["actions_taken"] == 2
    assert res["coverage_pct"] == round(100.0 * 2 / 3, 2)
    # On flat market, both BUY and SELL must yield negative net returns due to spread + slippage + commission
    assert res["total_net_return"] < 0.0
    assert res["win_rate_pct"] == 0.0


def test_benchmark_baseline_evaluator():
    """Test 6: Validates calculation of majority, class prior, and momentum baselines."""
    evaluator = BenchmarkBaselineEvaluator()

    y_train = np.array([2, 2, 2, 0, 1])  # 60% BUY
    y_test = np.array([2, 0, 2, 1, 2])   # 60% BUY

    # Majority baseline
    maj_acc = evaluator.evaluate_majority_baseline(y_train, y_test)
    assert maj_acc == 0.60

    # Class prior baseline
    prior_res = evaluator.evaluate_class_prior_baseline(y_train, y_test)
    assert "priors" in prior_res
    assert "brier_score" in prior_res
    assert prior_res["priors"][2] == 0.60  # P(BUY) = 0.60
    assert prior_res["priors"][0] == 0.20  # P(SELL) = 0.20
    assert prior_res["priors"][1] == 0.20  # P(NEUTRAL) = 0.20

    # Momentum baseline
    close_prices = np.array([1.1200, 1.1210, 1.1205, 1.1220, 1.1215, 1.1230])
    mom_acc = evaluator.evaluate_momentum_baseline(close_prices, y_test)
    assert 0.0 <= mom_acc <= 1.0


def test_governance_blocklist_in_research_pipeline(temp_research_dir):
    """Test 7: Proves that EURAUD and EURCAD are rejected by the research dataset manager."""
    manager = ResearchDatasetManager(base_dir=temp_research_dir)

    for blocked in ["EURAUD", "EURCAD"]:
        success, msg, _ = manager.save_research_dataset(
            dataset_id=f"blocked_{blocked}",
            symbol=blocked,
            candles=[],
            expected_interval="5m",
            provider="TEST",
        )
        assert success is False
        assert "permanently blocked" in msg


def test_no_lookahead_leakage():
    """Test 8: Verifies that feature values at index t are strictly identical when future bars are truncated."""
    extractor = ScaleInvariantForexFeatureExtractor()
    n = 60
    prices = [1.1200 + 0.0005 * i for i in range(n)]
    df_full = pd.DataFrame({
        "open": prices, "high": [p + 0.0002 for p in prices],
        "low": [p - 0.0002 for p in prices], "close": prices, "volume": [100.0] * n
    })

    feat_full = extractor.extract_features(df_full)

    # Truncate at bar 40
    df_trunc = df_full.iloc[:41].copy()
    feat_trunc = extractor.extract_features(df_trunc)

    # Feature at index 40 must match exactly to 10 decimal places (zero future lookahead leakage)
    for col in extractor.FEATURE_NAMES:
        val_full = feat_full.iloc[40][col]
        val_trunc = feat_trunc.iloc[40][col]
        assert math.isclose(val_full, val_trunc, abs_tol=1e-9), f"Leakage detected in {col}: {val_full} != {val_trunc}"


def test_forex_cost_structure_and_dynamic_hurdles():
    """Test 9 (Phase 10): Validates instrument-specific cost structures and dynamic hurdle calculations."""
    # 1. EURUSD: spread=0.00015, slippage=0.000020, comm=0.000030, margin=0.000100
    cs_eurusd = DEFAULT_FX_COSTS["EURUSD"]
    p_eurusd = 1.1200
    fric_eurusd = cs_eurusd.round_trip_frictional_cost(p_eurusd)
    hurdle_eurusd = cs_eurusd.directional_hurdle(p_eurusd)

    expected_fric = (0.00015 + 2 * 0.000020) / 1.1200 + 2 * 0.000030  # ~ 2.30 bps
    expected_hurdle = expected_fric + 0.000100                         # ~ 3.30 bps
    assert math.isclose(fric_eurusd, expected_fric, abs_tol=1e-7)
    assert math.isclose(hurdle_eurusd, expected_hurdle, abs_tol=1e-7)

    # 2. EURCHF: spread=0.00020, slippage=0.000025, comm=0.000030, margin=0.000100
    cs_eurchf = DEFAULT_FX_COSTS["EURCHF"]
    p_eurchf = 0.9400
    fric_eurchf = cs_eurchf.round_trip_frictional_cost(p_eurchf)
    hurdle_eurchf = cs_eurchf.directional_hurdle(p_eurchf)

    expected_fric_chf = (0.00020 + 2 * 0.000025) / 0.9400 + 2 * 0.000030 # ~ 3.26 bps
    expected_hurdle_chf = expected_fric_chf + 0.000100                   # ~ 4.26 bps
    assert math.isclose(fric_eurchf, expected_fric_chf, abs_tol=1e-7)
    assert math.isclose(hurdle_eurchf, expected_hurdle_chf, abs_tol=1e-7)

    # 3. Zero-cost structure
    cs_zero = ForexCostStructure(spread=0.0, slippage=0.0, commission_rate=0.0, margin_rate=0.0)
    assert cs_zero.round_trip_frictional_cost(1.1000) == 0.0
    assert cs_zero.directional_hurdle(1.1000) == 0.0

    # 4. Invalid input check: non-positive price raises ValueError
    with pytest.raises(ValueError, match="Reference price must be positive"):
        cs_eurusd.round_trip_frictional_cost(0.0)
    with pytest.raises(ValueError, match="Reference price must be positive"):
        cs_eurusd.round_trip_frictional_cost(-1.12)


def test_session_gap_aware_feature_extraction():
    """Test 10 (Phase 10): Validates that session gaps and weekend closures do not distort features."""
    extractor = ScaleInvariantForexFeatureExtractor()

    # Construct series with consecutive bars, a missing bar gap, and a weekend closure gap
    base_t = datetime(2026, 6, 5, 20, 0, tzinfo=UTC) # Friday 16:00 NY = 20:00 UTC
    t_fri = [base_t + timedelta(minutes=5 * i) for i in range(12)] # 12 bars (up to 20:55)
    # Weekend gap: next is Sunday 17:00 NY = 21:00 UTC (June 7) -> 48h gap!
    sun_open = datetime(2026, 6, 7, 21, 0, tzinfo=UTC)
    # Sunday session: 10 bars, with a missing candle gap between index 15 and 16
    t_sun_1 = [sun_open + timedelta(minutes=5 * i) for i in range(4)] # bars 0, 1, 2, 3
    # missing 10m gap: skip 1 candle (delta = 600s)
    t_sun_2 = [sun_open + timedelta(minutes=25 + 5 * i) for i in range(16)] # 16 more bars
    timestamps = t_fri + t_sun_1 + t_sun_2 # total 32 bars

    # Inject a 50-pip weekend gap up: Friday closes at 1.1200, Sunday opens at 1.1250
    p_fri = [1.1200 + 0.0001 * i for i in range(12)]
    p_sun = [1.1250 + 0.0001 * i for i in range(20)]
    prices = p_fri + p_sun
    open_prices = [p_fri[0]] + p_fri[:-1] + [1.1250] + p_sun[:-1]

    df = pd.DataFrame({
        "timestamp": timestamps,
        "open": open_prices,
        "high": [max(o, c) + 0.0002 for o, c in zip(open_prices, prices)],
        "low": [min(o, c) - 0.0002 for o, c in zip(open_prices, prices)],
        "close": prices,
        "volume": [100.0] * len(prices),
    })

    feat = extractor.extract_features(df)
    assert not feat.empty
    assert len(feat) == len(df)

    # 1. At weekend reopening (index 12):
    # log_return_1 must reflect intra-bar return (close vs open), NOT the 50-pip weekend jump
    sun_open_ret1 = feat["log_return_1"].iloc[12]
    expected_intrabar = np.log(prices[12] / open_prices[12])
    assert math.isclose(sun_open_ret1, expected_intrabar, abs_tol=1e-7)

    # 2. NATR at Sunday open must NOT be inflated by a 50-pip weekend gap
    assert feat["natr_14"].iloc[12] < 0.0020, "NATR inflated by weekend jump!"

    # 3. Multi-bar returns: log_return_5 at Sunday open must be 0.0 (no cross-weekend bridging)
    assert feat["log_return_5"].iloc[12] == 0.0

    # 4. Causality across gap: modifying Sunday data must not affect Friday features
    df_mod = df.copy()
    df_mod.loc[12:, "close"] = df_mod.loc[12:, "close"] * 1.10
    df_mod.loc[12:, "high"] = df_mod.loc[12:, "high"] * 1.10
    feat_mod = extractor.extract_features(df_mod)
    for col in extractor.FEATURE_NAMES:
        assert np.allclose(feat.loc[:11, col], feat_mod.loc[:11, col], atol=1e-10)


def test_execution_timing_alignment_next_open():
    """Test 11 (Phase 10): Validates execution timing alignment on bar t+1 open and horizon exit."""
    labeler = ForexLabelingEngine(horizon_bars=2, cost_threshold=0.00020, entry_mode="next_open")

    # Bar 0: prediction emitted at close=1.1200
    # Bar 1: opens at 1.1205 (executable entry price = 1.1205!)
    # Bar 2: intermediate bar
    # Bar 3: opens at 1.1215 (executable exit price for horizon=2 = 1.1215!)
    # Expected return = (1.1215 - 1.1205) / 1.1205 = 0.0010 / 1.1205 = +0.0008925 >= 0.00020 => BUY (2.0)
    df = pd.DataFrame({
        "open":  [1.1195, 1.1205, 1.1210, 1.1215, 1.1190, 1.1180],
        "close": [1.1200, 1.1208, 1.1212, 1.1200, 1.1185, 1.1175],
    })

    labels = labeler.create_labels(df)

    # Bar 0: entry=open[1]=1.1205, exit=open[1+2=3]=1.1215 => forward_ret = +0.0008925 => BUY (2.0)
    assert labels.iloc[0] == 2.0

    # Bar 1: entry=open[2]=1.1210, exit=open[4]=1.1190 => forward_ret = (1.1190 - 1.1210)/1.1210 = -0.001784 <= -0.00020 => SELL (0.0)
    assert labels.iloc[1] == 0.0

    # Bar 2: entry=open[3]=1.1215, exit=open[5]=1.1180 => forward_ret = (1.1180 - 1.1215)/1.1215 = -0.00312 <= -0.00020 => SELL (0.0)
    assert labels.iloc[2] == 0.0

    # Terminal bars (indices 3, 4, 5) have incomplete horizon => MUST BE NaN
    assert np.isnan(labels.iloc[3])
    assert np.isnan(labels.iloc[4])
    assert np.isnan(labels.iloc[5])


def test_momentum_baseline_explicit_index_alignment():
    """Test 12 (Phase 10): Validates momentum baseline on non-trailing and non-contiguous test indices."""
    evaluator = BenchmarkBaselineEvaluator()

    # Price series of 10 bars
    # Past returns (bar t vs bar t-1):
    # t=0: ret=0
    # t=1: 1.1200 -> 1.1210 (UP)
    # t=2: 1.1210 -> 1.1205 (DOWN)
    # t=3: 1.1205 -> 1.1220 (UP)
    # t=4: 1.1220 -> 1.1218 (DOWN)
    # t=5: 1.1218 -> 1.1230 (UP)
    # t=6: 1.1230 -> 1.1235 (UP)
    # t=7: 1.1235 -> 1.1225 (DOWN)
    # t=8: 1.1225 -> 1.1240 (UP)
    # t=9: 1.1240 -> 1.1238 (DOWN)
    close_prices = np.array([1.1200, 1.1210, 1.1205, 1.1220, 1.1218, 1.1230, 1.1235, 1.1225, 1.1240, 1.1238])

    # Suppose test indices are non-trailing and non-contiguous: indices [2, 5, 8]
    # At t=2: past return is DOWN => pred = 0 (SELL)
    # At t=5: past return is UP   => pred = 2 (BUY)
    # At t=8: past return is UP   => pred = 2 (BUY)
    test_indices = np.array([2, 5, 8])

    # Case A: Perfect match test labels: [0, 2, 2]
    y_test_match = np.array([0.0, 2.0, 2.0])
    acc_match = evaluator.evaluate_momentum_baseline(close_prices, y_test_match, test_indices=test_indices)
    assert acc_match == 1.0

    # Case B: Inverted test labels: [2, 0, 0]
    y_test_inverted = np.array([2.0, 0.0, 0.0])
    acc_inv = evaluator.evaluate_momentum_baseline(close_prices, y_test_inverted, test_indices=test_indices)
    assert acc_inv == 0.0

    # Case C: 3-class policy with hurdle
    # Hurdle = 0.0005 (5 pips)
    # t=2: return is (1.1205 - 1.1210)/1.1210 = -0.000446 -> inside [-0.0005, 0.0005] => NO_TRADE (1)
    # t=5: return is (1.1230 - 1.1218)/1.1218 = +0.001070 >= 0.0005 => BUY (2)
    # t=8: return is (1.1240 - 1.1225)/1.1225 = +0.001336 >= 0.0005 => BUY (2)
    y_test_hurdle = np.array([1.0, 2.0, 2.0])
    acc_hurdle = evaluator.evaluate_momentum_baseline(close_prices, y_test_hurdle, test_indices=test_indices, hurdle=0.0005)
    assert acc_hurdle == 1.0


def test_feature_schema_versioning(temp_research_dir):
    """Test 13 (Phase 10): Validates feature schema version constant and metadata verification."""
    # 1. Version constant exists and matches standard
    assert ScaleInvariantForexFeatureExtractor.FEATURE_PIPELINE_VERSION == "FX-FEAT-v1.0.0"

    # 2. Saving a dataset records the schema version and ordered feature names
    manager = ResearchDatasetManager(base_dir=temp_research_dir)
    base_dt = datetime(2026, 10, 19, 10, 0, tzinfo=UTC)
    candles = [
        ForexCandleContractV1(
            schema_version=SCHEMA_VERSION,
            symbol="EURUSD",
            provider="CLEAN_TEST",
            native_interval="5m",
            price_basis=PriceBasis.MID,
            timestamp_utc=(base_dt + timedelta(minutes=5 * i)).isoformat(),
            retrieved_at_utc=base_dt.isoformat(),
            open=1.1200 + i * 0.0001,
            high=1.1210 + i * 0.0001,
            low=1.1190 + i * 0.0001,
            close=1.1205 + i * 0.0001,
            volume=100.0,
            is_closed=True,
        )
        for i in range(5)
    ]

    success, msg, meta = manager.save_research_dataset(
        dataset_id="versioned_ds_001",
        symbol="EURUSD",
        candles=candles,
        expected_interval="5m",
        provider="TEST",
    )
    assert success is True
    assert meta.feature_pipeline_version == "FX-FEAT-v1.0.0"
    assert meta.feature_names == list(ScaleInvariantForexFeatureExtractor.FEATURE_NAMES)

    # 3. Loading the dataset verifies schema compatibility
    loaded = manager.load_research_dataset("versioned_ds_001")
    assert loaded["metadata"]["feature_pipeline_version"] == "FX-FEAT-v1.0.0"

    # 4. Incompatible schema version rejection test
    # Tamper with the saved json file to simulate an incompatible future/legacy version
    ds_file = temp_research_dir / "datasets" / "versioned_ds_001.json"
    import json
    data = json.loads(ds_file.read_text(encoding="utf-8"))
    data["metadata"]["feature_pipeline_version"] = "FX-FEAT-v9.9.9-INCOMPATIBLE"
    ds_file.write_text(json.dumps(data), encoding="utf-8")

    with pytest.raises(ValueError, match="Incompatible feature pipeline version"):
        manager.load_research_dataset("versioned_ds_001")
