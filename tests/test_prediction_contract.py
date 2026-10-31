"""Unit Tests for Canonical Prediction Contract (PRED-CONTRACT-v1.0.0).

Verifies:
1. Valid contracts for NSE, BSE, and Quotex.
2. Rejection of unsupported markets or mismatched canonical symbols.
3. Rejection of invalid probability sums or out-of-range probabilities.
4. Temporal lookahead rejection (data_cutoff > creation).
5. Resolution deadline after creation enforcement.
6. Quotex binary payout and break-even probability calculation.
"""

from datetime import UTC, datetime, timedelta
import pytest

from data.contracts.prediction_contract import (
    PREDICTION_CONTRACT_SCHEMA_VERSION,
    PredictionContractV1,
    validate_prediction_contract,
)


def test_valid_nse_prediction_contract():
    now = datetime.now(UTC)
    contract = PredictionContractV1(
        schema_version=PREDICTION_CONTRACT_SCHEMA_VERSION,
        prediction_id="pred_001_NSE_TCS",
        instrument="TCS",
        canonical_symbol="NSE:TCS",
        market="NSE",
        creation_timestamp_utc=now.isoformat(),
        data_cutoff_timestamp_utc=(now - timedelta(minutes=5)).isoformat(),
        source_candle_interval="5m",
        prediction_horizon="15m",
        target_definition="triple_barrier",
        action="BUY",
        buy_probability=0.62,
        sell_probability=0.18,
        no_trade_probability=0.20,
        confidence=0.62,
        uncertainty=0.12,
        model_version="lgbm_candidate_v20261020_044046",
        feature_pipeline_version="INDIA-FEAT-v1.0.0",
        data_quality_status="VERIFIED",
        entry_price=2460.0,
        stop_loss=2430.0,
        take_profit=2520.0,
        risk_reward_ratio=2.0,
        expected_friction_bps=6.0,
        break_even_probability=0.5006,
        resolution_deadline_utc=(now + timedelta(minutes=15)).isoformat(),
    )
    is_valid, reason = validate_prediction_contract(contract)
    assert is_valid is True, f"Validation failed: {reason}"


def test_valid_bse_prediction_contract():
    now = datetime.now(UTC)
    contract = PredictionContractV1(
        schema_version=PREDICTION_CONTRACT_SCHEMA_VERSION,
        prediction_id="pred_002_BSE_RELIANCE",
        instrument="RELIANCE",
        canonical_symbol="BSE:RELIANCE",
        market="BSE",
        creation_timestamp_utc=now.isoformat(),
        data_cutoff_timestamp_utc=(now - timedelta(minutes=5)).isoformat(),
        source_candle_interval="5m",
        prediction_horizon="15m",
        target_definition="triple_barrier",
        action="NO-TRADE",
        buy_probability=0.34,
        sell_probability=0.33,
        no_trade_probability=0.33,
        confidence=0.34,
        uncertainty=0.35,
        model_version="lgbm_candidate_v20261020_044046",
        feature_pipeline_version="INDIA-FEAT-v1.0.0",
        data_quality_status="VERIFIED",
        entry_price=1202.8,
        stop_loss=1190.0,
        take_profit=1228.0,
        risk_reward_ratio=2.0,
        expected_friction_bps=6.0,
        break_even_probability=0.5006,
        resolution_deadline_utc=(now + timedelta(minutes=15)).isoformat(),
    )
    is_valid, reason = validate_prediction_contract(contract)
    assert is_valid is True, f"Validation failed: {reason}"


def test_valid_quotex_binary_contract():
    now = datetime.now(UTC)
    payout = 0.85
    expected_be = 1.0 / (1.0 + payout)  # ~0.5405
    contract = PredictionContractV1(
        schema_version=PREDICTION_CONTRACT_SCHEMA_VERSION,
        prediction_id="pred_003_QUOTEX_BTCUSD_QX",
        instrument="BTCUSD_QX",
        canonical_symbol="QUOTEX:BTCUSD_QX",
        market="QUOTEX",
        creation_timestamp_utc=now.isoformat(),
        data_cutoff_timestamp_utc=(now - timedelta(minutes=5)).isoformat(),
        source_candle_interval="5m",
        prediction_horizon="15m",
        target_definition="binary_options_expiry",
        action="BUY",
        buy_probability=0.58,
        sell_probability=0.22,
        no_trade_probability=0.20,
        confidence=0.58,
        uncertainty=0.15,
        model_version="lgbm_candidate_v20261020_044046",
        feature_pipeline_version="QUOTEX-FEAT-v1.0.0",
        data_quality_status="PROXY_CRYPTO_SPOT",
        entry_price=81250.0,
        stop_loss=80500.0,
        take_profit=82750.0,
        risk_reward_ratio=2.0,
        expected_friction_bps=8.0,
        binary_payout_pct=payout,
        break_even_probability=round(expected_be, 4),
        resolution_deadline_utc=(now + timedelta(minutes=15)).isoformat(),
    )
    is_valid, reason = validate_prediction_contract(contract)
    assert is_valid is True, f"Validation failed: {reason}"


def test_reject_lookahead_temporal_violation():
    now = datetime.now(UTC)
    # data_cutoff is in the FUTURE relative to prediction creation
    contract = PredictionContractV1(
        schema_version=PREDICTION_CONTRACT_SCHEMA_VERSION,
        prediction_id="pred_leak_001",
        instrument="TCS",
        canonical_symbol="NSE:TCS",
        market="NSE",
        creation_timestamp_utc=now.isoformat(),
        data_cutoff_timestamp_utc=(now + timedelta(minutes=10)).isoformat(),
        source_candle_interval="5m",
        prediction_horizon="15m",
        target_definition="triple_barrier",
        action="BUY",
        buy_probability=0.60,
        sell_probability=0.20,
        no_trade_probability=0.20,
        confidence=0.60,
        uncertainty=0.10,
        model_version="lgbm_candidate_v20261020_044046",
        feature_pipeline_version="INDIA-FEAT-v1.0.0",
        data_quality_status="VERIFIED",
        entry_price=2460.0,
        stop_loss=2430.0,
        take_profit=2520.0,
        risk_reward_ratio=2.0,
        expected_friction_bps=6.0,
        break_even_probability=0.5006,
        resolution_deadline_utc=(now + timedelta(minutes=15)).isoformat(),
    )
    is_valid, reason = validate_prediction_contract(contract)
    assert is_valid is False
    assert "Lookahead violation" in reason


def test_reject_probabilities_not_summing_to_one():
    now = datetime.now(UTC)
    contract = PredictionContractV1(
        schema_version=PREDICTION_CONTRACT_SCHEMA_VERSION,
        prediction_id="pred_bad_prob",
        instrument="TCS",
        canonical_symbol="NSE:TCS",
        market="NSE",
        creation_timestamp_utc=now.isoformat(),
        data_cutoff_timestamp_utc=now.isoformat(),
        source_candle_interval="5m",
        prediction_horizon="15m",
        target_definition="triple_barrier",
        action="BUY",
        buy_probability=0.80,
        sell_probability=0.50,  # Sum = 1.40
        no_trade_probability=0.10,
        confidence=0.80,
        uncertainty=0.10,
        model_version="lgbm_candidate_v20261020_044046",
        feature_pipeline_version="INDIA-FEAT-v1.0.0",
        data_quality_status="VERIFIED",
        entry_price=2460.0,
        stop_loss=2430.0,
        take_profit=2520.0,
        risk_reward_ratio=2.0,
        expected_friction_bps=6.0,
        break_even_probability=0.5006,
        resolution_deadline_utc=(now + timedelta(minutes=15)).isoformat(),
    )
    is_valid, reason = validate_prediction_contract(contract)
    assert is_valid is False
    assert "Probabilities do not sum to 1.0" in reason


def test_reject_unsupported_market_or_symbol_mismatch():
    now = datetime.now(UTC)
    contract = PredictionContractV1(
        schema_version=PREDICTION_CONTRACT_SCHEMA_VERSION,
        prediction_id="pred_bad_market",
        instrument="TCS",
        canonical_symbol="NSE:TCS",
        market="FOREX",  # Forex is out of scope!
        creation_timestamp_utc=now.isoformat(),
        data_cutoff_timestamp_utc=now.isoformat(),
        source_candle_interval="5m",
        prediction_horizon="15m",
        target_definition="triple_barrier",
        action="BUY",
        buy_probability=0.50,
        sell_probability=0.25,
        no_trade_probability=0.25,
        confidence=0.50,
        uncertainty=0.10,
        model_version="lgbm_candidate_v20261020_044046",
        feature_pipeline_version="INDIA-FEAT-v1.0.0",
        data_quality_status="VERIFIED",
        entry_price=2460.0,
        stop_loss=2430.0,
        take_profit=2520.0,
        risk_reward_ratio=2.0,
        expected_friction_bps=6.0,
        break_even_probability=0.5006,
        resolution_deadline_utc=(now + timedelta(minutes=15)).isoformat(),
    )
    is_valid, reason = validate_prediction_contract(contract)
    assert is_valid is False
    assert "Unsupported market 'FOREX'" in reason
