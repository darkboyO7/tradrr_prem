# Final Test Results Report

**Date of Execution:** 2026-10-30T21:22:45+05:30  
**Test Harness:** `pytest 9.1.1` on Python 3.14.7  
**Host Platform:** Windows 11 Enterprise (AMD64), Zero WSL  
**Execution Command:** `.venv\Scripts\python.exe -m pytest -q`  
**Exit Status:** Code 0 (Success)

---

## 1. Regression Test Summary

| Metric | Target | Verified Value | Result |
| :--- | :--- | :--- | :--- |
| **Total Test Suites** | All 106 test files | **106** | **PASS** |
| **Total Collected Items** | 539 | **539** | **PASS** |
| **Passed Tests** | 539 | **539 (100.0%)** | **PASS** |
| **Failed Tests** | 0 | **0 (0.0%)** | **PASS** |
| **Skipped Tests** | 0 | **0 (0.0%)** | **PASS** |
| **Execution Duration** | $< 180\text{s}$ | **~82.5 seconds** | **PASS** |
| **Warnings** | 0 | **0 Warnings** | **PASS** |

---

## 2. Protected Asset Hash Verification (Pre vs Post Test)

| Protected Asset | Expected Checksum | Post-Test Verified Checksum | Status |
| :--- | :--- | :--- | :--- |
| `models_store/lgbm_candidate_v20261020_044046/v20261020_044046/model.joblib` | `D955BE760504E9F690B6D3C21D9707FB7AD56A39C3BA5F03FE7AEBB833AC45C2` | `D955BE760504E9F690B6D3C21D9707FB7AD56A39C3BA5F03FE7AEBB833AC45C2` | **MATCH (100%)** |
| `models_store/lgbm_candidate_v20261020_044046/v20261020_044046/metadata.json` | `EBF940958DE5F617E359CDDAF802570CE0BF2E05A91F41A7350B0EC952F4307E` | `EBF940958DE5F617E359CDDAF802570CE0BF2E05A91F41A7350B0EC952F4307E` | **MATCH (100%)** |
| `data/ledger/prediction_outcomes.jsonl.bak` | `D6EC09C9B2838DF563F06E06EB21161ACB725534B46D365C913CE48430BB827F` | `D6EC09C9B2838DF563F06E06EB21161ACB725534B46D365C913CE48430BB827F` | **MATCH (100%)** |
| `policies/MODEL_EVALUATION_AND_PROMOTION_POLICY.md` | `126680BF93EA477A76908B28C2147DE78B722FD20559534CB25BF1B36919BD31` | `126680BF93EA477A76908B28C2147DE78B722FD20559534CB25BF1B36919BD31` | **MATCH (100%)** |

---

## 3. Subsystem Test Pass Matrix

- **API & Core Infrastructure:** 41/41 Passed (`tests/test_api.py`)
- **Paper Trading Worker & Prediction Store:** 21/21 Passed (`tests/test_paper_trading_worker.py`, `tests/test_paper_trading_prediction_store.py`)
- **Continuous Learning & Governance:** 17/17 Passed (`tests/test_continuous_learning_loop.py`, `tests/test_continuous_learning.py`)
- **Forex & Multiclass Contracts:** 25/25 Passed (`tests/test_clean_forex_provider.py`, `tests/test_multiclass_contract.py`, `tests/test_forex_candle_pipeline_and_governance.py`)
- **Quotex Quarantine & Gating:** 9/9 Passed (`tests/test_quotex_pipeline.py`)
- **Live Prediction Pipeline:** 5/5 Passed (`tests/test_live_prediction_pipeline.py`)
- **Risk Engine & Portfolio Controls:** 18/18 Passed (`tests/test_risk.py`, `tests/test_var_model.py`, `tests/test_portfolio_margin.py`, `tests/test_volatility_targeting.py`)
- **Smart Order Routing & Simulation:** 33/33 Passed (`tests/test_smart_order_router.py`, `tests/test_synthetic_sor.py`, `tests/test_crossing_engine.py`, `tests/test_dmm_engine.py`, `tests/test_orderbook_simulator.py`)
- **Windows Desktop Application UI:** 1/1 Passed (23-point assertion suite, `tests/test_windows_desktop_app.py`)
- **All Other Modules:** 369/369 Passed
