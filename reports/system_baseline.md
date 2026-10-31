# System Baseline and Forensic Audit Report

**Audit Date:** 2026-10-20  
**Project Root:** `D:\ant`  
**Host Platform:** Windows 11 Enterprise (AMD64), Python 3.14.7  
**Runtime Environment:** Native Windows local virtual environment (`D:\ant\.venv`)  
**Scope:** NSE Equities, BSE Equities, Quotex Research Only (Forex strictly out of scope)  
**Execution Safety:** Paper worker STOPPED; Real-money trading PERMANENTLY DISABLED  

---

## 1. Repository Structure and Git Status

### 1.1 Git Tracking State
- **Current Branch:** `master`
- **Head Commit:** `f9e17a4` — *Add EXE binaries via Git LFS + .gitattributes*
- **Previous Commit:** `c0560e3` — *Initial commit: Trading Intelligence Platform v1.0*
- **Modified Working Tree Files:** Uncommitted development improvements in `desktop/`, `data/ingestion/`, `learning/`, `models/`, and `tests/`.
- **Untracked Directories:** `data/contracts/`, `data/paper_trading/`, `models_store/`, `policies/`, `reports/`, `research/`.
- **Governance Constraint:** Zero destructive Git operations (no reset, checkout, clean, or commit without user authorization).

### 1.2 Core Directory Layout
- `backtesting/`: Walk-forward backtesting engines and event-driven trade simulators.
- `core/`: Event buses, process-locking primitives (`ProcessFileLock`), symbol resolvers, and recovery snapshot managers.
- `data/`: Ingestion pipelines for NSE (`data/ingestion/nse.py`), BSE (`data/ingestion/bse.py`), Quotex proxies (`data/ingestion/quotex.py`), operational ledgers (`data/ledger/`), and isolated paper trading storage (`data/paper_trading/`).
- `desktop/`: Native PySide6 Windows desktop application (`desktop/app.py`, `desktop/main_window.py`).
- `features/`: Multi-timeframe indicator pipelines (RSI, ADX, MACD, Bollinger Bands, ATR, VWAP, Order Flow Imbalance).
- `learning/`: Autonomous prediction outcome resolution service, learning controller, and model comparison pipelines.
- `models/`: Machine learning classifiers (`LightGBMPredictor`, LightGBM / XGBoost wrapper with Bayesian smoothing), baseline statistical models, and model registry.
- `paper_trading/`: Dedicated multi-market paper trading engine (`worker.py`, `prediction_store.py`, `health_report.py`).
- `reports/`: Audit baselines, manifests, implementation evidence, and release documentation.
- `tests/`: 103 test files covering 521 unit, integration, contract, and governance tests.

---

## 2. Python Environment & Dependency Baseline

- **Python Executable:** `D:\ant\.venv\Scripts\python.exe`
- **Python Version:** `3.14.7 (tags/v3.14.7:823f032, Aug  5 2026, 10:51:32) [MSC v.1944 64 bit (AMD64)]`
- **Core Frameworks:**
  - GUI: `PySide6` 6.12.0
  - ML & Numerical: `lightgbm` 4.7.0, `xgboost` 3.4.1, `scikit-learn` 1.9.1, `numpy` 2.5.3, `pandas` 3.0.6, `scipy` 1.18.1
  - Testing & Quality: `pytest` 9.1.1, `pytest-asyncio` 1.4.0, `ruff` 0.16.10, `mypy` 2.4.0
  - System & Web: `pywin32` 312, `httpx` 0.28.1, `aiohttp` 3.14.3, `fastapi` 0.142.2, `websockets` 17.2

---

## 3. Forensic Investigation of Reported Baseline Issues

### 3.1 Test Count Discrepancy (Claimed 520 vs True Baseline)
- **Investigation:** Pytest collection was audited across all test directories. The genuine baseline test suite collected 513 tests across 101 modules. The previous report erroneously recorded 520 across 104 files.
- **Current State:** 8 new unit tests were added across 2 new test modules:
  - `tests/test_multi_market_paper_worker.py`: 4 tests
  - `tests/test_resolver_ownership_and_single_instance.py`: 4 tests
  Total collected tests: **521 tests across 103 test files**. All 521 pass (100% pass rate in 78.35s).

### 3.2 Two Reported Test Failures (Protected Ledger Hash & Daily Health Report)
- **Investigation:** `tests/test_forex_model_compatibility_and_contract.py` and `paper_trading/health_report.py` asserted that the primary live ledger `data/ledger/prediction_outcomes.jsonl` matched a static hardcoded SHA-256 hash. Because the live ledger is operational and append-only, any valid maturation of pending predictions broke the hash assertion.
- **Remediation:** Refactored verification to decouple truly immutable assets from dynamic operational ledgers:
  - Immutable assets (`model.joblib`, `metadata.json`, `prediction_outcomes.jsonl.bak`, `MODEL_EVALUATION_AND_PROMOTION_POLICY.md`) are verified against strict SHA-256 cryptographic hashes.
  - Operational ledgers (`prediction_outcomes.jsonl`) are verified via prefix preservation (lines 1–407 byte-for-byte match `.bak`), strict JSONL schema compliance, monotonic timestamps (`resolved_at >= predicted_at`), and valid outcome attributes.

### 3.3 Prediction Outcomes Record Inflation (739 to 764)
- **Investigation:** The production ledger grew from 739 rows to 756, and subsequently to 764 rows.
- **Root Cause Identified:** When user/tests closed the desktop application window, `desktop/main_window.py` intercepted the event and minimized to the Windows System Tray instead of quitting. This left 6 orphaned background `pythonw.exe -m desktop.app` processes running headless (PIDs: 15912, 4692, 9412, 12224, 2796, 12952). Each running instance had a background thread `PredictionLedgerService.start_background_resolver` that woke up every 15 minutes and resolved pending predictions.
- **Forensic Verification:** All 25 new records (740–764) are valid JSONL, have unique prediction IDs, have valid prices, and strictly respect causal timing (`resolved_at >= predicted_at`). Zero corruption occurred.
- **Remediation Implemented:** Built `core/system_lock.py` (`ProcessFileLock`), single-instance guards on `desktop/app.py`, and single-owner resolver locking on `learning/ledger.py`. Stale processes are awaiting explicit user shutdown approval.

### 3.4 Frozen Champion Model Incompleteness
- **Investigation:** Candidate `models_store/lgbm_candidate_v20261020_044046/v20261020_044046` was trained on binary classes `[0, 2]` (DOWN and UP), completely lacking training samples for class `1` (NEUTRAL). Furthermore, `metadata.json` explicitly marks `"is_production_ready": false`.
- **Safety Handling:** Bayesian Dirichlet smoothing in `models/ml_classifier.py` prevents array out-of-bounds crashes, but does not constitute true 3-class learning. Model promotion remains strictly **BLOCKED**.

### 3.5 Forex-Dependent Paper-Trading Worker
- **Investigation:** `paper_trading/worker.py` was hardcoded to accept only `EURUSD` and `EURCHF`.
- **Remediation Delivered:** Refactored `paper_trading/worker.py` and `scripts/run_paper_worker.py` to route dynamically across `NSEDataProvider`, `BSEDataProvider`, and `QuotexMarketDataAdapter` (crypto spot proxy).

### 3.6 Missing Direct Quotex Contract Data
- **Investigation:** Quotex does not provide an authorized, public REST/WebSocket API for direct contract settlement.
- **Resolution:** Strict classification enforced: Quotex operates in research mode only via underlying spot proxies (e.g., Coinbase/Kraken BTC/USD). Synthetic Quotex OTC instruments (`MSFT_OTC`, `EURUSD_OTC`) are rejected fail-closed with `QUOTEX_DATA_UNAVAILABLE`.

---

## 4. Protected Assets Baseline Manifest

| Asset Path | Classification | Size (bytes) | SHA-256 Hash |
| :--- | :--- | :--- | :--- |
| `models_store/lgbm_candidate_v20261020_044046/v20261020_044046/model.joblib` | Immutable Model Weights | 39,636 | `d955be760504e9f690b6d3c21d9707fb7ad56a39c3ba5f03fe7aebb833ac45c2` |
| `models_store/lgbm_candidate_v20261020_044046/v20261020_044046/metadata.json` | Immutable Model Metadata | 1,168 | `ebf940958de5f617e359cddaf802570ce0bf2e05a91f41a7350b0ec952f4307e` |
| `data/ledger/prediction_outcomes.jsonl.bak` | Historical Outcomes Backup | 549,821 | `d6ec09c9b2838df563f06e06eb21161acb725534b46d365c913ce48430bb827f` |
| `policies/MODEL_EVALUATION_AND_PROMOTION_POLICY.md` | Governance Policy | 11,526 | `126680bf93ea477a76908b28c2147de78b722fd20559534cb25bf1b36919bd31` |
| `data/ledger/prediction_outcomes.jsonl` | Append-Only Live Ledger | 2,145,028 | `5ef926eea3ecd5c3fe39cc3c0a81e522a5357b165e3c791a3e8bac6e7c1d19bc` (764 records) |

---

## 5. Active Process State

| PID | Command Line | Start Time | Role / Owner | Action |
| :--- | :--- | :--- | :--- | :--- |
| **15912** | `pythonw.exe -m desktop.app` | 09:47:22 AM | Primary GUI instance 1 | Awaiting safe shutdown approval |
| **4692** | `pythonw.exe -m desktop.app` | 09:47:22 AM | Child helper process 1 | Awaiting safe shutdown approval |
| **9412** | `pythonw.exe -m desktop.app` | 01:59:40 PM | Primary GUI instance 2 | Awaiting safe shutdown approval |
| **12224** | `pythonw.exe -m desktop.app` | 01:59:40 PM | Child helper process 2 | Awaiting safe shutdown approval |
| **2796** | `pythonw.exe -m desktop.app` | 04:48:15 PM | Primary GUI instance 3 | Awaiting safe shutdown approval |
| **12952** | `pythonw.exe -m desktop.app` | 04:48:15 PM | Child helper process 3 | Awaiting safe shutdown approval |

**Paper Worker:** Inactive (`.paper_worker.lock` PID 20408 is terminated; 0 active trading workers).
