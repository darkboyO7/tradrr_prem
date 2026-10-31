# Institutional Live-Trading Readiness Dossier & Authorization Boundary

**Document ID:** `DOSSIER-LIVE-READINESS-2026-001`  
**Generated Date:** `2026-10-22T06:30:00+05:30` (01:00:00 UTC)  
**Project Root:** `D:\ant`  
**Host Environment:** Native Windows 11 Enterprise (AMD64) • Python 3.14.7 (`.venv`)  
**Target Markets:** National Stock Exchange of India (NSE) & Bombay Stock Exchange (BSE) Cash Equities  
**Governing Standard:** Section 18 Master Engineering Specification  
**Operational Status:** **ENGINEERING COMPLETE — LIVE REAL-MONEY TRADING STRICTLY DISABLED**  

---

## 1. Executive Summary & Authorization Boundary Statement

This dossier establishes the formal boundary between autonomous software engineering and live financial execution. All 18 functional engineering subsystems have been constructed, hardened, and verified under automated regression testing. 

> [!CAUTION]
> **MANDATORY AUTHORIZATION BOUNDARY:**  
> Real-money order execution is **hardcoded to disabled**. No live broker API keys are configured, and order routing to real exchange accounts is physically blocked in the codebase.  
> In compliance with Section 18, the system is **NOT declared 100% live-ready for autonomous execution** because:
> 1. The mandatory 95% out-of-sample prediction accuracy target has not yet been achieved (measured OOS overall accuracy is 54.98%, with active trade win rate of 63.03%).
> 2. The 60 contiguous calendar days of prospective paper-trading evidence required by `policies/MODEL_EVALUATION_AND_PROMOTION_POLICY.md` remain to be accumulated during live exchange trading hours.

---

## 2. Reproducible Test Suite Evidence

- **Regression Test Command:** `.venv\Scripts\python.exe -m pytest -q`
- **Total Test Files:** `105` test files
- **Total Test Items:** **532 passed, 0 failed, 2 warnings in 99.4 seconds (100% Pass Rate)**
- **Test Isolation:** Enforced via `tests/test_test_isolation.py` (read-only traps prevent test runs from altering production ledgers or champion model weights).
- **Process Locking & Recovery:** Enforced via `tests/test_resolver_ownership_and_single_instance.py` (Windows kernel `msvcrt.locking` on `data/.desktop_app.lock`).

---

## 3. Verified Market Data Provenance & Ingestion Requirements

| Venue / Instrument Class | Data Source / Provider | Data Type | Latency | Data Quality Status |
| :--- | :--- | :---: | :---: | :---: |
| **NSE Cash Equities (3,550+)** | `TRADINGVIEW_NSE_LIVE` & `GROWW_NSE` | Real tick quotes & authentic 5m OHLCV bars | 280–360 ms | **VERIFIED FRESH** (Stale quotes > 30s rejected) |
| **BSE Cash Equities (5,500+)** | `TRADINGVIEW_BSE_LIVE` & `GROWW_BSE` | Real tick quotes & authentic 5m OHLCV bars | 250–330 ms | **VERIFIED FRESH** (Cross-listed scrips linked via ISIN) |
| **Persistent Equity Cache** | `data/universe/equity_candles_cache.json` | Authentic 5m bars for liquid equities | < 1 ms | **VERIFIED INTACT** (Sub-millisecond offline recovery) |
| **Quotex Spot Crypto Proxies** | `Kraken / Coinbase Public Feeds` | Public spot cryptocurrency quotes (`BTCUSD_QX`) | 120–200 ms | **RESEARCH ONLY** (Isolated from cash equities) |
| **Quotex Synthetic OTC** | None (No public broker API) | Broker-proprietary synthetic pricing | N/A | **FAIL-CLOSED** (Emits `QUOTEX_DATA_UNAVAILABLE` & `NO_TRADE`) |
| **Forex Pairs** | Deliberately Frozen | Legacy historical reference | N/A | **OUT OF SCOPE** (Deliberately frozen) |

---

## 4. Empirical Out-of-Sample Accuracy & Model Governance

Detailed findings from `reports/model_evaluation.md` evaluated on untouched out-of-sample data across liquid NSE equities:

| Evaluation Metric | Observed Value | Required Acceptance Hurdle | Status |
| :--- | :---: | :---: | :---: |
| **Overall 3-Class Accuracy** | **54.98%** | $\ge 95.0\%$ | `ACCURACY_TARGET_NOT_MET` |
| **Wilson 95% CI Lower Bound** | **48.23%** | $\ge 95.0\%$ | `ACCURACY_TARGET_NOT_MET` |
| **Wilson 95% CI Upper Bound** | **61.54%** | N/A | Documented |
| **Active Trade Win Rate** | **63.03%** | $\ge 52.0\%$ | **PASS** |
| **Active Net Expectancy** | **+0.3192%** | $> +0.020\%$ (+2.0 bps) | **PASS** |
| **Multiclass Brier Score** | **0.6173** | $< 0.6667$ (Random Guess) | **PASS** |
| **Expected Calibration Error (ECE)**| **0.1050** | $< 0.1500$ | **PASS** |
| **Coverage Rate** | **100.0%** (Unconstrained) | $> 25.0\%$ | **PASS** |
| **Purged Embargo Enforcement** | **3 bars (15m)** | $\ge \text{lookahead\_bars}$ | **PASS (Zero Leakage)** |
| **Model Production Readiness** | `is_production_ready: false` | Explicit Operator Approval | **GOVERNED STAGING** |

### 3x3 Confusion Matrix (OOS Evaluation)
```
                Predicted SELL    Predicted NEUTRAL    Predicted BUY
True SELL     :       55                  0                 36
True NEUTRAL  :       22                  0                 15
True BUY      :       22                  0                 61
```

---

## 5. Transaction Costs, Slippage & Risk Bounds

The platform implements institutional-grade friction modeling:
- **Indian Equities Total Friction Budget:** `6.0 bps` (0.060%) per active round-trip trade:
  * Statutory Securities Transaction Tax (STT): 2.5 bps
  * Exchange Turnover & SEBI Charges: 0.7 bps
  * Stamp Duty & GST: 0.8 bps
  * Institutional Brokerage: 1.0 bps
  * Modeled Market Impact / Slippage: 1.0 bps
- **Spread Treatment:** Spread and slippage are embedded directly into simulated entry (Ask + slippage) and exit (Bid - slippage) prices; zero double-counting.
- **Risk & Exposure Controls:**
  * Maximum Portfolio Drawdown Threshold: `4.00%` (Breach triggers emergency circuit breaker).
  * Per-Trade Capital Allocation: Calculated dynamically via the Desktop Risk Calculator (`0.5%` to `2.0%` of account equity).
  * Minimum Reward-to-Risk Hurdle: $R:R \ge 1.5$.

---

## 6. Prospective Paper-Trading & Outcome Ledgers

- **Operational Outcome Store:** [`data/ledger/prediction_outcomes.jsonl`](file:///D:/ant/data/ledger/prediction_outcomes.jsonl) (2,713 historical records; lines 1–407 byte-for-byte identical to immutable backup baseline).
- **Pending Predictions Queue:** [`data/ledger/pending_predictions.jsonl`](file:///D:/ant/data/ledger/pending_predictions.jsonl) (164 active records awaiting forecast horizon maturation).
- **Prospective Paper Worker Status:** `STOPPED` (Requires operator activation via `scripts/run_paper_worker.py`).
- **Health Reporting:** Integrated daily health check via [`run_health_check.bat`](file:///D:/ant/run_health_check.bat).

---

## 7. Cryptographic Baseline Verification

All 4 protected baseline assets verified byte-for-byte intact:
1. `models_store/lgbm_candidate_v20261020_044046/v20261020_044046/model.joblib`:  
   `D955BE760504E9F690B6D3C21D9707FB7AD56A39C3BA5F03FE7AEBB833AC45C2`
2. `models_store/lgbm_candidate_v20261020_044046/v20261020_044046/metadata.json`:  
   `EBF940958DE5F617E359CDDAF802570CE0BF2E05A91F41A7350B0EC952F4307E`
3. `data/ledger/prediction_outcomes.jsonl.bak`:  
   `D6EC09C9B2838DF563F06E06EB21161ACB725534B46D365C913CE48430BB827F`
4. `policies/MODEL_EVALUATION_AND_PROMOTION_POLICY.md`:  
   `126680BF93EA477A76908B28C2147DE78B722FD20559534CB25BF1B36919BD31`

---

## 8. Rollback, Recovery & Failure Modes

1. **Stale Market Feed (>30s):** System sets `is_stale = True`, updates badge to `STALE DATA`, and abstains with `NO_TRADE`.
2. **Missing Candles:** Ingestion automatically falls back to `data/universe/equity_candles_cache.json`.
3. **Unexpected Application Termination:** Single-instance lockfile (`data/.desktop_app.lock`) is released by Windows OS kernel. Restart automatically re-indexes pending predictions.
4. **Model Rollback Procedure:** To revert a champion model, set the active champion pointer in `data/learning/learning_controller_state.json` to the target version directory and verify SHA-256 against `reports/protected_assets_manifest.json`.

---

## 9. Proposed Limited Rollout & Monitoring Plan

When live trading authorization is eventually granted, execution must proceed through three sequential gates:
1. **Phase 1: Shadow Desktop Monitoring (09:15–15:30 IST):** Run desktop application in view-only mode to verify live feed latency, session open/close countdown, and indicator alignment against actual exchange quotes.
2. **Phase 2: Automated Paper Execution:** Run background paper worker (`scripts/run_paper_worker.py`) for 60 contiguous calendar days across liquid NSE stocks (TCS, RELIANCE, INFY) to accumulate 100+ resolved prospective predictions.
3. **Phase 3: Controlled Micro-Lot Live Pilot:** Only after Phase 2 achieves verified positive expectancy and operator grants explicit written approval, deploy a single micro-lot instrument (1 share of liquid stock) with hard risk limits.
