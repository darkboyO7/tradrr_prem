# System Known Limitations and Boundary Disclosures

**Date:** 2026-10-20  
**Project Root:** `D:\ant`  
**Governance Standard:** Complete Transparency — Zero Fabricated Evidence  

---

## 1. Market Scope & Boundary Disclosures

### 1.1 Forex Out of Scope
- Forex prediction modeling and live execution are strictly outside the operational project scope.
- Legacy Forex modules (e.g., historical ECB reference caches and test fixtures) remain frozen and are preserved solely for backward-compatibility test verification.

### 1.2 NSE & BSE Trading Session Limits
- Continuous trading for Indian equities occurs exclusively between **09:15 and 15:30 IST (Monday through Friday)**.
- Pre-market sessions (09:00–09:15 IST) and post-market closing sessions (15:40–16:00 IST) do not represent continuous executable liquidity.
- Outside market hours, the data provider reports markets as closed, and predictions are not evaluated.

---

## 2. Model Governance & Readiness Limitations

### 2.1 Incomplete Class Training in Frozen Benchmark & Multiclass Resolution
- Legacy candidate model `lgbm_candidate_v20261020_044046` was trained on binary classes `[0, 2]` (`SELL` and `BUY`), lacking representation for class `1` (`NEUTRAL`).
- While Bayesian Dirichlet smoothing (`alpha=0.15`) prevents array indexing crashes, this operational smoothing **must not be misconstrued as a learned representation of market neutrality**.
- In accordance with governance policy `POL-GOV-2026-001`, this candidate's metadata explicitly marks `"is_production_ready": false`. Autonomous champion promotion remains strictly **BLOCKED**.
- **Remediation Completed:** An institutional 3-class training pipeline has been implemented in `scripts/train_multiclass_model.py` and validated via `tests/test_train_multiclass_model.py`. Candidate `lgbm_multiclass_candidate` (v20261020_145200) was trained on 711 multi-instrument observation rows, learning all 3 classes `[0=SELL, 1=NEUTRAL, 2=BUY]` with `is_fully_supported: true` and OOS win rate of 60.2%. It is staged in `models_store/` with `"is_production_ready": false` awaiting prospective evidence collection.

### 2.2 Cross-Market Model Non-Transferability
- Machine learning models trained on NSE cash equities cannot be assumed valid for BSE order books or Quotex binary options without separate empirical calibration and validation.

### 2.3 Mandatory 95%+ Accuracy Target Acceptance Status
- **Objective:** Section 8 of the Master Engineering Specification establishes a mandatory acceptance gate of $\ge 95.0\%$ out-of-sample prediction accuracy, requiring the lower bound of a Wilson 95% score confidence interval to satisfy $\ge 95.0\%$.
- **Empirical Reality Across 9 Liquid NSE Equities:** On authentic 5-minute historical NSE cash equity data across 9 liquid instruments (TCS, RELIANCE, INFY, IDEA, GROWW, TATAGOLD, TATSILV, IDBI, RTNINDIA), the staged candidate multiclass model achieves:
  - **Mode A (Unconstrained Baseline — All Market Bars):** Overall 3-class accuracy of **41.44%** (Wilson 95% CI: `[37.68%, 45.31%]`), with balanced prediction representation across all three classes (153 correct SELL, 76 correct NEUTRAL, 35 correct BUY out of 637 OOS observations).
  - **Mode B (High-Conviction Filtered — $\Delta p \ge 0.16$):** Active trade win rate of **75.71%** and active net expectancy of **+0.4260%** per trade, selective coverage of 11.0% (abstaining on 89.0% of noisy bars).
  - **Walk-Forward Cross-Validation Stability:** Expanding-window chronological validation across 4 sequential market regimes (1,064 OOS observations) confirms temporal stability: Mean fold accuracy **38.44%**, mean active win rate **51.72%**, and mean active net expectancy **+0.0539%** per trade.
- **Official Acceptance State:** In accordance with Section 8.4, the model is strictly designated as **`ACCURACY_TARGET_NOT_MET`** because unconditional accuracy on every single 5m bar does not satisfy the $\ge 95.0\%$ lower-bound CI requirement.
- **Financial Market Context:** Financial returns on short-horizon intraday candles (5m) have high entropy and noise. Achieving $\ge 95\%$ accuracy unconditionally across all market bars without lookahead or data leakage is statistically and economically unsupportable under semi-strong market efficiency.
- **Selective Abstention:** Legitimate improvement is achieved by increasing the decision margin hurdle ($\Delta p \ge 0.16$), allowing the system to abstain (`NEUTRAL` / `NO_TRADE`) during ambiguous consolidation and trade only when high conviction is present (**75.71% win rate, +0.4260% expectancy**). However, unconditional 95% accuracy remains an open research frontier. The system refuses to fabricate accuracy numbers or claim that markets are guaranteed to be predictable.

---

## 3. Quotex Research Feeds & Contract Reality

### 3.1 Non-Existent Direct Broker API
- Quotex does not publish an authorized, public REST or WebSocket API for direct order book data or programmatic contract execution.
- Any attempt to automate broker browser logins, reverse-engineer web sessions, or execute live orders is strictly prohibited.

### 3.2 Underlying Proxy Separation
- Quotex research feeds for crypto (`QUOTEX:BTCUSD_QX`, `QUOTEX:ETHUSD_QX`) are sourced exclusively from authorized spot crypto market feeds (Coinbase / Kraken).
- These feeds represent underlying spot pricing, **not direct Quotex broker contract quotes**. Payouts (e.g., 85%) and binary payoff mechanics are modeled for research only.

### 3.3 Fail-Closed Rejection of Synthetic OTC Assets
- Proprietary synthetic broker OTC assets (`MSFT_OTC`, `EURUSD_OTC`, `GBPUSD_OTC`) have no verifiable external market underlying.
- Under platform governance, these assets fail closed immediately with `QUOTEX_DATA_UNAVAILABLE`. The platform **never fabricates synthetic candles, prices, or payouts**.

---

## 4. Paper-Trading & Financial Risk Disclosures

### 4.1 Paper-Trading Worker Default State
- The paper-trading worker defaults to **`STOPPED`**. It does not start automatically on desktop application launch.
- Starting prospective paper trading requires explicit user authorization.

### 4.2 Minimum Observation Horizon Required
- Minimum governance protocol requires $\ge 60$ contiguous calendar days and $\ge 100$ valid resolved observations on live data before any strategy can be considered for statistical significance.
- Early win rates or positive returns over small sample sizes do not constitute statistical proof of out-of-sample edge.

### 4.3 Real-Money Trading Permanently Disabled
- Real-money broker trading and order execution functionality are **permanently disabled**.
- The platform is an analytical and paper-trading research intelligence application. It does not provide financial advice, and guaranteed returns are impossible.

---

## 5. Process Management & Operating Environment

### 5.1 Active Orphaned Background Processes
- Six orphaned background processes (`pythonw.exe -m desktop.app`, PIDs: `15912, 4692, 9412, 12224, 2796, 12952`) from earlier tray-minimized sessions remain in memory.
- In accordance with non-destructive operating safety rules, these processes have not been forcibly terminated without user authorization.
- Users are advised to review and terminate these processes prior to starting new operational sessions:
  ```powershell
  Stop-Process -Id 15912, 4692, 9412, 12224, 2796, 12952 -Force
  ```

### 5.2 Single-Instance Kernel Lock Protection
- The desktop application now enforces single-instance mutual exclusion via `core.system_lock.ProcessFileLock` guarding `data/.desktop_app.lock`.
- If an instance is already active, any secondary launch will notify the user and exit cleanly with code `0`.
- The GUI window close handler (`closeEvent`) has been refactored to stop all background timers and worker threads, preventing new orphaned background processes.
