# Pre-Registered Model Evaluation and Promotion Policy

**Document ID:** `POL-GOV-2026-001`  
**Status:** `PRE-REGISTERED & FROZEN`  
**Effective Date & Start Timestamp:** `2026-10-20T11:45:00+00:00`  
**Governed Platform:** Trading Intelligence Platform (`D:\ant`)  
**Scope:** Prospective Paper-Trading, Continuous Learning Promotion, Live Order Gating  

### Immutable Evaluation Fingerprints
- **Candidate Model ID**: `lgbm_candidate_v20261020_044046`
- **Model Weights SHA-256**: `d955be760504e9f690b6d3c21d9707fb7ad56a39c3ba5f03fe7aebb833ac45c2`
- **Model Metadata SHA-256**: `ebf940958de5f617e359cddaf802570ce0bf2e05a91f41a7350b0ec952f4307e`
- **Production Ledger Baseline SHA-256**: `5875b2386278a702a4012ef323ad7db4dc113fa443a522877fba29fd8215b6c1`
- **Calibration Temperature**: \(T = 1.9615\) (Fitted on validation partition)
- **Decision Margin Hurdle**: \(0.0800\) (\(\max(p) - \text{second\_max}(p) \ge 0.0800\))

---

## 1. Purpose and Governance Principles

This policy establishes strict, tamper-proof criteria for evaluating whether candidate machine-learning models produce genuinely useful, statistically significant, and economically viable trading signals on prospective, out-of-sample market observations.

### Core Principles
1. **Zero Retrospective Threshold Tuning**: All performance thresholds, statistical gates, friction assumptions, and observation periods are pre-registered in this document before prospective evaluation begins. Modifying criteria post-observation is strictly prohibited.
2. **True Out-of-Sample Isolation**: Historical backtests and walk-forward evaluations serve exclusively for exploratory hypothesis generation. Promotion to production requires verifiable prospective paper-trading performance on live market data collected forward in time.
3. **Economic Viability Over Abstract Accuracy**: Raw directional accuracy is insufficient. A candidate must demonstrate positive net returns after institutional transaction friction, tight drawdown bounds, and superior calibration compared to naive and empirical baselines.
4. **Data Integrity Guarantee**: Predictions must be persisted before outcome prices are observable. Missing market data must defer or expire predictions without fabricating prices.
5. **No Double-Counting of Transaction Costs**: When executable fills use bid/ask quotes with slippage, the spread and slippage are already embedded in fill prices. External commission and financing must be added, but spread cannot be deducted twice.

---

## 2. Prospective Observation Requirements

Before any candidate model can be considered for shadow-trading or live trading promotion, it must complete the following minimum prospective observation protocol:

| Requirement Parameter | Minimum Threshold | Rationale |
| :--- | :--- | :--- |
| **Prospective Calendar Duration** | \(\ge 60\) contiguous calendar days | Ensures exposure to multiple macroeconomic regimes, central-bank announcements, and volatility shifts. |
| **Minimum Evaluated Predictions** | \(\ge 100\) completed prospective observations | Provides sufficient degrees of freedom for statistical significance testing. |
| **Minimum Active Trades Taken** | \(\ge 35\) non-neutral trades | Prevents models from passing purely through extreme inertia or 100% no-trade dormancy. |
| **Instrument Representation** | \(\ge 3\) independent currency pairs / assets | Ensures cross-sectional generalizability across disparate liquidity and volatility profiles. |
| **Data Completeness** | \(\ge 98.0\%\) valid resolved observations | Less than 2.0% allowable data deferrals or expirations. Zero fabricated quotes. |

### Pre-Declared Treatment of Special Cases
- **Missing Market Data**: Defer resolution for up to 3 calendar days (72 hours). If valid broker quotes cannot be acquired within 72 hours, resolve as `EXPIRED_WITHOUT_DATA` with `net_return = 0.0`.
- **Exclusion from Win-Rate Denominator**: Expired and invalid records must be tracked separately and **excluded from the active trade win-rate denominator**.
- **Overlapping Predictions**: If an instrument already has an active pending prediction for the same completed candle, subsequent duplicate signals are rejected.
- **Unresolved at Milestone Close**: Pending predictions that have not reached their forecast horizon are listed as pending and excluded from resolved performance statistics.

---

## 3. Net-Performance and Risk Hurdle Requirements

Candidate performance must be computed net of all transaction costs using realistic market friction:
- **Forex**: 1.5 bps (0.015%) per active trade (accounting for 0.75 bps spread + 0.5 bps commission + 0.25 bps slippage).
- **Indian Equities**: 6.0 bps (0.060%) per active trade (statutory STT, stamp duty, exchange charges, SEBI fees, brokerage, slippage).

### Required Performance Gates
1. **Net Cumulative Return**:
   $$\text{Net Return} = \prod_{t=1}^{N} (1 + r_{\text{net}, t}) - 1 > 0.00\%$$
   The candidate model must generate positive net cumulative return after realistic friction.
2. **Sharpe / Expectancy Gate**:
   - Trade-level net expectancy \(\ge +0.02\%\) (+2.0 bps per active trade).
   - Annualized Net Sharpe Ratio \(\ge 1.00\) (assuming 252 trading days).
3. **Maximum Allowable Drawdown (Peak-to-Trough)**:
   $$\text{Max Drawdown} \le 4.00\%$$
   Any breach exceeding 4.00% drawdown immediately invalidates the prospective candidate and halts evaluation.
4. **Profitable Trade Ratio (Net Win Rate)**:
   $$\text{Net Win Rate} = \frac{N_{\text{trades with } r_{\text{net}} > 0}}{N_{\text{total active trades}}} \ge 52.0\%$$

---

## 4. Benchmark Hurdle Comparisons

The candidate model must simultaneously outperform all pre-registered baselines evaluated on the **exact same timestamps and execution assumptions**:

| Benchmark | Comparison Metric | Promotion Requirement | Failure Implication |
| :--- | :--- | :--- | :--- |
| **No-Trade Hurdle** | Net Cumulative Return | Candidate Net Return \(> 0.00\%\) | Candidate destroys capital after fees; rejected. |
| **No-Trade Hurdle** | Maximum Drawdown | Candidate Max DD \(\le 4.00\%\) | Strategy risk exceeds capital budget; rejected. |
| **Class-Prior Probability Baseline** | Multiclass Brier Score | \(\text{Brier}_{\text{candidate}} < \text{Brier}_{\text{prior}}\) | Candidate probability calibration is worse than empirical base rate; rejected. |
| **Class-Prior Probability Baseline** | Multiclass Log Loss | \(\text{LogLoss}_{\text{candidate}} < \text{LogLoss}_{\text{prior}}\) | Candidate introduces probability distortion; rejected. |
| **Persistence / Naive Momentum** | Balanced Accuracy | \(\text{BalAcc}_{\text{candidate}} > \text{BalAcc}_{\text{persistence}}\) | Model fails to beat trivial 1-bar price lag; rejected. |
| **Persistence / Naive Momentum** | Net Cumulative Return | \(\text{NetRet}_{\text{candidate}} > \text{NetRet}_{\text{persistence}}\) | Persistence generates higher net alpha; rejected. |

---

## 5. Calibration and Probability Quality Gates

Probabilities emitted by the model directly govern capital sizing and risk gating. The following calibration criteria must be satisfied:

1. **Multiclass Probability Axiom**:
   $$\sum_{k=0}^{2} p_{ik} = 1.0 \pm 10^{-5}, \quad p_{ik} \ge 0.0, \quad \forall i, k$$
2. **Expected Calibration Error (ECE)**:
   - Evaluated across 10 uniform bins spanning confidence range \([0.333, 1.000]\).
   - Requirement: \(\text{ECE} \le 8.00\%\).
3. **Confidence-Accuracy Monotonicity**:
   - Higher confidence bins must exhibit non-decreasing empirical accuracy across all populated bins containing \(\ge 10\) observations.

---

## 6. Uncertainty Estimation and Statistical Power Clarification

Because currency pairs sharing a base currency (e.g. EURAUD, EURCAD, EURCHF) exhibit contemporaneous correlation, and financial time series exhibit volatility clustering, observations cannot be treated as independent identically distributed (i.i.d.) trials.

### Mandatory Statistical Protocol
1. **Contemporaneous Cross-Sectional Dependence**:
   - When aggregating across currency pairs, cross-pair covariance must be explicitly factored into variance estimation.
2. **Block-Bootstrapped Confidence Intervals**:
   - Primary metric confidence intervals must be computed using **Stationary Block Bootstrap** with an expected block length of \(L = 5\) bars to preserve temporal autocorrelation.
   - **Hurdle Gate vs. Profitability Distinction**:
     * *Hurdle Requirement*: The 95% lower confidence bound of net cumulative return must be strictly greater than \(-0.50\%\):
       $$\text{CI}_{0.025}(\text{Net Cumulative Return}) > -0.50\%$$
     * *Statistical Profitability Requirement*: A confidence interval whose lower bound is negative (e.g. \([-0.35\%, +1.40\%]\)) **does NOT prove positive expected returns**. A formal claim of profitability requires the lower bound to be strictly positive (\(\text{CI}_{0.025} > 0.00\%\)).
3. **Regime Stratification**:
   - Performance must be separately reported across:
     * Trending Regimes (\(\text{ADX}_{14} \ge 25.0\))
     * Mean-Reverting / Choppy Regimes (\(\text{ADX}_{14} < 25.0\))
   - A candidate must not suffer severe catastrophic drawdown (> 3.5%) in either regime.

---

## 7. Multi-Stage Gating Architecture

Evaluation criteria are strictly separated into three independent tiers:

```
[ Tier A: Probability Model Quality ]
  * Brier Score < Class-Prior Brier
  * Log Loss < Class-Prior Log Loss
  * ECE <= 8.00%
  * Row sum normalization = 1.0

           │
           ▼ (Must pass Tier A)
[ Tier B: Paper-Trading Operational Readiness ]
  * Crash-safe append-only store active
  * Process lock active (zero duplicate workers)
  * Live broker data feed verified fresh (< 60s)
  * Executable bid/ask fills without double-counting

           │
           ▼ (Must pass Tier B and accumulate 60d / 100 obs)
[ Tier C: Future Production Promotion Decision ]
  * Net return > 0.00% after realistic friction
  * Max drawdown <= 4.00%
  * Net Win Rate >= 52.0%
  * Outperformed No-Trade, Class-Prior, and Persistence
  * Explicit Dual Human + Governance Approval
```

---

## 8. Failure Conditions and Rollback Triggers

If any of the following conditions occur during the prospective evaluation period, evaluation is immediately terminated and the candidate is rejected:
1. **Drawdown Breach**: Cumulative drawdown exceeds 4.00% at any point.
2. **Cost Destruction**: Cumulative net return drops below -1.50% at realistic friction.
3. **Execution Lookahead Invalidation**: Any audit detects that a prediction relied on unfinalized bar data or prices that were non-executable at prediction time.
4. **Data Fabrication**: Any observation price or outcome is fabricated or interpolated rather than derived from authentic broker feed data.
5. **Singleton Leakage**: Any automated test suite or background runner pollutes the production prediction store.

---

## 9. Promotion Review Process

1. **Independent Verification**: Upon completing 60 days and 100 observations, an automated verification script generates `paper_trading_audit_report.json` and `paper_trading_ledger.csv`.
2. **Dual-Signoff Gate**: Both system governance and human administrator must verify SHA-256 hashes of the model, code commit, and ledger files.
3. **Shadow Stage**: Approved models transition to a 14-day shadow-execution phase before real capital allocation.
4. **Zero Automatic Activation**: Live order execution clients are permanently disabled and cannot be activated without explicit manual intervention.
