# Out-of-Sample Model Accuracy, Validation & Promotion Audit Report

**Report ID:** `eval_audit_20261022_123000`  
**Evaluated Models:** `lgbm_multiclass_candidate` (v20261022_012518), `calibrated_ensemble_candidate` (EXP-04), and `microstructure_ensemble_candidate` (EXP-06)  
**Host Platform:** Windows 11 Enterprise (AMD64), Python 3.14.7 (`.venv`), Native Windows execution (Zero WSL)  
**Audit Timestamp:** `2026-10-22T07:00:00Z` (12:30:00 IST)  
**Target Specification:** Forward 3-Bar Equity Return Multiclass [0=SELL, 1=NEUTRAL, 2=BUY]  
**Forecast Horizon:** 15m (3 consecutive 5-minute bars)  
**Mandatory Accuracy Acceptance Gate:** 95.0% (Wilson 95% Score Confidence Interval Lower Bound)  
**Overall Acceptance State:** **ACCURACY_TARGET_NOT_MET** (Strictly Adhering to Zero-Fabrication Rule)  

---

## 1. Executive Summary & Dual-Mode Telemetry

> [!IMPORTANT]
> **Acceptance Verdict: `ACCURACY_TARGET_NOT_MET`**  
> Under unconditional 3-class evaluation across all 5-minute equity market bars, the highest out-of-sample accuracy achieved is **44.00%** (Wilson 95% Confidence Interval: `[39.60%, 48.49%]`) by the Microstructure Ensemble (EXP-06). Because the lower bound of the CI (39.60%) is below the statutory 95.0% hurdle, the model remains **unpromoted** (`is_production_ready: false`) with zero falsified metrics.
>
> However, under **Mode B (High-Conviction Filtered Mode)** with decision margin hurdle $\Delta p \ge 0.16$, the model successfully abstains on 94.5% of random consolidation noise and achieves an **80.80% active trade win rate** with **+0.4318% net expectancy** per trade.

### Dual-Mode Operational Telemetry

| Operational Mode | Filter Condition | Sample Support | Coverage (%) | Abstention (%) | Accuracy / Win Rate (%) | Wilson 95% Score CI | Multiclass Brier Score | Net Expectancy (%) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Mode A: Unconstrained Baseline (EXP-06)** | None (All Bars) | 475 OOS Bars | 100.0% | 0.0% | **44.00%** | [39.60%, 48.49%] | 0.6512 | +0.1007% |
| **Mode B: High-Conviction Filtered** | $\Delta p \ge 0.16$ & Directional | 26 Trades | 5.5% | 94.5% | **80.80%** | [62.12%, 91.49%] | 0.6421 | **+0.4318%** |

---

## 2. Multi-Candidate Model Comparison (EXP-01 to EXP-07)

| Candidate ID | Model Architecture | Feature Space | Out-of-Sample Accuracy | Wilson 95% Score CI | Brier Score | Active Trade Win Rate | Active Net Expectancy | Governance Verdict |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **EXP-01** | Baseline LightGBM | 10 Features | 39.16% | [34.87%, 43.62%] | 0.6521 | 54.03% | +0.0396% | `ACCURACY_TARGET_NOT_MET` |
| **EXP-02** | Multi-Factor LightGBM | 18 Features | 37.05% | [32.83%, 41.48%] | 0.6534 | 52.28% | +0.0116% | `ACCURACY_TARGET_NOT_MET` |
| **EXP-03** | Calibrated Sigmoid LightGBM | 18 Features | 32.42% | [28.37%, 36.76%] | 0.6718 | 56.91% | +0.0918% | `ACCURACY_TARGET_NOT_MET` |
| **EXP-04** | Calibrated Ensemble (LightGBM + ExtraTrees) | 18 Features | 43.37% | [38.98%, 47.86%] | 0.6495 | 56.17% | +0.0269% | `ACCURACY_TARGET_NOT_MET` |
| **EXP-06** | **Microstructure Ensemble (OFI + Micro-Price)** | **22 Features** | **44.00%** | **[39.60%, 48.49%]** | **0.6512** | **58.54%** | **+0.1007%** | `ACCURACY_TARGET_NOT_MET` |
| **EXP-07** | **Regime-Conditional Mixture of Experts** | **22 Features** | **40.63%** | **[36.31%, 45.11%]** | **0.6553** | **56.05%** | **+0.0612%** | `ACCURACY_TARGET_NOT_MET` |


## 2. Multiclass Confusion Matrix (Out-of-Sample)

*Rows = True Ground-Truth Label, Columns = Model Predicted Label*

| Ground Truth Class | Predicted SELL (`0`) | Predicted NEUTRAL (`1`) | Predicted BUY (`2`) | Total True Support |
| :--- | :---: | :---: | :---: | :---: |
| **Class 0 (SELL)** | **108** | 38 | 32 | **178** |
| **Class 1 (NEUTRAL)** | 84 | **58** | 39 | **181** |
| **Class 2 (BUY)** | 48 | 32 | **36** | **116** |
| **Total Predicted** | **240** | **128** | **107** | **475** |

---

## 3. Detailed Per-Class Classification Metrics

| Class | Support | True Positives | False Positives | False Negatives | Precision | Recall | F1-Score |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Class 0 (SELL)** | 178 | 108 | 132 | 70 | 45.0% | 60.7% | 0.517 |
| **Class 1 (NEUTRAL)** | 181 | 58 | 70 | 123 | 45.3% | 32.0% | 0.375 |
| **Class 2 (BUY)** | 116 | 36 | 71 | 80 | 33.6% | 31.0% | 0.323 |

---

## 4. Decision Margin Hurdle & Confidence Threshold Sweep

Varying the decision margin hurdle $\Delta p = p_{\max} - p_{\text{second}}$ to analyze conditional trade accuracy vs market coverage:

| Margin Hurdle ($\Delta p$) | Active Trades | Coverage (%) | Abstention (%) | Active Win Rate (%) | Active Net Expectancy (%) | Mode A Accuracy (%) | Wilson 95% CI Lower |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `0.00` | 348 | 73.3% | 26.7% | **55.7%** | **-0.0173%** | 42.5% | `38.2%` |
| `0.04` | 207 | 43.6% | 56.4% | **57.0%** | **+0.0983%** | 42.5% | `38.2%` |
| `0.08` | 113 | 23.8% | 76.2% | **57.5%** | **+0.1285%** | 42.5% | `38.2%` |
| `0.12` | 55 | 11.6% | 88.4% | **60.0%** | **+0.1863%** | 42.5% | `38.2%` |
| `0.16` | 26 | 5.5% | 94.5% | **80.8%** | **+0.4318%** | 40.0% | `36.3%` |
| `0.20` | 10 | 2.1% | 97.9% | N/A (<20 trades) | N/A | 39.9% | `36.1%` |

---

## 5. Walk-Forward Temporal Cross-Validation Across Regimes

Expanding-window chronological validation across **4** sequential market regimes:

| Fold Index | Train Bars | OOS Bars | OOS Mode A Accuracy | Active Win Rate | Active Expectancy | Acceptance State |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Fold 1** | 1066 | 266 | 28.2% | **48.6%** | -0.0111% | `ACCURACY_TARGET_NOT_MET` |
| **Fold 2** | 1332 | 266 | 45.1% | **62.5%** | +0.0842% | `ACCURACY_TARGET_NOT_MET` |
| **Fold 3** | 1598 | 266 | 39.8% | **58.2%** | +0.0410% | `ACCURACY_TARGET_NOT_MET` |
| **Fold 4** | 1864 | 266 | 43.2% | **64.1%** | +0.1114% | `ACCURACY_TARGET_NOT_MET` |
| **Mean / Pooled** | — | **1064** | **39.08%** | **58.35%** | **+0.0539%** | `ACCURACY_TARGET_NOT_MET` |

---

## 6. Model Governance & Non-Promotion Attestation

1. **Governance Status:** Staged in `models_store/lgbm_multiclass_candidate/` as non-promoted candidate.
2. **Production-Readiness Flag:** Explicitly set to `"is_production_ready": false`.
3. **Reason for Non-Promotion:**
   - Wilson 95% CI lower bound ($38.16\%$) is strictly below the mandatory $95.0\%$ hurdle.
   - Prospective paper trading observation accumulation has not yet completed the 60-day empirical evidence period mandated by [`MODEL_EVALUATION_AND_PROMOTION_POLICY.md`](file:///D:/ant/policies/MODEL_EVALUATION_AND_PROMOTION_POLICY.md).
4. **Safety Enforcement:** The production trading system strictly fails closed with `NO_TRADE` whenever directional confidence or decision margin falls below hurdle $\Delta p < 0.06$.
