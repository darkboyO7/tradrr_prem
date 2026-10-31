# Out-of-Sample Model Accuracy & Evaluation Audit Report

**Report ID:** `eval_audit_20261030_155513`  
**Evaluated Model ID:** `lgbm_multiclass_candidate`  
**Model Version:** `v20261022_012518`  
**Audit Timestamp:** `2026-10-30T15:55:13.409715+00:00`  
**Target Specification:** `Forward 3-Bar Equity Return Multiclass [0=SELL, 1=NEUTRAL, 2=BUY]`  
**Forecast Horizon:** `15m`  
**Mandatory Accuracy Acceptance Gate:** `95.0%` (Wilson 95% Confidence Interval Lower Bound)  
**Overall Acceptance State:** **ACCURACY_TARGET_NOT_MET**  

---

## 1. Executive Summary & Dual-Mode Telemetry

> [!IMPORTANT]
> **Acceptance Verdict: `ACCURACY_TARGET_NOT_MET`**  
>
> **Dual-Mode Operational Telemetry:**
> - **Mode A (Unconstrained Baseline — All Market Bars):**
>   - 3-Class Accuracy: **42.23%**
>   - Wilson 95% Score Confidence Interval: **[38.45%, 46.10%]**
>   - Coverage: **77.5%** | Abstention: **22.4%**
>   - Multiclass Brier Score: **0.6598** (Random Guess Baseline: `0.6667`)
>   - Expected Calibration Error (ECE): **0.0292**
> - **Mode B (High-Conviction Filtered — Hurdle $\Delta p \ge 0.16$):**
>   - Active Trade Win Rate: **60.00%**
>   - Active Net Expectancy: **+0.2140%**
>   - Selective Coverage: **12.6%** (Abstention: **87.4%**)
>   - Conditional Accuracy: **31.9%**  

---

## 2. 3-Class Multiclass Confusion Matrix

*Rows = True Ground-Truth Label, Columns = Model Predicted Label*

| Ground Truth Class | Predicted SELL (`0`) | Predicted NEUTRAL (`1`) | Predicted BUY (`2`) | Total Support |
| :--- | :---: | :---: | :---: | :---: |
| **Class 0 (SELL)** | **165** | 53 | 40 | **258** |
| **Class 1 (NEUTRAL)** | 97 | **61** | 30 | **188** |
| **Class 2 (BUY)** | 119 | 29 | **43** | **191** |
| **Total Predicted** | **381** | **143** | **113** | **637** |

---

## 3. Detailed Per-Class Classification Metrics

| Class | Support | True Positives | False Positives | False Negatives | Precision | Recall | F1-Score |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Class 0 (SELL)** | 258 | 165 | 216 | 93 | 43.3% | 63.9% | 0.516 |
| **Class 1 (NEUTRAL)** | 188 | 61 | 82 | 127 | 42.7% | 32.5% | 0.369 |
| **Class 2 (BUY)** | 191 | 43 | 70 | 148 | 38.0% | 22.5% | 0.283 |

---

## 4. Decision Margin Hurdle & Confidence Threshold Sweep

Varying the decision margin hurdle $\Delta p = p_{\max} - p_{\text{second}}$ to analyze conditional trade accuracy vs market coverage:

| Margin Hurdle ($\Delta p$) | Coverage (%) | Abstention (%) | Active Samples | Active Win Rate (%) | Active Net Expectancy (%) | Overall Accuracy (%) | Wilson 95% CI Lower |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `0.00` | 77.5% | 22.4% | 494 | **53.0%** | **+0.0555%** | 42.2% | `38.5%` |
| `0.05` | 48.2% | 51.8% | 307 | **51.1%** | **+0.0535%** | 36.9% | `33.2%` |
| `0.08` | 31.6% | 68.5% | 201 | **56.2%** | **+0.1036%** | 36.1% | `32.5%` |
| `0.12` | 20.9% | 79.1% | 133 | **56.4%** | **+0.1501%** | 34.1% | `30.5%` |
| `0.16` | 12.6% | 87.4% | 80 | **60.0%** | **+0.2140%** | 31.9% | `28.4%` |
| `0.20` | 8.0% | 92.0% | 51 | **60.8%** | **+0.2549%** | 31.2% | `27.8%` |
| `0.25` | 3.9% | 96.1% | 25 | **68.0%** | **+0.1863%** | 30.0% | `26.6%` |
| `0.30` | 1.9% | 98.1% | 12 | **50.0%** | **+0.0117%** | 29.0% | `25.6%` |

---

## 5. Walk-Forward Temporal Cross-Validation Across Regimes

Expanding-window chronological validation across **4** sequential market regimes (1064 total OOS observations):

| Fold Index | Train Bars | OOS Bars | OOS Accuracy | Active Win Rate | Active Expectancy | Acceptance State |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Fold 1** | 1066 | 266 | 28.2% | **48.6%** | **-0.0111%** | `ACCURACY_TARGET_NOT_MET` |
| **Fold 2** | 1332 | 266 | 42.1% | **45.2%** | **+0.0892%** | `ACCURACY_TARGET_NOT_MET` |
| **Fold 3** | 1598 | 266 | 43.2% | **58.0%** | **+0.0670%** | `ACCURACY_TARGET_NOT_MET` |
| **Fold 4** | 1864 | 266 | 40.2% | **55.0%** | **+0.0705%** | `ACCURACY_TARGET_NOT_MET` |

- **Mean Fold Accuracy:** **38.44%** (+/- 6.01%)
- **Mean Active Trade Win Rate:** **51.72%**
- **Mean Active Net Expectancy:** **+0.0539%** per trade
- **Aggregate Out-of-Sample Accuracy:** **38.44%** (Wilson 95% CI: `[35.56%, 41.40%]`)

---

## 6. Data Leakage & Temporal Integrity Verification

| Check Parameter | Audit Status | Audit Details |
| :--- | :---: | :--- |
| **Purged Embargo Enforcement** | **PASS** | Purged embargo gap of 3 bars strictly prevents overlapping forecast labels. |
| **Timestamp Non-Overlap** | **PASS** | Zero timestamp intersection between train and untouched OOS holdout partitions. |
| **Chronological Monotonicity** | **PASS** | Max train timestamp < Min OOS timestamp; zero future information leakage. |
| **Causality of Feature Computations** | **PASS** | FeaturePipeline strictly computes backwards-looking rolling indicators without lookahead. |

---

## 7. Audit Verdict & Governance Determination

1. **Empirical Accuracy Status:**
   - Raw unconditional 3-class accuracy on untouched out-of-sample data is **42.23%** (Wilson 95% CI: `[38.45%, 46.10%]`).
   - Active trade win rate on directional signals is **53.04%** with net expectancy of **+0.0555%**.
   - Under Section 8.4, because the lower bound of the 95% Wilson confidence interval (`38.45%`) is below the required 95.0% hurdle, the official acceptance state is strictly:
     `ACCURACY_TARGET_NOT_MET`.
2. **Production Staging & Protection:**
   - Candidate model remains staged in `models_store/` with `"is_production_ready": false`.
   - Production champion promotion remains strictly governed and blocked until 60 contiguous calendar days of prospective live paper evidence are accumulated.
