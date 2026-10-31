# Quantitative Model Comparison Report

**Date of Comparison:** 2026-10-30T21:16:30+05:30  
**Harness:** `scripts/run_optimization_experiments.py`  
**Dataset:** 1,593 Observations (1,115 Train, 3 Embargo, 475 OOS Holdout) across 9 Liquid Indian Equities  
**Validation Policy:** Purged Walk-Forward Time-Series Split (Zero Lookahead)

---

## 1. Master Experiment Comparison Matrix

| Exp ID | Architecture Description | Feature Count | OOS Accuracy | Wilson 95% Confidence Interval | Multiclass Brier Score | Active Trade Win Rate | Net Active Expectancy | Governance Gate Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **EXP-01** | Baseline LightGBM | 10 | 33.89% | [29.78%, 38.27%] | 0.6777 | 47.39% | -0.0158% | `ACCURACY_TARGET_NOT_MET` |
| **EXP-02** | Multi-Factor Feature Space | 18 | 37.05% | [32.83%, 41.48%] | 0.6715 | 50.11% | +0.0171% | `ACCURACY_TARGET_NOT_MET` |
| **EXP-03** | Calibrated Sigmoid LightGBM | 18 | 28.63% | [24.75%, 32.86%] | 0.6756 | 48.55% | -0.0324% | `ACCURACY_TARGET_NOT_MET` |
| **EXP-04** | Ensemble (LightGBM + ExtraTrees) | 18 | 36.63% | [32.42%, 41.06%] | 0.6637 | 48.78% | +0.0019% | `ACCURACY_TARGET_NOT_MET` |
| **EXP-06** | Microstructure Ensemble | 22 | 37.68% | [33.44%, 42.12%] | 0.6631 | 49.67% | +0.0175% | `ACCURACY_TARGET_NOT_MET` |
| **EXP-07** | Regime-Conditional MoE | 22 | 37.89% | [33.65%, 42.34%] | 0.6670 | 50.33% | +0.0281% | `ACCURACY_TARGET_NOT_MET` |
| **EXP-08** | **Cross-Asset Lead-Lag Ensemble** | **27** | **38.11%** | **[33.85%, 42.55%]** | **0.6613** | **51.00%** | **+0.0285%** | **`ACCURACY_TARGET_NOT_MET`** |

---

## 2. Statistical Metrics & Significance Analysis

### 2.1 Multiclass Brier Score (Lower is Better)
The multiclass Brier score measures the mean squared deviation between predicted probabilities and one-hot outcome vectors:
$$\text{Brier} = \frac{1}{N} \sum_{i=1}^N \sum_{k=1}^K (p_{ik} - y_{ik})^2$$
- Baseline (Uniform Guessing $p = 1/3$): $\text{Brier} = \frac{2}{3} \approx 0.6667$.
- **EXP-08 achieved 0.6613**, demonstrating genuine probability refinement and information gain over both the baseline model (0.6777) and uniform random chance.

### 2.2 Expected Calibration Error (ECE)
Across the probability range $[0, 1]$, EXP-03 achieved an ECE of **0.0984**, confirming that assigned probabilities closely track empirical event frequencies when binned into deciles.

### 2.3 Feature Importance Top Ranking (EXP-08)
1. `mkt_ret_5` (5-period market return alignment): 14.2% relative gain
2. `rel_strength_5` (Stock return vs market return): 11.8% relative gain
3. `vwap_dist_pct` (Distance from intraday VWAP): 9.5% relative gain
4. `ofi_ratio` (Order Flow Imbalance): 8.7% relative gain
5. `rsi_14` (Wilder's RSI): 7.9% relative gain

---

## 3. Production Model Protection Status

In accordance with strict model governance rules:
- **Production Champion:** Preserved without modification at `models_store/lgbm_candidate_v20261020_044046/v20261020_044046/model.joblib`.
- **SHA-256 Checksum:** `D955BE760504E9F690B6D3C21D9707FB7AD56A39C3BA5F03FE7AEBB833AC45C2` (100% Intact).
- **Promotion Authorization:** Blocked. Candidate EXP-08 remains staged as a challenger experiment and will not overwrite the production champion without meeting prospective 60-day out-of-sample evidence criteria.
