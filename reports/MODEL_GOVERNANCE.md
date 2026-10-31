# Model Governance and Promotion Policy Audit

**Document Reference:** `POL-GOV-2026-001`  
**Governing Engine:** `models/registry.py` & `scripts/run_model_governance.py`  
**Active Production Champion:** `lgbm_candidate_v20261020_044046`  
**Champion SHA-256 Checksum:** `D955BE760504E9F690B6D3C21D9707FB7AD56A39C3BA5F03FE7AEBB833AC45C2`  
**Metadata SHA-256 Checksum:** `EBF940958DE5F617E359CDDAF802570CE0BF2E05A91F41A7350B0EC952F4307E`

---

## 1. Governance Principles & Pre-Registered Gates

To prevent statistical overfitting, survivorship bias, and silent degradation:

1. **Pre-Registered Promotion Hurdles:**
   - **Minimum Prospective Observation:** $\ge 60$ contiguous calendar days of paper-trading evaluation.
   - **Minimum Completed Observations:** $\ge 100$ resolved prospective predictions.
   - **Minimum Active Trades:** $\ge 35$ non-neutral trades.
   - **Net Cumulative Return:** $\text{Net Return} > 0.00\%$ after 6.0 bps Indian equity friction.
   - **Trade Expectancy:** Net expectancy $\ge +2.0\text{ bps}$ per active trade.
   - **Annualized Sharpe Ratio:** $\ge 1.00$.
   - **Maximum Allowable Drawdown:** Peak-to-trough drawdown $\le 4.00\%$.
   - **Net Win Rate:** $\ge 52.0\%$ across active trades.
   - **Benchmark Superiority:** Must simultaneously beat the flat no-trade hurdle and empirical market benchmark.

2. **Reversible Champion-Challenger Architecture:**
   - Candidate models are quarantined in isolated directory paths under `models_store/`.
   - The production pointer in `models_store/registry.json` is updated ONLY when a candidate clears all gates and operator authorization is logged.
   - If a newly promoted model suffers a drawdown exceeding $4.00\%$, `ContinuousLearningController.rollback_champion()` automatically restores the previous verified champion.

---

## 2. Current Model Store Inventory

| Staged Model Artifact | Creation Timestamp | Feature Space | Multiclass Brier | Promotion Status |
| :--- | :--- | :--- | :--- | :--- |
| `lgbm_candidate_v20261020_044046` | 2026-10-20T04:40:46 UTC | 18 Features | 0.6715 | **PRODUCTION CHAMPION (LOCKED)** |
| `lgbm_candidate_v20261020_051950` | 2026-10-20T05:19:50 UTC | 18 Features | 0.6710 | Staged Challenger (Pending Evaluation) |
| `lgbm_candidate_v20261020_140533` | 2026-10-20T14:05:33 UTC | 22 Features (Micro) | 0.6631 | Staged Challenger (Pending Evaluation) |
| `lgbm_multiclass_candidate` (EXP-08) | 2026-10-30T15:46:05 UTC | 27 Features (Cross-Asset) | **0.6613** | **STAGED CHALLENGER (EXP-08)** |

---

## 3. Governance Verdict & Action Summary

- **Production Champion Integrity:** 100% Verified. Hash matching byte-for-byte.
- **Candidate Promotion Status:** **BLOCKED / REJECTED** from automated promotion. While EXP-08 demonstrates statistically superior Brier score (0.6613 vs 0.6715), it must complete the mandatory prospective paper-trading evaluation protocol before any promotion can be authorized.
- **Paper Trading Daemon Status:** Preserved in `STOPPED` state awaiting explicit operator command.
