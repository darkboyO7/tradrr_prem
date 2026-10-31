# Antigravity Trading Intelligence Platform (TIP)

An institutional quantitative trading intelligence application built for multi-market state analysis, controlled self-learning, walk-forward backtesting, prospective paper-trading, and rigorous regulatory governance.

**Platform:** Native Windows 10/11 Local Application (No WSL or Web Server required)  
**Supported Scope:** NSE Equities, BSE Equities, Quotex Research Feeds (Spot Crypto Proxy)  
**Execution Mode:** Research & Controlled Paper Trading (Real-Money Execution is **Permanently Disabled**)  

---

## 1. Quick Start (Windows Native)

### 1.1 Environment Setup
```powershell
# Using PowerShell
.\setup_environment.ps1

# Or using Windows Command Prompt / Batch
setup_environment.bat
```

### 1.2 Launch Desktop Application
```powershell
# Launch GUI Desktop App
.\launch_app.ps1
# or
launch_app.bat

# Headless Verification Mode (Startup validation -> Exit code 0)
.venv\Scripts\python.exe -m desktop.app --verify-and-exit
```

### 1.3 Execute Full Test Suite (521 Tests)
```powershell
.\run_tests.ps1
# or
run_tests.bat
```

### 1.4 Generate Daily Health Report & Diagnostics
```powershell
# Daily Prospective Health Report
.\run_health_check.bat

# Full Operational Status & Diagnostics
.\run_diagnostics.bat
```

---

## 2. Core Market Scopes & Data Integration

| Market Segment | Data Feeds | Session Hours | Capabilities | Status |
| :--- | :--- | :--- | :--- | :--- |
| **NSE Equities** | Groww REST API + TradingView India Scanner | 09:15 – 15:30 IST (Mon-Fri) | Live quotes, 5m/15m/1h/1d candles, order routing | **OPERATIONAL** |
| **BSE Equities** | Groww Scrip REST API + TradingView BSE Scanner | 09:15 – 15:30 IST (Mon-Fri) | Scrip code resolution (`5xxxxx`), live quotes, candles | **OPERATIONAL** |
| **Quotex Research** | Coinbase / Kraken Spot REST (Crypto Proxy) | 24/7 (Crypto) | Underlying market proxies (`BTCUSD_QX`, `ETHUSD_QX`) | **RESEARCH ONLY** |
| **Quotex OTC** | N/A (Proprietary Broker Synthetic) | N/A | Fail-closed (`QUOTEX_DATA_UNAVAILABLE`); zero fabrication | **BLOCKED** |
| **Forex** | Frozen Reference Cache | N/A | Preserved for test compatibility only | **OUT OF SCOPE** |

---

## 3. System Architecture & Process Concurrency

```
               ┌────────────────────────────────────────────────────────┐
               │    Windows Desktop Application (PySide6 / Qt6)         │
               │   Guarded by Kernel Lock: data/.desktop_app.lock       │
               └───────────┬────────────────────────────────┬───────────┘
                           │                                │
                           ▼                                ▼
       ┌─────────────────────────────────────┐  ┌─────────────────────────────────────┐
       │   Background Outcome Resolver       │  │   Prospective Paper Worker          │
       │   data/ledger/.resolver.lock        │  │   data/paper_trading/.paper_worker.lock │
       │   Resolves matured predictions      │  │   Defaults to STOPPED                   │
       └───────────────────┬─────────────────┘  └──────────────────┬──────────────────┘
                           │                                       │
                           ▼                                       ▼
       ┌─────────────────────────────────────┐  ┌─────────────────────────────────────┐
       │   Operational Production Ledger     │  │   Quarantined Paper Storage         │
       │   data/ledger/prediction_outcomes   │  │   data/paper_trading/paper_pred     │
       │   (764 records, lines 1-407 = .bak) │  │   data/paper_trading/paper_outcomes │
       └─────────────────────────────────────┘  └─────────────────────────────────────┘
```

### 3.1 Single-Instance Process Enforcement
- **Primitive:** [`core/system_lock.py`](file:///D:/ant/core/system_lock.py) (`ProcessFileLock`) utilizing non-blocking kernel locking (`msvcrt.locking` on Windows, `fcntl.flock` on POSIX).
- **Desktop Application:** Guarded by `data/.desktop_app.lock`. Duplicate launches exit cleanly without spawning duplicate GUI windows or threads.
- **Outcome Resolver:** Guarded by `data/ledger/.resolver.lock`. Secondary processes operate as passive consumers without starting competing resolver threads.
- **Paper-Trading Worker:** Guarded by `data/paper_trading/.paper_worker.lock`. Ensures strictly one active paper worker.

---

## 4. Model Governance & Pre-Registered Policy

All model evaluation and promotion are governed by [`policies/MODEL_EVALUATION_AND_PROMOTION_POLICY.md`](file:///D:/ant/policies/MODEL_EVALUATION_AND_PROMOTION_POLICY.md) (Policy ID `POL-GOV-2026-001`).

### Immutable Benchmark Fingerprints
- **Model Candidate ID:** `lgbm_candidate_v20261020_044046`
- **Model Weights SHA-256:** `d955be760504e9f690b6d3c21d9707fb7ad56a39c3ba5f03fe7aebb833ac45c2`
- **Model Metadata SHA-256:** `ebf940958de5f617e359cddaf802570ce0bf2e05a91f41a7350b0ec952f4307e`
- **Production Readiness:** `"is_production_ready": false` (**Promotion strictly BLOCKED**)

### Key Governance Requirements
1. **Prospective Observation:** $\ge 60$ contiguous calendar days and $\ge 100$ valid resolved observations on live forward data required before promotion review.
2. **Economic Viability:** Net cumulative return $> 0.00\%$ after realistic friction (6.0 bps for Indian equities, 8.0 bps for Quotex proxies).
3. **Risk Boundaries:** Maximum drawdown $\le 4.00\%$; net win rate on active trades $\ge 52.0\%$.
4. **Explicit Human Sign-Off:** Automated self-learning is authorized to train candidates in staging, but **strictly prohibited from promoting to champion without human authorization**.

---

## 5. Audit Reports and Forensic Records

Comprehensive technical audits and baselines are maintained in the `reports/` directory:
- [`reports/system_baseline.md`](file:///D:/ant/reports/system_baseline.md): Full baseline audit, dependency list, and forensic investigation.
- [`reports/protected_assets_manifest.json`](file:///D:/ant/reports/protected_assets_manifest.json): Machine-readable cryptographic inventory of protected files and hashes.
- [`reports/master_build_progress.md`](file:///D:/ant/reports/master_build_progress.md): Phase-by-phase execution tracking.
- [`reports/implementation_evidence.md`](file:///D:/ant/reports/implementation_evidence.md): Requirement-to-test evidence matrix.
- [`reports/data_provider_status.md`](file:///D:/ant/reports/data_provider_status.md): Provider latency, session rules, and verification.
- [`reports/model_governance_report.md`](file:///D:/ant/reports/model_governance_report.md): Candidate lineage and evaluation policy.
- [`reports/backtest_and_evaluation_report.md`](file:///D:/ant/reports/backtest_and_evaluation_report.md): 764-record empirical evaluation and backtesting analytics.
- [`reports/paper_trading_readiness.md`](file:///D:/ant/reports/paper_trading_readiness.md): Paper worker checklist and operational readiness.
- [`reports/security_and_reliability_review.md`](file:///D:/ant/reports/security_and_reliability_review.md): Security review, secrets hygiene, and test isolation.
- [`reports/known_limitations.md`](file:///D:/ant/reports/known_limitations.md): Boundary disclosures and operational limitations.
- [`reports/final_acceptance_report.md`](file:///D:/ant/reports/final_acceptance_report.md): Final independent acceptance audit report.

---

## 6. Safety and Compliance Declarations

- **Real-Money Trading:** **PERMANENTLY DISABLED**. All orders and executions are strictly simulated in paper-trading mode.
- **Data Honesty Guarantee:** The platform never fabricates candles, prices, or payouts. When external data is unavailable, the system reports `DATA_UNAVAILABLE` fail-closed.
- **No Guaranteed Returns:** Past performance, historical backtests, and paper trading results do not guarantee future profitability.
