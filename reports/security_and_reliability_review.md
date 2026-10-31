# Application Security and Operational Reliability Review

**Date:** 2026-10-20  
**Project Root:** `D:\ant`  
**Review Scope:** Secrets handling, deserialization, network boundaries, test isolation, and process locking  
**Auditor:** Quantitative Systems Security & Reliability Engineering  

---

## 1. Secrets Management & Credential Hygiene

### 1.1 Findings
- **Zero Hardcoded Secrets in Codebase:** Automated AST/Regex scanning across all `.py` files confirmed no production API keys, passwords, or tokens are embedded in source code.
- **Environment Isolation:** Secrets are loaded via `core/config.py` using `pydantic-settings` from local `.env` files.
- **Log Redaction:** All HTTP clients, adapters, and background workers sanitize request payloads and omit authentication headers from logging outputs.
- **Configuration Templates:** A clean `.env.example` is maintained without real API keys.

---

## 2. Model Deserialization & Artifact Integrity

### 2.1 Threat Vector Analysis
Machine learning model serialization via Python `pickle` / `joblib` presents arbitrary code execution risks if untrusted or modified artifacts are deserialized.

### 2.2 Implemented Hardening
1. **Cryptographic Sealing:** All production models are pre-registered with their SHA-256 hashes in `reports/protected_assets_manifest.json` and [`policies/MODEL_EVALUATION_AND_PROMOTION_POLICY.md`](file:///D:/ant/policies/MODEL_EVALUATION_AND_PROMOTION_POLICY.md).
2. **Hash Verification Prior to Inference:** [`paper_trading/worker.py`](file:///D:/ant/paper_trading/worker.py) and [`paper_trading/health_report.py`](file:///D:/ant/paper_trading/health_report.py) compute the SHA-256 hash of `model.joblib` and verify it against `d955be760504e9f690b6d3c21d9707fb7ad56a39c3ba5f03fe7aebb833ac45c2` before loading weights into memory.
3. **Local Store Boundary Enforcement:** Artifact paths are strictly resolved within `models_store/` and sanitized to prevent directory traversal attacks (`../`).

---

## 3. Test Isolation & Production Mutation Protection

### 3.1 Hardened Guards
To prevent unit or integration tests from accidentally writing to or corrupting production ledgers or champion models:
- **Model Registry Test Trap ([`models/registry.py`](file:///D:/ant/models/registry.py#L46-L56)):** Detects `PYTEST_CURRENT_TEST` and throws `PermissionError` if any test attempts to write a model directly into `models_store/` without using isolated temporary fixture directories.
- **Ledger Quarantine:** All paper-trading and backtesting tests inject temporary directories (`temp_store_dir`), ensuring zero writes touch `data/ledger/prediction_outcomes.jsonl` or `data/ledger/pending_predictions.jsonl`.
- **Pre-Registered Immutable Baseline:** `data/ledger/prediction_outcomes.jsonl.bak` is write-protected as a static 407-record historical baseline.

---

## 4. Process Exclusivity & Single-Instance Concurrency

### 4.1 Locking Architecture ([`core/system_lock.py`](file:///D:/ant/core/system_lock.py))
- Uses non-blocking OS kernel file locking (`msvcrt.locking` on Windows, `fcntl.flock` on POSIX).
- Records executing PID and UTC timestamp.
- Detects stale locks from ungraceful terminations using OS process inspection (`psutil` / Win32 process handles) and safely recycles locks without manual intervention.

### 4.2 Guarded Subsystems
1. **Desktop Application UI:** Guarded by `data/.desktop_app.lock`. Prevents duplicate windows and background worker thread contention.
2. **Outcome Resolver Service:** Guarded by `data/ledger/.resolver.lock`. Secondary processes operate as read-only consumers without spawning competing resolver threads.
3. **Paper-Trading Worker:** Guarded by `data/paper_trading/.paper_worker.lock`. Ensures only one trading worker instance can ever poll feeds or evaluate signals.

---

## 5. Network Reliability & Bounded Failure Handling

1. **Explicit Request Timeouts:** All HTTP requests to external market feeds (Groww, TradingView, Coinbase, Kraken) specify explicit timeouts (4.0s to 6.0s). Zero infinite-hang sockets.
2. **Bounded Exponential Backoff:** Transient network disconnects trigger bounded retries with jitter up to a maximum threshold, followed by clean fail-closed logging.
3. **Session Awareness:** Continuous matching requests outside market hours (09:15–15:30 IST for Indian equities) fail closed cleanly without attempting redundant HTTP network polling.
