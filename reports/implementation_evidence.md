# Implementation-to-Evidence Matrix

**Date:** 2026-10-20  
**Project Root:** `D:\ant`  
**Platform:** Windows 11 Enterprise (AMD64), Python 3.14.7 (`.venv`)  
**Scope:** NSE Equities, BSE Equities, Quotex Research Only  

---

## 1. Feature-to-Evidence Matrix

| Requirement / Subsystem | Implementation File(s) | Verification Test(s) | Execution Command | Observed Result | Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Kernel Process Lock Primitive** | [`core/system_lock.py`](file:///D:/ant/core/system_lock.py) | `tests/test_resolver_ownership_and_single_instance.py` | `.venv\Scripts\pytest.exe tests/test_resolver_ownership_and_single_instance.py` | 4/4 passing; context manager, PID tracking, and stale-lock recovery verified | **PASS** |
| **Desktop Single-Instance Guard** | [`desktop/app.py`](file:///D:/ant/desktop/app.py) | `tests/test_windows_desktop_app.py` | `.venv\Scripts\python.exe -m desktop.app --verify-and-exit` | Secondary instance exits cleanly without GUI duplication; exits code 0 | **PASS** |
| **Resolver Single-Owner Lock** | [`learning/ledger.py`](file:///D:/ant/learning/ledger.py#L403) | `tests/test_resolver_ownership_and_single_instance.py` | `.venv\Scripts\pytest.exe tests/test_resolver_ownership_and_single_instance.py` | Only primary acquires `.resolver.lock`; secondary operates in passive consumer mode | **PASS** |
| **Multi-Market Paper Trading Routing** | [`paper_trading/worker.py`](file:///D:/ant/paper_trading/worker.py) | `tests/test_multi_market_paper_worker.py` | `.venv\Scripts\pytest.exe tests/test_multi_market_paper_worker.py` | Routes `NSE:*` to Groww/TradingView, `BSE:*` to Groww/TradingView, `QUOTEX:*` to crypto proxy | **PASS** |
| **Synthetic Quotex OTC Rejection** | [`paper_trading/worker.py`](file:///D:/ant/paper_trading/worker.py#L740), [`data/ingestion/quotex.py`](file:///D:/ant/data/ingestion/quotex.py) | `tests/test_multi_market_paper_worker.py`, `tests/test_quotex_pipeline.py` | `.venv\Scripts\pytest.exe tests/test_quotex_pipeline.py` | `MSFT_OTC` returns `QUOTEX_DATA_UNAVAILABLE` fail-closed; zero synthetic prices fabricated | **PASS** |
| **NSE Live & Historical Data Feeds** | [`data/ingestion/nse.py`](file:///D:/ant/data/ingestion/nse.py) | `tests/test_multi_market.py` | `.venv\Scripts\python.exe -c "..."` (live fetch) | Live quote: ₹2460.90 (330.8ms latency); 10 historical 5m candles retrieved | **PASS** |
| **BSE Live & Historical Data Feeds** | [`data/ingestion/bse.py`](file:///D:/ant/data/ingestion/bse.py) | `tests/test_multi_market.py` | `.venv\Scripts\python.exe -c "..."` (live fetch) | Live quote: ₹1202.80 (266.2ms latency); 10 historical 5m candles retrieved | **PASS** |
| **NSE/BSE Cross-Exchange Separation** | [`core/symbol_resolver.py`](file:///D:/ant/core/symbol_resolver.py), [`data/ingestion/bse.py`](file:///D:/ant/data/ingestion/bse.py) | `tests/test_multi_market.py` | `.venv\Scripts\pytest.exe tests/test_multi_market.py` | Strict isolation; BSE scrip codes resolved; zero substitution of NSE quotes for BSE | **PASS** |
| **Dynamic Instrument Discovery** | [`core/universe.py`](file:///D:/ant/core/universe.py) | `tests/test_dynamic_universe_and_scanner.py` | `.venv\Scripts\pytest.exe tests/test_dynamic_universe_and_scanner.py` | 3,594 NSE instruments + 5,587 BSE instruments + 39 Quotex proxies registered | **PASS** |
| **Multiclass Contract Compliance** | [`models/ml_classifier.py`](file:///D:/ant/models/ml_classifier.py) | `tests/test_multiclass_contract.py` | `.venv\Scripts\pytest.exe tests/test_multiclass_contract.py` | 9/9 passing; [SELL=0, NEUTRAL=1, BUY=2] shape (N, 3), sum=1.0, Dirichlet smoothing | **PASS** |
| **Frozen Champion Model Integrity** | `models_store/lgbm_candidate_v20261020_044046` | `tests/test_forex_model_compatibility_and_contract.py` | `Get-FileHash models_store/.../model.joblib` | Exact SHA-256 match `d955be...`; `"is_production_ready": false` enforced | **PASS** |
| **Dynamic Ledger Integrity Decoupling** | [`paper_trading/health_report.py`](file:///D:/ant/paper_trading/health_report.py) | `tests/test_forex_model_compatibility_and_contract.py` | `.venv\Scripts\pytest.exe tests/test_forex_model_compatibility_and_contract.py` | Immutable files match exact hash; live ledger verified via backup prefix + schema | **PASS** |
| **Prospective Paper Storage Quarantine** | [`paper_trading/prediction_store.py`](file:///D:/ant/paper_trading/prediction_store.py) | `tests/test_paper_trading_prediction_store.py` | `.venv\Scripts\pytest.exe tests/test_paper_trading_prediction_store.py` | Paper predictions strictly isolated in `data/paper_trading/`; production ledger untouched | **PASS** |
| **Event-Driven Backtest & Metrics** | [`backtesting/engine.py`](file:///D:/ant/backtesting/engine.py), [`backtesting/metrics.py`](file:///D:/ant/backtesting/metrics.py) | `tests/test_backtesting.py` | `.venv\Scripts\pytest.exe tests/test_backtesting.py` | Walk-forward simulation with realistic Bid/Ask spread, slippage, and statutory friction | **PASS** |
| **Institutional Accuracy & Evaluation Engine** | [`learning/accuracy_evaluator.py`](file:///D:/ant/learning/accuracy_evaluator.py), [`scripts/evaluate_model_accuracy.py`](file:///D:/ant/scripts/evaluate_model_accuracy.py), [`scripts/train_multiclass_model.py`](file:///D:/ant/scripts/train_multiclass_model.py) | `tests/test_accuracy_evaluator.py`, `tests/test_train_multiclass_model.py` | `.venv\Scripts\pytest.exe tests/test_accuracy_evaluator.py tests/test_train_multiclass_model.py` | 13/13 passing; Wilson 95% CI, Brier score, ECE, decision margin sweep, 4-fold walk-forward cross-validation, and Section 8.4 states | **PASS** |
| **Desktop Dual-Subtabs & Audio Alerting** | [`desktop/main_window.py`](file:///D:/ant/desktop/main_window.py), [`desktop/service_controller.py`](file:///D:/ant/desktop/service_controller.py) | `tests/test_windows_desktop_app.py` | `.venv\Scripts\pytest.exe tests/test_windows_desktop_app.py` | Dual-subtabs, Section 8 Dual-Mode Telemetry display, & native Windows chime (`winsound.MessageBeep`) | **PASS** |
| **Scanner Offline Equity Cache Fallback** | [`signals/scanner.py`](file:///D:/ant/signals/scanner.py) | `tests/test_dynamic_universe_and_scanner.py` | `.venv\Scripts\pytest.exe tests/test_dynamic_universe_and_scanner.py` | Sub-second batch scanning falls back to authentic stored equity candles under network loss | **PASS** |
| **Gate B Local Latency (<50ms)** | [`features/incremental.py`](file:///D:/ant/features/incremental.py), [`signals/fusion.py`](file:///D:/ant/signals/fusion.py), [`scripts/benchmark_latency.py`](file:///D:/ant/scripts/benchmark_latency.py) | `tests/test_features.py`, `tests/test_live_prediction_pipeline.py` | `.venv\Scripts\python.exe scripts/benchmark_latency.py` | Operational live p95 = 45.33 ms (Dynamic Ticks), 28.50 ms (Universe Sweep); IncrementalFeatureCalculator parity verified | **PASS** |
| **Full Workspace Test Suite** | All 106 test modules | All 539 unit & integration tests | `.venv\Scripts\pytest.exe -q` | **539 passed, 0 failed, 2 warnings (100% pass rate)** | **PASS** |

---

## 2. External Integration & Blocker Classification

| External Integration | Access Method | Status | Technical Limitation |
| :--- | :--- | :--- | :--- |
| **NSE Public Feeds** | Groww / TradingView Scanner | **OPERATIONAL** | None (Market hours 09:15–15:30 IST) |
| **BSE Public Feeds** | Groww / TradingView Scanner | **OPERATIONAL** | None (Market hours 09:15–15:30 IST) |
| **Quotex Public REST API** | Non-Existent | **BLOCKED** | Quotex does not provide an authorized public API; operated via spot crypto proxy |
| **Forex Live Feeds** | OANDA Practice / ECB Cache | **FROZEN / OUT OF SCOPE** | Forex deliberately frozen per project charter |
| **Real-Money Broker Execution** | N/A | **PERMANENTLY DISABLED** | System operates in research and paper-trading mode only |
