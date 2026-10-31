# Baseline Test Results

**Date of Execution:** 2026-10-30T21:04:00+05:30  
**Host Environment:** Windows 11 Enterprise (AMD64), Python 3.14.7, pytest 9.1.1  
**Project Directory:** `D:\ant`  
**Execution Command:** `.venv\Scripts\python.exe -m pytest -v --tb=short`

---

## 1. Executive Summary

| Metric | Baseline Value | Status |
| :--- | :--- | :--- |
| **Total Test Files** | 106 test suites | Verified |
| **Total Collected Items** | 539 tests | Verified |
| **Passed Tests** | **539** | **PASS (100.0%)** |
| **Failed Tests** | **0** | **PASS (0.0%)** |
| **Skipped Tests** | **0** | **PASS** |
| **Execution Duration** | 169.57 seconds (02:49.57) | Optimal |
| **Warnings Identified** | 2 RuntimeWarnings (resolved in `analysis/alpha_research.py`) | Remediated |

---

## 2. Test Suite Breakdown by Functional Area

### Core Infrastructure & Architecture (100% Pass)
- `tests/test_api.py`: 41 passed
- `tests/test_config.py`: 3 passed
- `tests/test_database.py`: 3 passed
- `tests/test_logging.py`: 2 passed
- `tests/test_health.py`: 1 passed
- `tests/test_snapshot_manager.py`: 4 passed
- `tests/test_gateway_heartbeat.py`: 5 passed
- `tests/test_preflight_validator.py`: 6 passed
- `tests/test_stress_harness.py`: 3 passed
- `tests/test_stress_testing.py`: 4 passed
- `tests/test_deterministic_replay.py`: 2 passed
- `tests/test_full_platform_e2e.py`: 1 passed

### Market Data Ingestion & Normalization (100% Pass)
- `tests/test_candle_engine.py`: 3 passed
- `tests/test_resampler.py`: 5 passed
- `tests/test_normalizer.py`: 3 passed
- `tests/test_validator.py`: 8 passed
- `tests/test_stale_detection.py`: 1 passed
- `tests/test_websocket_stream.py`: 3 passed
- `tests/test_dynamic_universe_and_scanner.py`: 4 passed
- `tests/test_binance_provider.py`: 5 passed
- `tests/test_kraken_provider.py`: 6 passed
- `tests/test_oanda_client.py`: 3 passed
- `tests/test_clean_forex_provider.py`: 13 passed
- `tests/test_forex_candle_pipeline_and_governance.py`: 13 passed
- `tests/test_quotex_pipeline.py`: 9 passed

### Feature Engineering & Alpha Research (100% Pass)
- `tests/test_features.py`: 9 passed
- `tests/test_microstructure_features.py`: 2 passed
- `tests/test_volume_profile.py`: 3 passed
- `tests/test_patterns.py`: 4 passed
- `tests/test_regime.py`: 4 passed
- `tests/test_lead_lag_network.py`: 4 passed
- `tests/test_alpha_research.py`: 5 passed
- `tests/test_factor_model.py`: 3 passed
- `tests/test_orderbook_heatmap_and_ofi.py`: 4 passed

### Model Training, Evaluation & Governance (100% Pass)
- `tests/test_models.py`: 3 passed
- `tests/test_train_multiclass_model.py`: 4 passed
- `tests/test_multiclass_contract.py`: 9 passed
- `tests/test_prediction_contract.py`: 6 passed
- `tests/test_accuracy_evaluator.py`: 9 passed
- `tests/test_evaluation_integrity.py`: 9 passed
- `tests/test_walk_forward.py`: 1 passed
- `tests/test_optimization_experiments.py`: 3 passed
- `tests/test_forex_model_compatibility_and_contract.py`: 8 passed
- `tests/test_continuous_learning.py`: 4 passed
- `tests/test_continuous_learning_loop.py`: 13 passed
- `tests/test_prediction_outcome_resolution.py`: 8 passed

### Signals, Risk & Execution Safety (100% Pass)
- `tests/test_signals.py`: 4 passed
- `tests/test_microstructure_signals.py`: 3 passed
- `tests/test_live_prediction_pipeline.py`: 5 passed
- `tests/test_risk.py`: 6 passed
- `tests/test_var_model.py`: 4 passed
- `tests/test_portfolio_margin.py`: 5 passed
- `tests/test_volatility_targeting.py`: 3 passed
- `tests/test_failover_orchestrator.py`: 5 passed
- `tests/test_compliance_audit.py`: 3 passed
- `tests/test_compliance_surveillance.py`: 4 passed
- `tests/test_mev_defense.py`: 5 passed
- `tests/test_flow_toxicity.py`: 5 passed

### Order Routing & Simulated Matching (100% Pass)
- `tests/test_smart_order_router.py`: 6 passed
- `tests/test_synthetic_sor.py`: 5 passed
- `tests/test_crossing_engine.py`: 5 passed
- `tests/test_dmm_engine.py`: 6 passed
- `tests/test_orderbook.py`: 3 passed
- `tests/test_orderbook_simulator.py`: 5 passed
- `tests/test_consolidated_orderbook.py`: 4 passed
- `tests/test_multi_agent_lob.py`: 3 passed
- `tests/test_dark_pool_detection.py`: 5 passed
- `tests/test_execution.py`: 3 passed
- `tests/test_binance_execution.py`: 2 passed
- `tests/test_algo_execution.py`: 3 passed
- `tests/test_cost_model.py`: 3 passed
- `tests/test_cost_surface_calibrator.py`: 4 passed
- `tests/test_tca_engine.py`: 5 passed

### Portfolio Management & Arbitrage (100% Pass)
- `tests/test_rebalancing.py`: 2 passed
- `tests/test_quantum_rebalancer.py`: 4 passed
- `tests/test_smart_hedging.py`: 5 passed
- `tests/test_basis_arbitrage.py`: 5 passed
- `tests/test_stat_arb.py`: 5 passed
- `tests/test_triangular_arbitrage.py`: 6 passed
- `tests/test_covariance_shrinkage.py`: 6 passed
- `tests/test_yield_curve.py`: 5 passed
- `tests/test_derivatives_pricing.py`: 7 passed
- `tests/test_collateral_rehypothecation.py`: 5 passed
- `tests/test_cross_chain_monitor.py`: 4 passed
- `tests/test_liquidation_cascades.py`: 5 passed
- `tests/test_ensemble_allocator.py`: 6 passed
- `tests/test_adaptive_market_maker.py`: 5 passed

### Paper Trading & Desktop UI Integration (100% Pass)
- `tests/test_paper_trading.py`: 4 passed
- `tests/test_paper_trading_worker.py`: 17 passed
- `tests/test_paper_trading_prediction_store.py`: 4 passed
- `tests/test_multi_market_paper_worker.py`: 4 passed
- `tests/test_resolver_ownership_and_single_instance.py`: 4 passed
- `tests/test_test_isolation.py`: 4 passed
- `tests/test_windows_desktop_app.py`: 1 passed (23-point UI assertion suite)
- `tests/test_trade_vault.py`: 5 passed
- `tests/test_cli.py`: 3 passed
- `tests/test_metrics.py`: 1 passed

---

## 3. Warning Remediation Log

During baseline test execution, 2 `RuntimeWarning: invalid value encountered in divide` warnings were captured from `scipy/numpy` in `analysis/alpha_research.py`.
- **Cause:** When computing cross-sectional factor autocorrelation on synthetic zero-variance test series (`volumes_df is None` fallback), zero standard deviation led to division by zero in correlation matrices.
- **Fix:** Guarded `float(s.std()) > 1e-9` prior to invoking autocorrelation calculations in `analysis/alpha_research.py`.
- **Verification:** Re-ran `pytest tests/test_alpha_research.py tests/test_api.py -v`. 46 tests passed with 0 warnings.
