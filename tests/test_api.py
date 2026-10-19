"""Tests for FastAPI REST Application Endpoints."""

import pytest
from httpx import ASGITransport, AsyncClient

from api.main import app


@pytest.mark.asyncio
async def test_api_health_and_metrics():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        # 1. Health endpoint
        res = await client.get("/health")
        assert res.status_code == 200
        data = res.json()
        assert "status" in data

        # 2. Metrics endpoint
        res_m = await client.get("/metrics")
        assert res_m.status_code == 200
        metrics_data = res_m.json()
        assert "provider" in metrics_data


@pytest.mark.asyncio
async def test_api_portfolio_state():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        res = await client.get("/portfolio/state")
        assert res.status_code == 200
        data = res.json()
        assert "cash" in data
        assert "equity" in data
        assert "active_positions_count" in data


@pytest.mark.asyncio
async def test_api_signals_and_risk_evaluation():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        # 1. Evaluate signals
        sig_req = {
            "symbol": "BTC-USD",
            "current_price": 65000.0,
            "atr": 450.0,
        }
        res_sig = await client.post("/signals/evaluate", json=sig_req)
        assert res_sig.status_code == 200
        sig_data = res_sig.json()
        assert "action" in sig_data
        assert "confidence" in sig_data
        assert "risk_reward_ratio" in sig_data

        # 2. Evaluate risk
        res_risk = await client.post("/risk/evaluate", json=sig_data)
        assert res_risk.status_code == 200
        risk_data = res_risk.json()
        assert "status" in risk_data
        assert "approved_quantity" in risk_data


@pytest.mark.asyncio
async def test_api_prometheus_and_dashboard():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        # Prometheus metrics endpoint
        res_p = await client.get("/prometheus")
        assert res_p.status_code == 200
        assert "portfolio_equity_usd" in res_p.text

        # Dashboard UI endpoint
        res_d = await client.get("/dashboard")
        assert res_d.status_code == 200
        assert "text/html" in res_d.headers.get("content-type", "")
        assert "Trading Intelligence Platform" in res_d.text


@pytest.mark.asyncio
async def test_api_consolidated_orderbook_and_smart_routing():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        # 1. Consolidated BBO endpoint
        res_bbo = await client.get("/orderbook/BTC-USD/consolidated")
        assert res_bbo.status_code == 200
        bbo_data = res_bbo.json()
        assert "bbo" in bbo_data
        assert "depth" in bbo_data
        assert bbo_data["bbo"]["symbol"] == "BTC-USD"

        # 2. Smart routing plan endpoint
        route_req = {
            "symbol": "BTC-USD",
            "side": "BUY",
            "quantity": 1.0,
            "policy": "BEST_PRICE",
            "execute": False,
        }
        res_route = await client.post("/execution/smart-route", json=route_req)
        assert res_route.status_code == 200
        route_data = res_route.json()
        assert "plan" in route_data
        assert route_data["plan"]["parent_quantity"] == 1.0
        assert len(route_data["plan"]["slices"]) >= 1

        # 3. Arbitrage scan endpoint
        res_arb = await client.get("/arbitrage/scan")
        assert res_arb.status_code == 200
        arb_data = res_arb.json()
        assert "active_opportunities" in arb_data
        assert "stats" in arb_data


@pytest.mark.asyncio
async def test_api_risk_var_stress_and_factors():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        # 1. VaR endpoint
        var_req = {
            "weights": {"BTC-USD": 0.6, "ETH-USD": 0.4},
            "portfolio_value": 100000.0,
            "confidence_level": 0.95,
        }
        res_var = await client.post("/risk/var", json=var_req)
        assert res_var.status_code == 200
        var_data = res_var.json()
        assert "parametric_var" in var_data
        assert "historical_var" in var_data
        assert "monte_carlo_var" in var_data
        assert var_data["parametric_var"] > 0

        # 2. Stress testing endpoint
        stress_req = {
            "portfolio_equity": 100000.0,
            "allocations_usd": {"BTC-USD": 60000.0, "ETH-USD": 40000.0},
        }
        res_stress = await client.post("/risk/stress-test", json=stress_req)
        assert res_stress.status_code == 200
        stress_reports = res_stress.json()
        assert len(stress_reports) == 5
        assert "portfolio_drawdown_pct" in stress_reports[0]

        # 3. Factors endpoint
        factor_req = {
            "weights": {"BTC-USD": 0.6, "ETH-USD": 0.4},
        }
        res_factors = await client.post("/risk/factors", json=factor_req)
        assert res_factors.status_code == 200
        factor_data = res_factors.json()
        assert "portfolio_beta_market" in factor_data
        assert "systematic_risk_pct" in factor_data


@pytest.mark.asyncio
async def test_api_volatility_targeting_rebalance_and_cost():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        # 1. Volatility Targeting endpoint
        vol_req = {
            "base_weights": {"BTC-USD": 0.6, "ETH-USD": 0.4},
            "target_volatility_annualized": 0.15,
        }
        res_vol = await client.post("/portfolio/volatility-target", json=vol_req)
        assert res_vol.status_code == 200
        vol_data = res_vol.json()
        assert "applied_multiplier" in vol_data
        assert "target_weights" in vol_data

        # 2. Rebalancing Plan endpoint
        reb_req = {
            "current_holdings": {"BTC-USD": 1.0, "ETH-USD": 5.0},
            "current_cash": 20000.0,
            "current_prices": {"BTC-USD": 65000.0, "ETH-USD": 3500.0},
            "target_weights": {"BTC-USD": 0.50, "ETH-USD": 0.40},
            "deadband_drift_pct": 0.02,
        }
        res_reb = await client.post("/portfolio/rebalance/plan", json=reb_req)
        assert res_reb.status_code == 200
        reb_data = res_reb.json()
        assert "orders" in reb_data
        assert "estimated_turnover_usd" in reb_data

        # 3. Cost Estimate endpoint
        cost_req = {
            "symbol": "BTC-USD",
            "side": "BUY",
            "quantity": 2.0,
            "current_price": 65000.0,
        }
        res_cost = await client.post("/execution/cost-estimate", json=cost_req)
        assert res_cost.status_code == 200
        cost_data = res_cost.json()
        assert "total_shortfall_bps" in cost_data
        assert "recommended_strategy" in cost_data


@pytest.mark.asyncio
async def test_api_algo_execution_suite():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        # 1. Volume Profile endpoint
        res_prof = await client.get("/execution/algo/profile/BTC-USD")
        assert res_prof.status_code == 200
        prof_data = res_prof.json()
        assert "bin_fractions" in prof_data
        assert len(prof_data["bin_fractions"]) == 24

        # 2. Algo Schedule endpoint
        sched_req = {
            "symbol": "BTC-USD",
            "side": "BUY",
            "quantity": 1.5,
            "algo_type": "TWAP",
            "duration_minutes": 15.0,
            "num_slices": 5,
            "arrival_price": 65000.0,
        }
        res_sched = await client.post("/execution/algo/schedule", json=sched_req)
        assert res_sched.status_code == 200
        plan_data = res_sched.json()
        assert plan_data["algo_type"] == "TWAP"
        assert len(plan_data["slices"]) == 5
        assert plan_data["parent_quantity"] == 1.5

        # 3. Algo Execute endpoint
        res_exec = await client.post("/execution/algo/execute", json=plan_data)
        assert res_exec.status_code == 200
        exec_data = res_exec.json()
        assert exec_data["status"] == "COMPLETED"
        assert exec_data["filled_quantity"] == 1.5
        assert "arrival_slippage_bps" in exec_data
        assert "market_vwap_slippage_bps" in exec_data


@pytest.mark.asyncio
async def test_api_compliance_audit_and_surveillance():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        # 1. Audit Verify endpoint
        res_v = await client.get("/compliance/audit/verify")
        assert res_v.status_code == 200
        v_data = res_v.json()
        assert v_data["is_valid"] is True
        assert v_data["total_audited_records"] >= 1

        # 2. Audit Record endpoint
        rec_req = {
            "record_type": "ORDER_SUBMISSION",
            "action": "API_LIMIT_BUY",
            "payload": {"symbol": "BTC-USD", "quantity": 0.5, "price": 64500.0},
            "actor": "API_TEST_RUNNER",
        }
        res_rec = await client.post("/compliance/audit/record", json=rec_req)
        assert res_rec.status_code == 200
        rec_data = res_rec.json()
        assert "entry_hash" in rec_data
        assert rec_data["sequence"] >= 1

        # Re-verify integrity
        res_v2 = await client.get("/compliance/audit/verify")
        assert res_v2.json()["is_valid"] is True

        # 3. MiFID export endpoint
        res_mifid = await client.get("/compliance/export/mifid")
        assert res_mifid.status_code == 200
        mifid_data = res_mifid.json()
        assert len(mifid_data) >= 2

        # 4. Surveillance Scan endpoint
        surv_req = {
            "orders": [
                {
                    "symbol": "BTC-USD",
                    "side": "BUY",
                    "price": 65000.0,
                    "quantity": 1.0,
                    "timestamp": "2024-01-01T12:00:00Z",
                },
                {
                    "symbol": "BTC-USD",
                    "side": "SELL",
                    "price": 65000.0,
                    "quantity": 1.0,
                    "timestamp": "2024-01-01T12:00:01Z",
                },
            ]
        }
        res_surv = await client.post("/compliance/surveillance/scan", json=surv_req)
        assert res_surv.status_code == 200
        surv_alerts = res_surv.json()
        assert len(surv_alerts) >= 1
        assert surv_alerts[0]["pattern"] == "WASH_TRADING"


@pytest.mark.asyncio
async def test_api_disaster_recovery_and_ha():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        # 1. Snapshot Create endpoint
        res_create = await client.post("/recovery/snapshot/create", json={"snapshot_id": "api_test_snap_001"})
        assert res_create.status_code == 200
        meta = res_create.json()
        assert meta["snapshot_id"] == "api_test_snap_001"
        assert "checksum_sha256" in meta
        assert "portfolio_equity" in meta

        # 2. Snapshot List endpoint
        res_list = await client.get("/recovery/snapshot/list")
        assert res_list.status_code == 200
        list_data = res_list.json()
        assert "snapshots" in list_data
        assert any(s["snapshot_id"] == "api_test_snap_001" for s in list_data["snapshots"])

        # 3. Snapshot Restore endpoint
        res_restore = await client.post("/recovery/snapshot/restore", json={"snapshot_id": "api_test_snap_001"})
        assert res_restore.status_code == 200
        restore_data = res_restore.json()
        assert restore_data["status"] == "RESTORED"
        assert restore_data["snapshot_id"] == "api_test_snap_001"
        assert "equity" in restore_data

        # 4. HA Status endpoint
        res_ha = await client.get("/recovery/ha/status")
        assert res_ha.status_code == 200
        ha_data = res_ha.json()
        assert "node_id" in ha_data
        assert "role" in ha_data
        assert ha_data["role"] in ["PRIMARY", "STANDBY"]
        assert "term_epoch" in ha_data


@pytest.mark.asyncio
async def test_api_stress_benchmark():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        req = {"tick_count": 500, "symbol": "BTC-USD"}
        res = await client.post("/system/benchmark/stress", json=req)
        assert res.status_code == 200
        data = res.json()
        assert data["total_events"] == 500
        assert data["throughput_events_per_sec"] > 0.0
        assert "latency_p50_us" in data
        assert "latency_p99_us" in data


@pytest.mark.asyncio
async def test_api_pairs_trading_endpoints():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        # 1. Cointegration test endpoint
        prices_x = [60000.0 + (i * 10.0) for i in range(50)]
        prices_y = [(px * 0.05) + (i % 3) for i, px in enumerate(prices_x)]

        coint_req = {
            "asset_y": "ETH-USD",
            "asset_x": "BTC-USD",
            "prices_y": prices_y,
            "prices_x": prices_x,
        }
        res_coint = await client.post("/analysis/pairs/cointegration", json=coint_req)
        assert res_coint.status_code == 200
        coint_data = res_coint.json()
        assert "is_cointegrated" in coint_data
        assert "static_hedge_ratio" in coint_data
        assert "half_life_bars" in coint_data

        # 2. Spread signal endpoint
        spread_req = {
            "pair_name": "ETH/BTC",
            "asset_y": "ETH-USD",
            "asset_x": "BTC-USD",
            "price_y": 3000.0,
            "price_x": 60000.0,
            "z_entry": 2.0,
            "z_exit": 0.5,
        }
        res_sig = await client.post("/analysis/pairs/spread-signal", json=spread_req)
        assert res_sig.status_code == 200
        sig_data = res_sig.json()
        assert sig_data["pair_name"] == "ETH/BTC"
        assert "action" in sig_data
        assert "z_score" in sig_data
        assert "weight_y" in sig_data
        assert "weight_x" in sig_data


@pytest.mark.asyncio
async def test_api_ofi_and_heatmap_endpoints():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        # 1. OFI compute endpoint
        ofi_req = {
            "bbo_updates": [
                {"bid_price": 65000.0, "bid_size": 2.0, "ask_price": 65010.0, "ask_size": 3.0},
                {"bid_price": 65002.0, "bid_size": 4.0, "ask_price": 65010.0, "ask_size": 3.0},
                {"bid_price": 65002.0, "bid_size": 1.0, "ask_price": 65008.0, "ask_size": 5.0},
            ],
            "rolling_window": 10,
        }
        res_ofi = await client.post("/orderbook/ofi/compute", json=ofi_req)
        assert res_ofi.status_code == 200
        ofi_data = res_ofi.json()
        assert len(ofi_data) == 3
        assert "instantaneous_ofi" in ofi_data[1]
        assert "predicted_tick_direction" in ofi_data[1]

        # 2. Heatmap endpoint
        res_hm = await client.get("/orderbook/BTC-USD/heatmap?levels=15")
        assert res_hm.status_code == 200
        hm_data = res_hm.json()
        assert hm_data["symbol"] == "BTC-USD"
        assert "price_levels" in hm_data
        assert "liquidity_concentration_index" in hm_data


@pytest.mark.asyncio
async def test_api_portfolio_margin_endpoint():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        margin_req = {
            "cash_balance": 50000.0,
            "collateral_assets": [
                {"asset": "BTC", "quantity": 1.0, "mark_price": 60000.0, "haircut_pct": 0.10}
            ],
            "positions": [
                {"symbol": "BTC-PERP", "quantity": 2.0, "mark_price": 60000.0, "side": "LONG", "initial_margin_rate": 0.10, "maintenance_margin_rate": 0.05, "unrealized_pnl": 4000.0},
                {"symbol": "ETH-PERP", "quantity": 10.0, "mark_price": 3000.0, "side": "SHORT", "initial_margin_rate": 0.10, "maintenance_margin_rate": 0.05, "unrealized_pnl": -500.0},
            ],
        }
        res = await client.post("/risk/margin/evaluate", json=margin_req)
        assert res.status_code == 200
        data = res.json()
        assert "total_nominal_equity_usd" in data
        assert "total_collateral_value_usd" in data
        assert "total_initial_margin_usd" in data
        assert "total_maintenance_margin_usd" in data
        assert "margin_usage_pct" in data
        assert "free_margin_usd" in data
        assert "status" in data
        assert data["status"] == "HEALTHY"
        assert data["free_margin_usd"] > 0


@pytest.mark.asyncio
async def test_api_derivatives_pricing_and_greeks():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        # 1. Black-Scholes Pricing & Greeks
        bs_req = {
            "spot": 65000.0,
            "strike": 65000.0,
            "time_to_expiry": 0.25,
            "volatility": 0.55,
            "risk_free_rate": 0.04,
            "dividend_yield": 0.0,
            "option_type": "CALL",
        }
        res_bs = await client.post("/derivatives/pricing/black-scholes", json=bs_req)
        assert res_bs.status_code == 200
        bs_data = res_bs.json()
        assert "price" in bs_data
        assert "delta" in bs_data
        assert "gamma" in bs_data
        assert "vega_1pct" in bs_data
        assert "theta_daily" in bs_data
        assert bs_data["delta"] > 0.4 and bs_data["delta"] < 0.7

        # 2. Merton Jump-Diffusion
        merton_req = {
            "spot": 65000.0,
            "strike": 65000.0,
            "time_to_expiry": 0.25,
            "volatility": 0.50,
            "jump_intensity": 1.0,
            "jump_mean": -0.05,
            "jump_volatility": 0.15,
            "risk_free_rate": 0.04,
            "dividend_yield": 0.0,
            "option_type": "CALL",
        }
        res_merton = await client.post("/derivatives/pricing/merton", json=merton_req)
        assert res_merton.status_code == 200
        merton_data = res_merton.json()
        assert "price" in merton_data
        assert merton_data["price"] > 0

        # 3. Implied Volatility Solver
        iv_req = {
            "market_price": bs_data["price"],
            "spot": 65000.0,
            "strike": 65000.0,
            "time_to_expiry": 0.25,
            "risk_free_rate": 0.04,
            "dividend_yield": 0.0,
            "option_type": "CALL",
        }
        res_iv = await client.post("/derivatives/volatility/solve-iv", json=iv_req)
        assert res_iv.status_code == 200
        iv_data = res_iv.json()
        assert iv_data["is_valid"] is True
        assert abs(iv_data["implied_volatility"] - 0.55) < 1e-3

        # 4. Volatility Surface Skew
        skew_req = {
            "spot": 65000.0,
            "quotes": [
                {"strike": 55000.0, "time_to_expiry": 0.082, "implied_vol": 0.65},
                {"strike": 65000.0, "time_to_expiry": 0.082, "implied_vol": 0.52},
                {"strike": 75000.0, "time_to_expiry": 0.082, "implied_vol": 0.58},
            ],
            "time_to_expiry": 0.082,
        }
        res_skew = await client.post("/derivatives/volatility/surface/skew", json=skew_req)
        assert res_skew.status_code == 200
        skew_data = res_skew.json()
        assert "metrics" in skew_data
        assert "atm_volatility" in skew_data["metrics"]

        # 5. Derivatives Portfolio Risk Aggregation
        port_req = {
            "positions": [
                {
                    "contract_id": "BTC-C-65000",
                    "symbol": "BTC-USD",
                    "option_type": "CALL",
                    "strike": 65000.0,
                    "time_to_expiry": 0.25,
                    "quantity": 5.0,
                    "mark_price": 5000.0,
                    "underlying_price": 65000.0,
                    "implied_volatility": 0.55,
                }
            ]
        }
        res_port = await client.post("/derivatives/portfolio/greeks", json=port_req)
        assert res_port.status_code == 200
        port_data = res_port.json()
        assert "net_delta" in port_data
        assert "dollar_delta_usd" in port_data
        assert "delta_neutral_hedge_shares" in port_data


@pytest.mark.asyncio
async def test_api_smart_hedging_endpoints():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        hedge_plan_req = {
            "portfolio_greeks": {
                "total_market_value_usd": 75000.0,
                "net_delta": 2.25,
                "dollar_delta_usd": 146250.0,
                "net_gamma": 0.0003,
                "dollar_gamma_usd": 1500.0,
                "net_vega_1pct_usd": 850.0,
                "net_theta_daily_usd": -120.0,
                "net_rho_1pct_usd": 40.0,
                "net_vanna_usd": 5.0,
                "net_volga_usd": 3.0,
                "delta_neutral_hedge_shares": -2.25,
                "timestamp": "2026-10-05T12:00:00Z",
            },
            "spot_symbol": "BTC-USD",
            "spot_price": 65000.0,
            "perp_symbol": "BTC-PERP",
            "perp_price": 65010.0,
            "delta_tolerance": 0.50,
        }
        res_plan = await client.post("/execution/hedge/plan", json=hedge_plan_req)
        assert res_plan.status_code == 200
        plan_data = res_plan.json()
        assert plan_data["hedging_required"] is True
        assert plan_data["trigger_reason"] == "DELTA_BAND_BREACH"
        assert len(plan_data["orders"]) == 1
        assert plan_data["orders"][0]["side"] == "SELL"
        assert plan_data["orders"][0]["quantity"] == 2.25

        # Execute hedge plan
        res_exec = await client.post("/execution/hedge/execute", json=plan_data)
        assert res_exec.status_code == 200
        exec_data = res_exec.json()
        assert len(exec_data) == 1
        assert exec_data[0]["status"] == "FILLED"


@pytest.mark.asyncio
async def test_api_basis_arbitrage_endpoints():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        # 1. Update spot quote
        spot_req = {
            "underlying": "BTC",
            "venue": "coinbase",
            "symbol": "BTC-USD",
            "bid": 65000.0,
            "ask": 65010.0,
        }
        res_s = await client.post("/arbitrage/basis/spot/update", json=spot_req)
        assert res_s.status_code == 200

        # 2. Update perp rate
        perp_req = {
            "underlying": "BTC",
            "venue": "binance",
            "symbol": "BTCUSDT",
            "rate_per_interval": 0.0003,
            "mark_price": 65200.0,
            "index_price": 65005.0,
            "interval_hours": 8,
        }
        res_p = await client.post("/arbitrage/basis/perp/update", json=perp_req)
        assert res_p.status_code == 200

        # 3. Scan opportunities
        res_scan = await client.get("/arbitrage/basis/scan?min_net_apr_pct=3.0")
        assert res_scan.status_code == 200
        opps = res_scan.json()
        assert len(opps) >= 1
        assert "opportunity_id" in opps[0]
        assert "strategy_type" in opps[0]
        assert "estimated_net_apr_pct" in opps[0]

        # 4. Open basis position
        open_req = {"opportunity": opps[0], "quantity": 1.5}
        res_open = await client.post("/arbitrage/basis/position/open", json=open_req)
        assert res_open.status_code == 200
        pos_data = res_open.json()
        assert "position_id" in pos_data
        assert pos_data["quantity"] == 1.5
        assert pos_data["is_active"] is True

        # 5. List basis positions
        res_list = await client.get("/arbitrage/basis/positions")
        assert res_list.status_code == 200
        pos_list = res_list.json()
        assert len(pos_list) >= 1
        assert any(p["position_id"] == pos_data["position_id"] for p in pos_list)


@pytest.mark.asyncio
async def test_api_internal_crossing_engine_endpoints():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        # 1. Inspect crossing book
        res_b = await client.get("/execution/crossing/BTC-USD/book")
        assert res_b.status_code == 200
        book_data = res_b.json()
        assert "resting_buys" in book_data
        assert "resting_sells" in book_data

        # 2. Submit matching order
        cross_req = {
            "symbol": "BTC-USD",
            "side": "BUY",
            "quantity": 0.5,
            "time_in_force": "IOC",
            "strategy_id": "TEST_CROSS_DESK",
            "current_bid": 65000.0,
            "current_ask": 65010.0,
        }
        res_c = await client.post("/execution/crossing/submit", json=cross_req)
        assert res_c.status_code == 200
        c_data = res_c.json()
        assert c_data["filled_quantity"] == 0.5
        assert c_data["average_match_price"] == 65005.0
        assert c_data["total_spread_savings_usd"] > 0
        assert c_data["total_fee_savings_usd"] > 0

        # 3. Retrieve crossing pool metrics
        res_m = await client.get("/execution/crossing/BTC-USD/metrics")
        assert res_m.status_code == 200
        m_data = res_m.json()
        assert m_data["cumulative_crossed_volume"] >= 0.5
        assert m_data["cumulative_total_savings_usd"] > 0

        # 4. Post resting order on SOL-USD and cancel
        resting_req = {
            "symbol": "SOL-USD",
            "side": "BUY",
            "quantity": 5.0,
            "time_in_force": "RESTING",
            "strategy_id": "TEST_RESTING_ORDER",
            "current_bid": 150.0,
            "current_ask": 150.2,
        }
        res_post = await client.post("/execution/crossing/submit", json=resting_req)
        assert res_post.status_code == 200
        post_data = res_post.json()
        assert post_data["remaining_unfilled_quantity"] == 5.0
        order_id = post_data["incoming_order_id"]

        res_del = await client.delete(f"/execution/crossing/SOL-USD/order/{order_id}")
        assert res_del.status_code == 200
        del_data = res_del.json()
        assert del_data["cancelled"] is True


@pytest.mark.asyncio
async def test_api_dmm_endpoints():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        # 1. Reset inventory
        res_reset = await client.post("/execution/dmm/inventory/reset", json={"symbol": "BTC-USD", "inventory": 0.0})
        assert res_reset.status_code == 200
        assert res_reset.json()["inventory"] == 0.0

        # 2. Calculate DMM quotes
        quote_req = {
            "symbol": "BTC-USD",
            "best_bid": 65000.0,
            "best_ask": 65010.0,
            "bid_volume": 2.5,
            "ask_volume": 1.5,
            "volatility_annualized": 0.50,
            "order_flow_imbalance": 0.0,
        }
        res_q = await client.post("/execution/dmm/quotes", json=quote_req)
        assert res_q.status_code == 200
        q_data = res_q.json()
        assert q_data["symbol"] == "BTC-USD"
        assert "mid_price" in q_data
        assert "reservation_price" in q_data
        assert "quotes" in q_data
        assert len(q_data["quotes"]) == 3
        assert q_data["inventory_state"] == "BALANCED"

        # 3. Record fill
        fill_req = {
            "symbol": "BTC-USD",
            "side": "BUY",
            "quantity": 1.5,
            "price": 65000.0,
        }
        res_fill = await client.post("/execution/dmm/fill", json=fill_req)
        assert res_fill.status_code == 200
        assert res_fill.json()["inventory"] == 1.5

        # 4. Check inventory endpoint
        res_inv = await client.get("/execution/dmm/BTC-USD/inventory")
        assert res_inv.status_code == 200
        inv_data = res_inv.json()
        assert inv_data["inventory_units"] == 1.5
        assert inv_data["max_inventory_units"] == 5.0
        assert inv_data["inventory_utilization_pct"] == 30.0


@pytest.mark.asyncio
async def test_api_tca_endpoints():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        trade_payload = {
            "order_id": "ord_tca_1",
            "symbol": "BTC-USD",
            "side": "BUY",
            "venue": "coinbase",
            "order_quantity": 1.0,
            "filled_quantity": 1.0,
            "decision_price": 65000.0,
            "arrival_price": 65002.0,
            "arrival_bid": 65001.0,
            "arrival_ask": 65003.0,
            "execution_price": 65004.0,
            "terminal_price": 65010.0,
            "decision_time": "2026-10-05T12:00:00Z",
            "arrival_time": "2026-10-05T12:00:00.020Z",
            "execution_time": "2026-10-05T12:00:00.080Z",
            "venue_fees_usd": 6.5,
        }

        # 1. Decompose trade
        res_d = await client.post("/execution/tca/decompose", json=trade_payload)
        assert res_d.status_code == 200
        d_data = res_d.json()
        assert d_data["symbol"] == "BTC-USD"
        assert "total_shortfall_usd" in d_data
        assert "delay_cost_usd" in d_data
        assert "spread_cost_usd" in d_data
        assert "fee_drag_usd" in d_data
        assert d_data["total_shortfall_usd"] > 0

        # 2. Portfolio report
        report_req = {"trades": [trade_payload]}
        res_r = await client.post("/execution/tca/report", json=report_req)
        assert res_r.status_code == 200
        r_data = res_r.json()
        assert r_data["total_orders_analyzed"] == 1
        assert "aggregate_shortfall_bps" in r_data
        assert "venue_scorecards" in r_data
        assert "coinbase" in r_data["venue_scorecards"]

        # 3. Venue scorecards
        res_v = await client.post("/execution/tca/venues/scorecards", json=report_req)
        assert res_v.status_code == 200
        v_data = res_v.json()
        assert "coinbase" in v_data
        assert v_data["coinbase"]["fill_rate_pct"] == 100.0
        assert "effective_to_quoted_ratio" in v_data["coinbase"]


@pytest.mark.asyncio
async def test_api_alpha_research_endpoints():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        # 1. Run Alpha Pipeline
        pipeline_req = {
            "symbols": ["BTC-USD", "ETH-USD", "SOL-USD", "AVAX-USD", "LINK-USD"],
            "periods": 45,
            "forward_horizon": 1,
        }
        res_pipe = await client.post("/analysis/alpha/pipeline/run", json=pipeline_req)
        assert res_pipe.status_code == 200
        p_data = res_pipe.json()
        assert "factor_evaluations" in p_data
        assert "neutralization_factors_applied" in p_data
        assert "top_ranked_assets" in p_data
        assert "bottom_ranked_assets" in p_data
        assert "asset_scores" in p_data
        assert len(p_data["top_ranked_assets"]) > 0


        # 2. Factor Neutralization endpoint
        neut_req = {
            "factor_scores": {"BTC-USD": 1.5, "ETH-USD": -0.8, "SOL-USD": 0.5, "AVAX-USD": -1.2},
            "risk_exposures": {
                "BTC-USD": {"MarketBeta": 1.0, "LogPrice": 11.0},
                "ETH-USD": {"MarketBeta": 1.1, "LogPrice": 8.1},
                "SOL-USD": {"MarketBeta": 1.4, "LogPrice": 5.0},
                "AVAX-USD": {"MarketBeta": 1.3, "LogPrice": 3.4},
            },
        }
        res_neut = await client.post("/analysis/alpha/factor/neutralize", json=neut_req)
        assert res_neut.status_code == 200
        n_data = res_neut.json()
        assert "neutralized_scores" in n_data
        assert "BTC-USD" in n_data["neutralized_scores"]
        assert len(n_data["neutralized_scores"]) == 4


@pytest.mark.asyncio
async def test_api_collateral_rehypothecation_endpoints():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        # 1. Venues collateral status
        res_stat = await client.get("/risk/collateral/venues/status")
        assert res_stat.status_code == 200
        stat_data = res_stat.json()
        assert "coinbase" in stat_data
        assert "binance" in stat_data
        assert "vault_fireblocks" in stat_data
        assert "margin_utilization_pct" in stat_data["coinbase"]
        assert "health_status" in stat_data["coinbase"]

        # 2. Generate rebalance plan
        res_plan = await client.post("/risk/collateral/rebalance/plan", json={})
        assert res_plan.status_code == 200
        plan_data = res_plan.json()
        assert "total_transfers_count" in plan_data
        assert "venue_initial_utilizations" in plan_data
        assert "venue_projected_utilizations" in plan_data
        assert "recommended_action" in plan_data

        # 3. Execute rebalance plan (dry run)
        exec_req = {"plan": plan_data, "dry_run": True}
        res_exec = await client.post("/risk/collateral/rebalance/execute", json=exec_req)
        assert res_exec.status_code == 200
        exec_data = res_exec.json()
        assert exec_data["status"] == "COMPLETED"
        assert exec_data["dry_run"] is True
        assert "post_rebalance_venues" in exec_data


@pytest.mark.asyncio
async def test_api_ensemble_allocator_endpoints():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        # 1. Ensemble Status
        res_stat = await client.get("/analysis/ensemble/status")
        assert res_stat.status_code == 200
        stat_data = res_stat.json()
        assert "algorithm" in stat_data
        assert "strategy_rankings" in stat_data
        assert "arms" in stat_data
        assert "MOMENTUM_TREND" in stat_data["arms"]

        # 2. Capital Allocation
        alloc_req = {
            "total_capital_usd": 150000.0,
            "current_regime": "BULL_TRENDING",
        }
        res_alloc = await client.post("/analysis/ensemble/allocate", json=alloc_req)
        assert res_alloc.status_code == 200
        alloc_data = res_alloc.json()
        assert alloc_data["market_regime"] == "BULL_TRENDING"
        assert alloc_data["total_capital_usd"] == 150000.0
        assert "strategy_allocations" in alloc_data
        assert "strategy_capital_usd" in alloc_data
        assert abs(sum(alloc_data["strategy_allocations"].values()) - 1.0) < 1e-3
        assert abs(sum(alloc_data["strategy_capital_usd"].values()) - 150000.0) < 1.0

        # 3. Update Reward
        reward_req = {
            "strategy_id": "MOMENTUM_TREND",
            "realized_pnl_usd": 750.0,
            "notional_traded_usd": 10000.0,
        }
        res_rew = await client.post("/analysis/ensemble/reward/update", json=reward_req)
        assert res_rew.status_code == 200
        rew_data = res_rew.json()
        assert rew_data["strategy_id"] == "MOMENTUM_TREND"
        assert rew_data["updated_pull_count"] >= 1
        assert "normalized_reward" in rew_data
        assert rew_data["normalized_reward"] > 0


@pytest.mark.asyncio
async def test_api_dark_pool_detection_endpoints():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        # 1. Process dark cross trade
        dark_trade_req = {
            "symbol": "BTC-USD",
            "venue": "coinbase",
            "price": 65010.0,
            "quantity": 1.5,
            "best_bid": 65000.0,
            "best_ask": 65020.0,
        }
        res_trade = await client.post("/analysis/darkpool/trades/process", json=dark_trade_req)
        assert res_trade.status_code == 200
        trade_data = res_trade.json()
        assert trade_data is not None
        assert trade_data["symbol"] == "BTC-USD"
        assert trade_data["liquidity_type"] == "DARK_CROSS_MIDPOINT"
        assert trade_data["spread_bps_at_time"] > 0

        # 2. Process book depth update
        depth_req = {
            "symbol": "BTC-USD",
            "venue": "binance",
            "side": "BUY",
            "price": 65000.0,
            "quantity": 5.0,
        }
        res_depth = await client.post("/analysis/darkpool/icebergs/update", json=depth_req)
        assert res_depth.status_code == 200

        # 3. Retrieve consolidated report
        res_rep = await client.get("/analysis/darkpool/report/BTC-USD")
        assert res_rep.status_code == 200
        rep_data = res_rep.json()
        assert rep_data["symbol"] == "BTC-USD"
        assert "active_icebergs" in rep_data
        assert "recent_dark_prints" in rep_data
        assert "dark_volume_share_pct" in rep_data
        assert "market_impact_implication" in rep_data
        assert len(rep_data["recent_dark_prints"]) >= 1


@pytest.mark.asyncio
async def test_api_preflight_compliance_endpoints():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        # 1. Retrieve PreFlight Risk Limits
        res_lim = await client.get("/compliance/preflight/limits")
        assert res_lim.status_code == 200
        lim_data = res_lim.json()
        assert "max_single_order_notional_usd" in lim_data
        assert "max_price_collar_pct" in lim_data
        assert "max_orders_per_second" in lim_data

        # 2. Validate clean order
        clean_order_req = {
            "order": {
                "cl_ord_id": "API_ORD_VALID_01",
                "symbol": "BTC-USD",
                "side": "BUY",
                "order_qty": 1.0,
                "ord_type": "LIMIT",
                "price": 65050.0,
                "account": "DESK_01",
            },
            "reference_mid_price": 65000.0,
        }
        res_val = await client.post("/compliance/preflight/validate", json=clean_order_req)
        assert res_val.status_code == 200
        val_data = res_val.json()
        assert val_data["is_approved"] is True
        assert val_data["reject_reason"] == "APPROVED"
        assert val_data["evaluated_notional_usd"] == 65050.0

        # 3. Validate fat-finger order
        fat_finger_req = {
            "order": {
                "cl_ord_id": "API_ORD_FAT_01",
                "symbol": "BTC-USD",
                "side": "BUY",
                "order_qty": 1.0,
                "ord_type": "LIMIT",
                "price": 85000.0,  # Far above 3% band
                "account": "DESK_01",
            },
            "reference_mid_price": 65000.0,
        }
        res_fat = await client.post("/compliance/preflight/validate", json=fat_finger_req)
        assert res_fat.status_code == 200
        fat_data = res_fat.json()
        assert fat_data["is_approved"] is False
        assert fat_data["reject_reason"] == "FAT_FINGER_PRICE_COLLAR_BREACH"

        # 4. Reset daily counters
        res_rst = await client.post("/compliance/preflight/reset")
        assert res_rst.status_code == 200
        assert res_rst.json()["status"] == "DAILY_LIMITS_RESET"


@pytest.mark.asyncio
async def test_api_gateway_heartbeat_endpoints():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        # 1. Gateway Status Report
        res_status = await client.get("/monitoring/gateways/status")
        assert res_status.status_code == 200
        status_data = res_status.json()
        assert "gateways" in status_data
        assert "active_primary_venue" in status_data
        assert "coinbase" in status_data["gateways"]
        assert "binance" in status_data["gateways"]

        # 2. Record Gateway Ping
        ping_req = {
            "venue": "coinbase",
            "rtt_ms": 28.5,
        }
        res_ping = await client.post("/monitoring/gateways/ping", json=ping_req)
        assert res_ping.status_code == 200
        ping_data = res_ping.json()
        assert ping_data["venue"] == "coinbase"
        assert ping_data["status"] == "RECORDED"
        assert ping_data["current_rtt_ms"] == 28.5
        assert ping_data["p95_latency_ms"] > 0

        # 3. Trigger Gateway Reconnect
        reconn_req = {
            "venue": "kraken",
        }
        res_reconn = await client.post("/monitoring/gateways/reconnect", json=reconn_req)
        assert res_reconn.status_code == 200
        reconn_data = res_reconn.json()
        assert reconn_data["venue"] == "kraken"
        assert reconn_data["reconnect_scheduled"] is True
        assert reconn_data["connection_state"] == "RECONNECTING"


@pytest.mark.asyncio
async def test_api_cost_surface_calibrator_endpoints():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        # 1. Retrieve cost parameters
        res_params = await client.get("/execution/cost-surface/parameters/BTC-USD?venue=coinbase")
        assert res_params.status_code == 200
        p_data = res_params.json()
        assert p_data["symbol"] == "BTC-USD"
        assert p_data["venue"] == "coinbase"
        assert "calibrated_eta" in p_data
        assert "calibrated_gamma" in p_data

        # 2. Calibrate & generate cost surface grid
        cal_req = {"symbol": "BTC-USD", "venue": "coinbase"}
        res_grid = await client.post("/execution/cost-surface/calibrate", json=cal_req)
        assert res_grid.status_code == 200
        g_data = res_grid.json()
        assert "surface_matrix_bps" in g_data
        assert "participation_bins" in g_data
        assert "volatility_bins" in g_data
        assert len(g_data["surface_matrix_bps"]) > 0

        # 3. Predict slippage
        pred_req = {
            "symbol": "BTC-USD",
            "venue": "coinbase",
            "side": "BUY",
            "order_quantity": 2.5,
            "order_duration_seconds": 60.0,
            "reference_price": 65000.0,
            "realized_volatility": 0.50,
            "order_flow_imbalance": 0.20,
        }
        res_pred = await client.post("/execution/cost-surface/predict", json=pred_req)
        assert res_pred.status_code == 200
        pred_data = res_pred.json()
        assert pred_data["symbol"] == "BTC-USD"
        assert pred_data["expected_total_slippage_bps"] > 0
        assert pred_data["expected_total_slippage_usd"] > 0
        assert pred_data["recommended_urgency"] in ["PASSIVE", "NORMAL", "AGGRESSIVE"]


@pytest.mark.asyncio
async def test_api_triangular_arbitrage_endpoints():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        # 1. Update quotes to create a known profitable triangle
        # USD -> BTC -> ETH -> USD on Binance
        await client.post("/analysis/triangular/quotes/update", json={
            "venue": "binance", "symbol": "BTC-USD", "bid": 59990.0, "ask": 60000.0, "bid_qty": 5.0, "ask_qty": 5.0
        })
        await client.post("/analysis/triangular/quotes/update", json={
            "venue": "binance", "symbol": "ETH-BTC", "bid": 0.049, "ask": 0.050, "bid_qty": 100.0, "ask_qty": 100.0
        })
        await client.post("/analysis/triangular/quotes/update", json={
            "venue": "binance", "symbol": "ETH-USD", "bid": 3200.0, "ask": 3205.0, "bid_qty": 50.0, "ask_qty": 50.0
        })

        # 2. Scan triangular arbitrage
        res_scan = await client.get("/analysis/triangular/scan?anchor_asset=USD")
        assert res_scan.status_code == 200
        scan_data = res_scan.json()
        assert scan_data["total_currency_pairs_active"] >= 3
        assert scan_data["opportunities_count"] >= 1
        assert scan_data["best_opportunity"] is not None
        assert scan_data["best_opportunity"]["triangle_path"] == ["USD", "BTC", "ETH", "USD"]
        assert scan_data["best_opportunity"]["net_return_bps"] > 0

        # 3. Retrieve rate graph
        res_graph = await client.get("/analysis/triangular/graph")
        assert res_graph.status_code == 200
        graph_data = res_graph.json()
        assert "nodes" in graph_data
        assert "edges" in graph_data
        assert "USD" in graph_data["nodes"]
        assert "BTC" in graph_data["nodes"]
        assert "ETH" in graph_data["nodes"]
        assert len(graph_data["edges"]) >= 6


@pytest.mark.asyncio
async def test_api_covariance_shrinkage_endpoints():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        # 1. Test /risk/covariance/shrink
        returns_data = [
            [0.01, 0.02, 0.005],
            [-0.015, -0.01, 0.002],
            [0.02, 0.015, -0.01],
            [-0.005, 0.008, 0.012],
            [0.03, 0.025, -0.008],
        ]
        symbols = ["BTC", "ETH", "SOL"]

        shrink_req = {
            "returns_matrix": returns_data,
            "symbols": symbols,
            "target_type": "constant_correlation",
            "annualize": True,
        }
        res_shrink = await client.post("/risk/covariance/shrink", json=shrink_req)
        assert res_shrink.status_code == 200
        shrink_res = res_shrink.json()
        assert shrink_res["symbols"] == symbols
        assert shrink_res["asset_count_n"] == 3
        assert shrink_res["sample_size_t"] == 5
        assert 0.0 <= shrink_res["shrinkage_intensity"] <= 1.0
        assert shrink_res["is_positive_definite"] is True
        assert len(shrink_res["shrunk_covariance"]) == 3

        # 2. Test /risk/covariance/minimum-variance
        mv_req = {
            "returns_matrix": returns_data,
            "symbols": symbols,
            "target_type": "constant_correlation",
            "long_only": True,
            "annualize": True,
        }
        res_mv = await client.post("/risk/covariance/minimum-variance", json=mv_req)
        assert res_mv.status_code == 200
        mv_res = res_mv.json()
        assert "shrinkage_summary" in mv_res
        assert "minimum_variance" in mv_res
        weights = mv_res["minimum_variance"]["weights"]
        assert len(weights) == 3
        assert sum(weights.values()) == pytest.approx(1.0, rel=1e-4)
        assert mv_res["minimum_variance"]["portfolio_volatility"] > 0.0


@pytest.mark.asyncio
async def test_api_orderbook_replay_endpoints():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        # 1. Post snapshot
        snap_req = {
            "symbol": "BTC-USD",
            "timestamp_ns": 1_000_000_000,
            "bids": [[65000.0, 5.0], [64990.0, 10.0]],
            "asks": [[65005.0, 4.0], [65015.0, 8.0]],
        }
        res_snap = await client.post("/backtest/orderbook/snapshot", json=snap_req)
        assert res_snap.status_code == 200
        snap_data = res_snap.json()
        assert snap_data["status"] == "APPLIED"
        assert snap_data["best_bid"] == 65000.0
        assert snap_data["best_ask"] == 65005.0

        # 2. Submit order
        order_req = {
            "symbol": "BTC-USD",
            "order_id": "API_SIM_01",
            "side": "BUY",
            "order_type": "LIMIT",
            "quantity": 1.0,
            "limit_price": 65000.0,
            "custom_latency_ns": 0,
        }
        res_order = await client.post("/backtest/orderbook/order", json=order_req)
        assert res_order.status_code == 200
        order_data = res_order.json()
        assert order_data["order_id"] == "API_SIM_01"
        assert order_data["status"] == "ACTIVE"

        # 3. Retrieve report
        res_rep = await client.get("/backtest/orderbook/report?symbol=BTC-USD")
        assert res_rep.status_code == 200
        rep_data = res_rep.json()
        assert rep_data["symbol"] == "BTC-USD"
        assert rep_data["total_orders_submitted"] >= 1


@pytest.mark.asyncio
async def test_api_synthetic_sor_endpoints():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        # 1. Update venue depth
        update_req = {
            "venue": "OKX",
            "symbol": "BTC-USD",
            "bids": [[64993.0, 3.0], [64985.0, 5.0]],
            "asks": [[65003.0, 2.5], [65010.0, 6.0]],
            "maker_fee_bps": -1.5,
            "taker_fee_bps": 7.0,
            "latency_ms": 6.0,
        }
        res_up = await client.post("/execution/sor/venues/update", json=update_req)
        assert res_up.status_code == 200
        up_data = res_up.json()
        assert up_data["status"] == "UPDATED"
        assert up_data["venue"] == "OKX"

        # 2. Plan liquidity split
        split_req = {
            "symbol": "BTC-USD",
            "side": "BUY",
            "parent_quantity": 4.0,
            "objective": "min_total_net_cost",
            "max_venues": 4,
        }
        res_split = await client.post("/execution/sor/split", json=split_req)
        assert res_split.status_code == 200
        plan_data = res_split.json()
        assert plan_data["symbol"] == "BTC-USD"
        assert plan_data["parent_quantity"] == 4.0
        assert plan_data["is_fully_allocated"] is True
        assert len(plan_data["child_allocations"]) >= 2
        assert plan_data["blended_vwap"] > 0.0


@pytest.mark.asyncio
async def test_api_solvency_cascade_endpoints():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        # 1. Evaluate solvency
        eval_req = {
            "collateral_assets": [
                {
                    "asset": "USD",
                    "quantity": 50000.0,
                    "base_mark_price": 1.0,
                    "base_haircut": 0.0,
                    "volatility_haircut_sensitivity": 0.1,
                    "concentration_haircut_penalty": 0.0,
                }
            ],
            "positions": [
                {
                    "symbol": "BTC-USD",
                    "side": "LONG",
                    "quantity": 2.0,
                    "mark_price": 60000.0,
                    "entry_price": 60000.0,
                    "mm_rate": 0.05,
                }
            ],
            "market_stress_multiplier": 1.0,
        }
        res_eval = await client.post("/risk/margin/solvency/evaluate", json=eval_req)
        assert res_eval.status_code == 200
        eval_data = res_eval.json()
        assert eval_data["solvency_state"] == "OPTIMAL"
        assert "liquidation_plan" in eval_data

        # 2. Stress simulation
        stress_req = {
            "collateral_assets": [
                {
                    "asset": "BTC",
                    "quantity": 2.0,
                    "base_mark_price": 60000.0,
                    "base_haircut": 0.10,
                    "volatility_haircut_sensitivity": 1.0,
                    "concentration_haircut_penalty": 0.5,
                }
            ],
            "positions": [
                {
                    "symbol": "BTC-USD",
                    "side": "LONG",
                    "quantity": 10.0,
                    "mark_price": 60000.0,
                    "entry_price": 60000.0,
                    "mm_rate": 0.05,
                }
            ],
            "scenario_name": "Crash 35%",
            "price_shock_pct": -0.35,
            "volatility_shock_multiplier": 2.0,
        }
        res_stress = await client.post("/risk/margin/solvency/stress", json=stress_req)
        assert res_stress.status_code == 200
        stress_data = res_stress.json()
        assert stress_data["shock_scenario_name"] == "Crash 35%"
        assert stress_data["price_shock_pct"] == -0.35
        assert "breach_stage_reached" in stress_data


@pytest.mark.asyncio
async def test_api_flow_toxicity_endpoints():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        # 1. Ingest trade
        trade_req = {
            "symbol": "BTC-USD",
            "price": 65005.0,
            "quantity": 2.5,
            "side": "BUY",
        }
        res_trade = await client.post("/analysis/toxicity/trade", json=trade_req)
        assert res_trade.status_code == 200
        trade_data = res_trade.json()
        assert trade_data["status"] == "PROCESSED"
        assert trade_data["symbol"] == "BTC-USD"
        assert 0.0 <= trade_data["current_vpin"] <= 1.0

        # 2. Get snapshot
        res_snap = await client.get("/analysis/toxicity/snapshot?symbol=BTC-USD")
        assert res_snap.status_code == 200
        snap_data = res_snap.json()
        assert snap_data["symbol"] == "BTC-USD"
        assert "current_vpin" in snap_data
        assert "toxicity_regime" in snap_data
        assert "roll_effective_spread" in snap_data
        assert "recommendation" in snap_data


@pytest.mark.asyncio
async def test_api_yield_curve_endpoints():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        # 1. Calibrate curve
        fit_req = {
            "underlying": "BTC",
            "tenor_points": [
                {"maturity_years": 0.0833, "yield_rate": 0.040, "weight": 1.0},
                {"maturity_years": 0.25, "yield_rate": 0.048, "weight": 1.0},
                {"maturity_years": 0.50, "yield_rate": 0.055, "weight": 1.0},
                {"maturity_years": 1.0, "yield_rate": 0.065, "weight": 1.0},
                {"maturity_years": 2.0, "yield_rate": 0.075, "weight": 1.0},
                {"maturity_years": 5.0, "yield_rate": 0.090, "weight": 1.0},
            ],
            "model_type": "NELSON_SIEGEL_SVENSSON",
        }
        res_calib = await client.post("/analysis/yield-curve/calibrate", json=fit_req)
        assert res_calib.status_code == 200
        calib_data = res_calib.json()
        assert calib_data["underlying"] == "BTC"
        assert "parameters" in calib_data
        assert "tenor_points" in calib_data
        assert len(calib_data["tenor_points"]) >= 6

        # 2. Evaluate spot rate
        spot_req = {
            "parameters": calib_data["parameters"],
            "maturity_years": 1.5,
        }
        res_spot = await client.post("/analysis/yield-curve/spot-rate", json=spot_req)
        assert res_spot.status_code == 200
        spot_data = res_spot.json()
        assert spot_data["maturity_years"] == 1.5
        assert 0.0 < spot_data["spot_rate"] < 0.20
        assert 0.0 < spot_data["discount_factor"] < 1.0


@pytest.mark.asyncio
async def test_api_mev_defense_endpoints():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        # 1. Assess MEV Risk
        assess_req = {
            "order_id": "API_MEV_01",
            "symbol": "ETH-USDC",
            "notional_usd": 250000.0,
            "pool_liquidity_usd": 1000000.0,
            "requested_slippage_bps": 100.0,
            "current_gas_price_gwei": 35.0,
        }
        res_assess = await client.post("/compliance/mev/assess", json=assess_req)
        assert res_assess.status_code == 200
        assess_data = res_assess.json()
        assert assess_data["order_id"] == "API_MEV_01"
        assert assess_data["risk_level"] in ("HIGH", "CRITICAL")
        assert assess_data["private_bundle_required"] is True
        assert assess_data["recommended_relay"] in ("FLASHBOTS_PROTECT", "MEV_SHARE_BUNDLE")

        # 2. Build Flashbots Bundle
        bundle_req = {
            "raw_signed_txs": ["0x0123456789abcdef", "0xfedcba9876543210"],
            "current_block_number": 19500000,
            "block_lead": 1,
        }
        res_bundle = await client.post("/compliance/mev/bundle", json=bundle_req)
        assert res_bundle.status_code == 200
        bundle_data = res_bundle.json()
        assert bundle_data["target_block_number"] == 19500001
        assert bundle_data["transactions_count"] == 2
        assert "bundle_id" in bundle_data


@pytest.mark.asyncio
async def test_api_lead_lag_network_endpoints() -> None:
    """Verify REST endpoints for micro-structural lead-lag network calibration and signal generation."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        # 1. Ingest Tick
        tick_req = {"symbol": "BTC-USD", "price": 65200.0}
        res_tick = await client.post("/analysis/lead-lag/ticks/ingest", json=tick_req)
        assert res_tick.status_code == 200
        assert res_tick.json()["status"] == "INGESTED"
        assert res_tick.json()["symbol"] == "BTC-USD"

        # 2. Batch Ingest Matrix
        matrix_req = {
            "prices": {
                "BTC-USD": [65000.0, 65050.0, 65120.0, 65200.0],
                "ETH-USD": [3200.0, 3202.0, 3205.0, 3210.0],
                "SOL-USD": [150.0, 150.1, 150.3, 150.5],
            }
        }
        res_mat = await client.post("/analysis/lead-lag/matrix/ingest", json=matrix_req)
        assert res_mat.status_code == 200
        assert res_mat.json()["status"] == "BATCH_INGESTED"

        # 3. Retrieve Lead-Lag Network Report
        res_net = await client.get("/analysis/lead-lag/network")
        assert res_net.status_code == 200
        net_data = res_net.json()
        assert "symbols" in net_data
        assert "profiles" in net_data
        assert "pairwise_links" in net_data
        assert "generated_signals" in net_data
        assert "matrix_spectral_gap" in net_data
        assert len(net_data["profiles"]) == 3


@pytest.mark.asyncio
async def test_api_cryptographic_trade_vault_endpoints() -> None:
    """Verify REST endpoints for Merkle trade proofs, settlement attestation, and ZK solvency."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        # 1. Record Fill in Vault
        fill_req = {
            "trade_id": "API_VAULT_T1",
            "order_id": "API_ORD_01",
            "symbol": "BTC-USD",
            "side": "BUY",
            "price": 65000.0,
            "quantity": 2.0,
            "venue": "BINANCE",
            "fee_usd": 1.5,
        }
        res_fill = await client.post("/compliance/vault/fills/record", json=fill_req)
        assert res_fill.status_code == 200
        receipt = res_fill.json()
        assert receipt["trade_id"] == "API_VAULT_T1"
        assert len(receipt["leaf_hash"]) == 64

        # 2. Get Inclusion Proof
        res_proof = await client.get(f"/compliance/vault/proof/inclusion/{receipt['trade_id']}")
        assert res_proof.status_code == 200
        proof_data = res_proof.json()
        assert proof_data["target_hash"] == receipt["leaf_hash"]
        assert proof_data["is_valid"] is True

        # 3. Verify Inclusion Proof
        res_verify = await client.post("/compliance/vault/proof/verify", json=proof_data)
        assert res_verify.status_code == 200
        assert res_verify.json()["is_valid"] is True

        # 4. Get Settlement Attestation
        res_attest = await client.get("/compliance/vault/settlement/attestation")
        assert res_attest.status_code == 200
        attest_data = res_attest.json()
        assert attest_data["trade_count"] >= 4  # 3 pre-seeded + 1 new
        assert len(attest_data["merkle_root"]) == 64
        assert len(attest_data["attestation_signature_hash"]) == 64

        # 5. Check Solvency Commitment
        solvency_req = {
            "epoch_id": "EPOCH_API_TEST",
            "asset_balances": {"USD": 600000.0, "BTC": 10.0},
            "liability_balances": {"USD": 200000.0, "BTC": 5.0},
        }
        res_solvency = await client.post("/compliance/vault/solvency/commitment", json=solvency_req)
        assert res_solvency.status_code == 200
        solv_data = res_solvency.json()
        assert solv_data["is_solvent"] is True
        assert solv_data["vault_epoch_id"] == "EPOCH_API_TEST"
        assert len(solv_data["root_commitment"]) == 64


@pytest.mark.asyncio
async def test_api_adaptive_market_maker_endpoints() -> None:
    """Verify REST endpoints for adaptive Avellaneda-Stoikov quoting and fill execution."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        # 1. Record Fill
        fill_req = {"symbol": "BTC-USD", "side": "BUY", "quantity": 1.5, "price": 65000.0}
        res_fill = await client.post("/execution/adaptive-mm/fill", json=fill_req)
        assert res_fill.status_code == 200
        assert res_fill.json()["status"] == "FILL_RECORDED"
        assert res_fill.json()["symbol"] == "BTC-USD"

        # 2. Check Inventory
        res_inv = await client.get("/execution/adaptive-mm/BTC-USD/inventory")
        assert res_inv.status_code == 200
        inv_data = res_inv.json()
        assert inv_data["symbol"] == "BTC-USD"
        assert inv_data["inventory"] == 2.0  # 0.5 pre-seeded + 1.5
        assert inv_data["utilization_pct"] > 0.0

        # 3. Calculate Adaptive Quotes
        quote_req = {
            "symbol": "BTC-USD",
            "best_bid": 65000.0,
            "best_ask": 65010.0,
            "volatility_annualized": 0.55,
            "drift_annualized": 0.2,
            "has_volatility_jump": False,
        }
        res_q = await client.post("/execution/adaptive-mm/quotes", json=quote_req)
        assert res_q.status_code == 200
        q_data = res_q.json()
        assert q_data["symbol"] == "BTC-USD"
        assert q_data["mid_price"] == 65005.0
        assert "reservation_price" in q_data
        assert len(q_data["quoting_tiers"]) == 5
        assert q_data["quoting_tiers"][0]["bid_fill_probability"] > 0.0


@pytest.mark.asyncio
async def test_api_quantum_rebalancing_endpoints() -> None:
    """Verify REST endpoints for quantum-inspired cardinality-constrained portfolio rebalancing."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        req = {
            "symbols": ["BTC", "ETH", "SOL", "AVAX", "LINK"],
            "expected_returns": {"BTC": 0.25, "ETH": 0.30, "SOL": 0.45, "AVAX": 0.35, "LINK": 0.20},
            "covariance_matrix": [
                [0.09, 0.04, 0.03, 0.02, 0.02],
                [0.04, 0.12, 0.05, 0.03, 0.03],
                [0.03, 0.05, 0.20, 0.06, 0.04],
                [0.02, 0.03, 0.06, 0.18, 0.04],
                [0.02, 0.03, 0.04, 0.04, 0.10],
            ],
            "max_cardinality": 3,
            "risk_aversion_lambda": 2.0,
            "transaction_cost_bps": 12.0,
            "solver_method": "SIMULATED_ANNEALING_QUBO",
        }
        res = await client.post("/risk/rebalance/quantum-optimize", json=req)
        assert res.status_code == 200
        data = res.json()
        assert data["total_assets_universe"] == 5
        assert data["selected_cardinality"] <= 3
        assert data["max_cardinality_limit"] == 3
        assert data["portfolio_volatility"] > 0.0
        assert len(data["allocations"]) == 5
        assert sum(a["target_weight"] for a in data["allocations"] if a["is_active"]) == pytest.approx(1.0, abs=1e-3)


@pytest.mark.asyncio
async def test_api_multi_agent_lob_endpoints() -> None:
    """Verify REST endpoints for multi-agent competitive equilibrium LOB simulation."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        req = {
            "symbol": "BTC-USD",
            "initial_mid_price": 65000.0,
            "steps": 25,
            "as_mm_count": 2,
            "grid_mm_count": 1,
            "toxic_taker_count": 1,
            "noise_taker_count": 1,
            "seed": 42,
        }
        res = await client.post("/backtest/sandbox/multi-agent/simulate", json=req)
        assert res.status_code == 200
        sim_data = res.json()
        assert sim_data["symbol"] == "BTC-USD"
        assert sim_data["total_steps"] == 25
        assert len(sim_data["agent_profiles"]) == 5  # 2 + 1 + 1 + 1
        assert "lob_metrics" in sim_data
        assert sim_data["lob_metrics"]["total_events_processed"] > 0
        assert sim_data["lob_metrics"]["average_spread_bps"] > 0.0
        assert "top_performer_agent_id" in sim_data


@pytest.mark.asyncio
async def test_api_cross_chain_monitor_endpoints() -> None:
    """Verify REST endpoints for cross-chain finality tracking, reorg safety, and bridge arbitrage."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        # 1. Update Block Height
        height_req = {
            "chain": "ETHEREUM",
            "current_block": 19500100,
            "finalized_block": 19500050,
        }
        res_h = await client.post("/compliance/cross-chain/height/update", json=height_req)
        assert res_h.status_code == 200
        h_data = res_h.json()
        assert h_data["chain"] == "ETHEREUM"
        assert h_data["unfinalized_blocks"] == 50
        assert h_data["is_reorg_safe"] is True

        # 2. Evaluate Route Quote
        quote_req = {
            "source_chain": "ARBITRUM",
            "destination_chain": "ETHEREUM",
            "token_symbol": "ETH",
            "bridge_protocol": "ACROSS",
            "trade_amount": 10.0,
        }
        res_q = await client.post("/compliance/cross-chain/quote", json=quote_req)
        assert res_q.status_code == 200
        q_data = res_q.json()
        assert q_data["source_chain"] == "ARBITRUM"
        assert q_data["destination_chain"] == "ETHEREUM"
        assert "net_arbitrage_profit_usd" in q_data
        assert "expected_latency_seconds" in q_data

        # 3. Retrieve Full Monitor Report
        res_rep = await client.get("/compliance/cross-chain/report")
        assert res_rep.status_code == 200
        rep_data = res_rep.json()
        assert len(rep_data["chains"]) == 6
        assert "arbitrage_opportunities" in rep_data





























