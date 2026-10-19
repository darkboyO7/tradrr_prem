# Antigravity Trading Intelligence Platform (TIP)

An institutional, modular quantitative trading intelligence platform built for professional quantitative research, high-frequency/low-latency real-time market state analysis, controlled self-learning, multi-venue execution, and rigid regulatory governance.

---

## Architecture Overview

```
 MARKET SOURCES (Coinbase WS / Kraken WS / Binance REST/WS)
       │
       ▼
 DATA ACQUISITION & DATA QUALITY ENGINE (Spike checks, Sequence, Timestamps)
       │
       ├───► HISTORICAL STORE (Parquet with SHA-256 Lineage & PostgreSQL / TimescaleDB)
       ├───► CONSOLIDATED ORDER BOOK & SYNTHETIC BBO (Coinbase, Kraken, Binance)
       └───► MULTI-TIMEFRAME RESAMPLER (1m, 5m, 15m, 1h, 4h, 1d)
                 │
                 ▼
          FEATURE ENGINE (60+ Price, Trend, Momentum, Volatility, Volume, Structure)
                 │
                 ├───► 3D MARKET REGIME ENGINE (Trend, Volatility, Liquidity)
                 ├───► HISTORICAL PATTERNS & VECTOR SIMILARITY (Cosine nearest neighbors)
                 └───► STATISTICAL & ML MODELS (LightGBM Directional Classifier)
                           │
                           ▼
                    SIGNAL FUSION ENGINE (Regime-Conditioned Multi-Source Evidence)
                           │
                           ▼
                 PRE-TRADE RISK ENGINE (Kelly / Fractional Sizing & Circuit Breakers)
                           │
                           ├───► ALMGREN-CHRISS COST MODEL & ALGO ENGINE (TWAP / VWAP)
                           ├───► SMART ORDER ROUTER & ARBITRAGE SCANNER
                           └───► PAPER TRADING SIMULATOR (Slippage, Fees, SL/TP Monitoring)
                                     │
                                     ├───► CRYPTOGRAPHIC IMMUTABLE AUDIT LEDGER (SHA-256 Chained)
                                     ├───► REGULATORY TRADE SURVEILLANCE (Wash / Spoof / Quote Stuffing)
                                     ├───► COLD-START DISASTER RECOVERY (Tamper-Evident Snapshots)
                                     └───► HIGH-AVAILABILITY FAILOVER ORCHESTRATOR (Epoch Fencing)
```

---

## Core Institutional Capabilities

1. **Provider-Neutral Market Data Ingestion**:
   - High-throughput WebSocket clients for **Coinbase Exchange**, **Kraken**, and **Binance**.
   - Built-in ping/pong heartbeat, automatic exponential-backoff reconnects, and configurable stale-data watchdogs.
   - P50, P95, and P99 latency instrumentation.

2. **Strict Data Quality Engine**:
   - Microsecond UTC timestamp validation, price spike thresholds, negative price guards, and missing candle interpolation.
   - Maintains immutable raw payloads alongside normalized canonical trade/quote events.

3. **Multi-Timeframe Feature Pipeline**:
   - Zero-lookahead feature engineering calculating 60+ quantitative indicators:
     - **Price**: Log returns, rolling volatility, range, body-to-wick ratios.
     - **Trend**: Moving average stacks (EMA 9/20/50/100/200), slope, ADX, +DI/-DI.
     - **Momentum**: RSI, MACD, Stochastic Oscillator, Rate of Change (ROC).
     - **Volatility**: ATR, Bollinger Bands, BB Squeeze ratio, Historical Volatility.
     - **Volume**: Relative Volume (RVOL), Volume Moving Averages, Spike flags.
     - **Market Structure**: Swing highs/lows, breakout/breakdown indicators, multi-timeframe alignment.

4. **3D Market Regime Classification**:
   - Orthogonal environment classification across 3 dimensions:
     - **Trend**: `STRONG_BULL`, `WEAK_BULL`, `RANGING`, `WEAK_BEAR`, `STRONG_BEAR`.
     - **Volatility**: `COMPRESSED`, `NORMAL`, `ELEVATED`, `EXTREME`.
     - **Liquidity**: `LOW_VOLUME`, `NORMAL_VOLUME`, `VOLUME_EXPANSION`, `VOLUME_CLIMAX`.

5. **Multi-Source Signal Fusion Engine**:
   - Synthesizes rule-based technical setups, candlestick patterns, and Machine Learning probabilities into deterministic `BUY`, `SELL`, or `NO-TRADE` decisions.
   - Computes dynamic ATR-based Stop-Loss (SL), Take-Profit (TP), and Risk-to-Reward (R:R) targets with explainable trade rationales.

6. **Pre-Trade Risk Engine & Dynamic Sizing**:
   - Fractional Kelly and Fixed Fractional position sizing algorithms.
   - Portfolio exposure caps, concurrent position limits, and peak-to-trough drawdown circuit breakers.

7. **Multi-Venue Consolidated Order Book & Smart Order Routing**:
   - Aggregated L2 limit order books computing Synthetic National Best Bid and Offer (BBO).
   - Real-time Cross-Exchange Spatial Arbitrage detection factoring in taker fees.
   - Smart Order Router (SOR) executing depth-weighted and fee-optimized child orders.

8. **Institutional Execution Cost Modeling & Algorithmic Slicing**:
   - Almgren-Chriss market impact engine evaluating temporary and permanent price impact.
   - Randomized TWAP schedule with time ($\pm 15\%$) and size ($\pm 10\%$) jitter.
   - Empirical 24-hour volume profile for volume-weighted VWAP execution.

9. **Portfolio Risk Analytics & Macro Stress Testing**:
   - Multi-asset Value-at-Risk (Parametric, Historical, Monte Carlo) and Euler Component VaR.
   - Systematic Factor Attribution (Market, Momentum, Volatility, Size).
   - Macro Stress Testing simulating severe liquidity crises, de-pegging, and crashes.

10. **Regulatory Compliance & Trade Surveillance**:
    - Cryptographic SHA-256 chained immutable audit ledger conforming to MiFID II RTS 22/25.
    - Automated surveillance flags for wash trading, spoofing/layering, and quote stuffing.

11. **Disaster Recovery & High Availability**:
    - Cold-start point-in-time state serialization with SHA-256 tamper detection.
    - High-availability failover orchestrator with lease-based heartbeats and epoch fencing to eliminate split-brain dual leaders.

12. **Microstructure Stress Harness & Parity Replay**:
    - High-frequency tick storm generator producing 10,000+ ticks/sec to profile throughput and microsecond latency quantiles.
    - Deterministic replay framework guaranteeing 0 bps mathematical drift between backtest and paper simulation.

---

## Quickstart Guide

### 1. Environment Setup
```powershell
# Create and activate Python virtual environment
python -m venv .venv
.venv\Scripts\activate

# Install dependencies in editable mode
pip install -e .[dev,ml]
```

### 2. Run Test Suite
```powershell
# Execute complete unit, integration, and E2E test suite (177 passing tests)
pytest -q
```

### 3. Master CLI Controller
The unified command-line tool `scripts/cli.py` exposes all system functions:

```powershell
# 1. System Health Diagnostics
python scripts/cli.py status

# 2. Event-Driven Backtesting
python scripts/cli.py backtest --symbol BTC-USD --candles 250 --initial-capital 100000

# 3. Live Paper Trading Simulation
python scripts/cli.py paper --symbol BTC-USD --candles 100 --delay 0.05

# 4. Strict Model Governance Audit
python scripts/cli.py govern --symbol BTC-USD --candles 200

# 5. Autonomous Continuous Learning Loop
python scripts/cli.py learn --symbol BTC-USD --candles 200 --force

# 6. Start REST API & Visual Command Dashboard
python scripts/cli.py serve --host 0.0.0.0 --port 8000
```

---

## REST API & WebSocket Endpoints

| Endpoint | Method | Description |
| :--- | :--- | :--- |
| `/health` | `GET` | Subsystem telemetry, database connection, cache status |
| `/metrics` | `GET` | Real-time platform throughput, latency, and drawdown metrics |
| `/prometheus` | `GET` | Prometheus exposition format metrics |
| `/dashboard` | `GET` | Interactive command center UI with signal testing & risk gauges |
| `/candles/{symbol}` | `GET` | OHLCV historical candle series with configurable timeframes |
| `/regime/{symbol}` | `GET` | Active 3D market regime classification |
| `/signals/evaluate` | `POST` | Multi-source regime-conditioned signal fusion |
| `/risk/evaluate` | `POST` | Pre-trade risk evaluation, ATR sizing, and circuit breakers |
| `/portfolio/state` | `GET` | Mark-to-market portfolio equity, cash, and open positions |
| `/orderbook/{symbol}/consolidated` | `GET` | Multi-venue aggregated order book and Synthetic BBO |
| `/arbitrage/scan` | `GET` | Active cross-exchange spatial arbitrage opportunities |
| `/execution/smart-route` | `POST` | Plan and execute multi-venue child order slices |
| `/execution/algo/profile/{symbol}`| `GET` | 24-hour empirical intraday volume profile |
| `/execution/algo/schedule` | `POST` | Schedule randomized TWAP / VWAP execution slices |
| `/execution/algo/execute` | `POST` | Execute algorithmic parent orders and measure shortfall |
| `/compliance/audit/verify` | `GET` | Verify cryptographic SHA-256 hash-chain integrity |
| `/compliance/audit/record` | `POST` | Append an auditable event to the immutable ledger |
| `/compliance/export/mifid` | `GET` | Export regulatory transaction records conforming to MiFID II |
| `/compliance/surveillance/scan` | `POST` | Scan orders for wash trading, spoofing, and quote stuffing |
| `/recovery/snapshot/create` | `POST` | Serialize point-in-time engine state with SHA-256 checksum |
| `/recovery/snapshot/list` | `GET` | Enumerate available disaster recovery state snapshots |
| `/recovery/snapshot/restore` | `POST` | Validate integrity and rehydrate state into memory |
| `/recovery/ha/status` | `GET` | High-availability cluster telemetry, active role, and heartbeats |
| `/system/benchmark/stress` | `POST` | Execute on-demand 10k-tick orderbook stress benchmark |
| `/ws/stream` | `WS` | Bi-directional streaming gateway for ticks, candles, and signals |

---

## Docker Compose Infrastructure

Deploy the complete containerized stack (API, Prometheus, TimescaleDB, Redis, MLflow):

```powershell
docker-compose up -d --build
```

- **Interactive Command Center**: [http://localhost:8000/dashboard](http://localhost:8000/dashboard)
- **FastAPI Interactive Docs**: [http://localhost:8000/docs](http://localhost:8000/docs)
- **Prometheus Metrics**: [http://localhost:8000/prometheus](http://localhost:8000/prometheus)
- **Prometheus Web UI**: [http://localhost:9090](http://localhost:9090)
- **MLflow Experiment Tracking**: [http://localhost:5000](http://localhost:5000)

---

## Technology Stack

| Layer | Technologies |
| :--- | :--- |
| **Language** | Python 3.12+ (operating on Python 3.14.7) |
| **API & Web** | FastAPI, Uvicorn, WebSockets, HTTPX, TailwindCSS |
| **Data Processing** | NumPy, Pandas 2.2+, PyArrow, Parquet |
| **Database & Cache** | PostgreSQL / TimescaleDB, SQLAlchemy 2.0 (async), Redis |
| **Quantitative ML** | Scikit-learn, LightGBM, MLflow |
| **Testing & Quality**| Pytest, Pytest-Asyncio, Ruff, Mypy |
| **Observability** | Prometheus, Loguru Structured JSON Logging |
| **Containers** | Docker, Docker Compose |
