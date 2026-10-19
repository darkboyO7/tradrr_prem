# Institutional Architectural Specification & Quantitative Foundations

This document details the architectural principles, mathematical formulations, latency profiles, and institutional governance guarantees implemented in the **Antigravity Trading Intelligence Platform (TIP)**.

---

## 1. End-to-End System Architecture

```mermaid
flowchart TD
    subgraph Market Ingestion & Microstructure
        C_WS["Coinbase WS"] --> NORM["Event Normalizer"]
        K_WS["Kraken WS"] --> NORM
        B_WS["Binance REST/WS"] --> NORM
        NORM --> VAL["Data Validator (Zero-Spike/Timestamps)"]
        VAL --> COB["Consolidated Order Book (Synthetic BBO)"]
        VAL --> CE["Candle Engine & MTF Resampler"]
    end

    subgraph Intelligence & Regimes
        CE --> FEAT["Feature Pipeline (60+ Microstructure & Indicators)"]
        FEAT --> REG["3D Market Regime Detector"]
        FEAT --> PAT["Candle Pattern & Cosine Similarity"]
        FEAT --> ML["LightGBM ML Classifier"]
        REG --> FUS["Signal Fusion Engine (ATR Envelopes)"]
        PAT --> FUS
        ML --> FUS
    end

    subgraph Risk & Cost Governance
        FUS --> RISK["Pre-Trade Risk Engine (Kelly / Circuit Breakers)"]
        RISK --> VAR["Portfolio VaR & Factor Attribution"]
        RISK --> STRESS["Portfolio Stress Testing Engine"]
        RISK --> COST["Almgren-Chriss Market Impact Model"]
    end

    subgraph Execution & Algorithms
        COST --> SOR["Smart Order Router"]
        COST --> ALGO["Algorithmic Execution (TWAP / VWAP)"]
        SOR --> EX_SIM["Simulated / Paper Execution"]
        ALGO --> EX_SIM
    end

    subgraph Compliance, HA & Recovery
        EX_SIM --> AUDIT["Immutable Audit Ledger (SHA-256 Chained)"]
        EX_SIM --> SURV["Regulatory Surveillance (Wash/Spoof)"]
        EX_SIM --> SNAP["Snapshot Manager (Point-in-Time State)"]
        SNAP --> HA["Failover Orchestrator (Epoch Fencing)"]
    end
```

---

## 2. Zero-Lookahead Bias Guarantee

In quantitative systems, lookahead bias occurs when future data is accidentally utilized to generate a signal, train a model, or normalize indicators for past bars.

The platform enforces strict point-in-time causality across all layers:

1. **Feature Computation**:
   - All indicators are computed strictly backward-looking using rolling windows with closed left boundaries.
   - For multi-timeframe features (e.g. merging 1h features onto 1m candles), the timestamp of the higher timeframe bar corresponds to its **completion time**, never its open or duration interval.
   ```python
   # Multi-timeframe point-in-time backward alignment
   merged_df = pd.merge_asof(
       lower_df.sort_values("timestamp"),
       higher_df.sort_values("timestamp"),
       on="timestamp",
       direction="backward"
   )
   ```

2. **Purged Walk-Forward Cross-Validation**:
   - Training windows precede testing windows chronologically.
   - An **embargo buffer** (default: 5 bars) is enforced between the end of the training window and the start of the validation window to purge serial autocorrelation and forward information leakage.

---

## 3. 3D Market Regime Engine

Market dynamics are modeled not as a single scalar, but as an orthogonal 3-dimensional state space:

$$S_{\text{regime}} = (R_{\text{trend}}, R_{\text{volatility}}, R_{\text{liquidity}})$$

### Dimension 1: Trend Regime
Evaluates directional persistence using the Exponential Moving Average (EMA) stack and the Average Directional Index (ADX):
- **`STRONG_BULL`**: $\text{Close} > \text{EMA}_{20} > \text{EMA}_{50} \quad\text{and}\quad \text{ADX}_{14} \ge 25 \quad\text{and}\quad +\text{DI} > -\text{DI}$
- **`WEAK_BULL`**: $\text{Close} > \text{EMA}_{20} \quad\text{and}\quad \text{ADX}_{14} < 25$
- **`RANGING`**: $\text{ADX}_{14} < 20 \quad\text{or}\quad \text{Close} \approx \text{EMA}_{20}$
- **`WEAK_BEAR`**: $\text{Close} < \text{EMA}_{20} \quad\text{and}\quad \text{ADX}_{14} < 25$
- **`STRONG_BEAR`**: $\text{Close} < \text{EMA}_{20} < \text{EMA}_{50} \quad\text{and}\quad \text{ADX}_{14} \ge 25 \quad\text{and}\quad -\text{DI} > +\text{DI}$

### Dimension 2: Volatility Regime
Evaluates volatility compression vs expansion relative to baseline:
- **`COMPRESSED`**: Bollinger Band Squeeze ($\text{BB Width} \le \text{Keltner Width}$) or $\frac{\sigma_{\text{rolling}}}{\sigma_{\text{baseline}}} < 0.75$
- **`NORMAL`**: $0.75 \le \frac{\sigma_{\text{rolling}}}{\sigma_{\text{baseline}}} < 1.30$
- **`ELEVATED`**: $1.30 \le \frac{\sigma_{\text{rolling}}}{\sigma_{\text{baseline}}} < 2.00$
- **`EXTREME`**: $\frac{\sigma_{\text{rolling}}}{\sigma_{\text{baseline}}} \ge 2.00$

### Dimension 3: Liquidity / Volume Regime
Evaluates volume participation via 20-period Relative Volume ($\text{RVOL}_{20}$):
- **`LOW_VOLUME`**: $\text{RVOL}_{20} < 0.70$
- **`NORMAL_VOLUME`**: $0.70 \le \text{RVOL}_{20} < 1.40$
- **`VOLUME_EXPANSION`**: $1.40 \le \text{RVOL}_{20} < 2.50$
- **`VOLUME_CLIMAX`**: $\text{RVOL}_{20} \ge 2.50$

---

## 4. Pre-Trade Risk Engine & Mathematical Position Sizing

The Risk Engine serves as an immutable gatekeeper. No order can be submitted without passing through its validation pipeline.

### Mathematical Sizing Formulas:

1. **Fixed Fractional ATR Risk Sizing**:
   The capital at risk per trade is strictly capped at a fraction $f_{\text{risk}}$ (typically $1\%$ to $2\%$) of total portfolio equity $E$:
   $$\text{Dollar Risk} = E \times f_{\text{risk}}$$
   Given an entry price $P_{\text{entry}}$ and Stop-Loss distance $\Delta_{\text{SL}} = |P_{\text{entry}} - P_{\text{SL}}| = k_{\text{ATR}} \times \text{ATR}_{14}$:
   $$\text{Quantity} = \frac{E \times f_{\text{risk}}}{\Delta_{\text{SL}}}$$

2. **Kelly Criterion with Fractional Damping**:
   For probabilistic setups with estimated win rate $W$ and win-loss payoff ratio $R$:
   $$f^* = \frac{W \cdot (R + 1) - 1}{R}$$
   To prevent aggressive drawdowns from parameter uncertainty, the system applies **Quarter-Kelly or Half-Kelly damping**:
   $$f_{\text{allocated}} = \max\left(0, \min\left(0.5 \times f^*, f_{\text{max\_trade}}\right)\right)$$

### Circuit Breakers:
- **Peak-to-Trough Drawdown**:
  $$\text{DD}_{\text{peak}} = \frac{E_{\text{peak}} - E}{E_{\text{peak}}} \ge 10.0\% \implies \text{HALT ALL TRADING}$$
- **Intraday Session Loss**:
  $$\text{DD}_{\text{session}} = \frac{E_{\text{session\_start}} - E}{E_{\text{session\_start}}} \ge 3.0\% \implies \text{HALT DAILY TRADING}$$

---

## 5. Controlled Self-Learning & Model Governance

The self-learning coordinator continuously evaluates live inference accuracy against actual bar realizations using Brier scores and Sharpe metrics:

1. **Outcome Tracking**: Every signal records its predicted direction, calibrated probability, and ATR boundary expectations.
2. **Degradation Detection**: If rolling 30-day Brier score exceeds 0.25 or rolling Sharpe drops below 0.80, a retraining cycle is automatically initiated.
3. **Walk-Forward Promotion Gate**: The candidate model trained on historical data must satisfy:
   - Out-of-sample Sharpe Ratio $\ge 1.0$
   - Out-of-sample Win Rate $\ge 50.0\%$
   - Out-of-sample Max Drawdown $\le 15.0\%$
   - Sharpe improvement $\ge +0.10$ over current production champion.

---

## 6. Multi-Venue Order Book, Synthetic BBO & Smart Order Routing

The `ConsolidatedOrderBook` aggregates Level 2 order books across Coinbase, Kraken, and Binance:
- **Synthetic BBO**: Evaluates $\max(\text{Bid}_v)$ and $\min(\text{Ask}_v)$ across all venues $v \in V$.
- **Cross-Exchange Spatial Arbitrage**: Detects when $\text{Bid}_{v_1} > \text{Ask}_{v_2}$ net of taker fees ($f_{v_1} + f_{v_2}$) and minimum executable slice sizes.
- **Smart Order Router (SOR)**: Slices large parent orders across venues based on liquidity depth, fees, and execution policies (`BEST_PRICE`, `MIN_FEE`, `DEPTH_WEIGHTED`).

---

## 7. Almgren-Chriss Market Impact & Algorithmic Slicing

Implements the institutional Almgren-Chriss framework to calculate implementation shortfall:

$$\text{Total Cost} = \text{Half-Spread Drag} + \text{Temporary Impact} + \text{Permanent Impact} + \text{Taker Fees}$$

- **Temporary Impact (Square-Root Law)**:
  $$\Delta P_{\text{temp}} = \eta \cdot \sigma \cdot \sqrt{\frac{Q}{V_{\text{ADV}}}}$$
- **Permanent Impact**:
  $$\Delta P_{\text{perm}} = \gamma \cdot \sigma \cdot \left(\frac{Q}{V_{\text{ADV}}}\right)$$
- **Algorithmic Execution (TWAP & VWAP)**:
  - Randomized time jitter ($\pm 15\%$) and size jitter ($\pm 10\%$) prevent predatory front-running by high-frequency market participants.
  - VWAP orders weight slices according to the 24-hour empirical volume profile.

---

## 8. Portfolio Risk Analytics & Macro Stress Testing

- **Value-at-Risk (VaR)**: Parametric (Variance-Covariance), Historical Simulation, and Monte Carlo (10,000 paths).
- **Euler Component VaR**: Additive risk decomposition allocating exact percentage risk contributions to each asset.
- **Systematic Factor Attribution**: Deconstructs portfolio returns into Market ($\beta$), Momentum, Volatility, and Size factors.
- **Macro Stress Scenarios**: Evaluates portfolio resilience under historical shocks (e.g. COVID Liquidity Crunch, Crypto De-Pegging, Flash Crash).

---

## 9. Regulatory Compliance, MiFID II Audit Trail & Trade Surveillance

- **Cryptographic Chained Audit Ledger**: Every submission, fill, cancellation, and risk override is persisted into a SHA-256 hash chain with strict sequence numbering ($H_i = \text{SHA256}(H_{i-1} \parallel D_i)$).
- **Regulatory Reporting**: Exports transaction records formatted in conformance with MiFID II RTS 22/25 standards.
- **Trade Surveillance**: Real-time detectors flag abusive patterns:
  - Wash trading (simultaneous opposing orders within tight price/time windows).
  - Spoofing / layering (excessive order-to-cancellation ratio $> 5.0$).
  - Quote stuffing (burst rate exceeding 50 msgs/sec).

---

## 10. Disaster Recovery, High Availability & Deterministic Replay

- **Cold-Start Snapshots**: Serializes full portfolio equity, cash, open positions, and realized PnL with canonical SHA-256 checksums. Rehydration verifies checksums before updating engine memory.
- **High-Availability Failover**: Lease-based heartbeats between primary and standby nodes. Automated failover triggers promotion upon primary timeout while epoch numbers prevent split-brain dual leaders.
- **Microstructure Stress Harness & Replay**: Generates 10,000+ synthetic tick storms to benchmark order book throughput ($> 5,000\text{ eps}$) and microsecond latency ($p50 < 2\text{ms}$). The deterministic replay engine enforces 0 bps mathematical drift between backtesting and paper simulation.
