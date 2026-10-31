# Forex-Specific Modeling Specification (Version 1.0.0)

**Document ID:** `SPEC-FX-MODEL-2026-001`  
**Target Instruments:** `EURUSD`, `EURCHF`  
**Governance Policy:** `POL-GOV-2026-001`  
**Status:** DRAFT / DESIGN SPECIFICATION ONLY (TRAINING PROHIBITED IN CURRENT PHASE)

---

## 1. Target Instruments and Native Intervals
- **Eligible Currency Pairs:** `EURUSD` and `EURCHF` (Interbank spot market quotes).
- **Permanently Blocked Currency Pairs:** `EURAUD` and `EURCAD`.
- **Native Ingestion Interval:** $5$-minute completed OHLC bars ($5\text{m}$).
- **Prohibited Data Inputs:** Indicative daily central bank reference fixing rates (ECB/Frankfurter) cannot be used as training inputs or inference inputs for intraday execution.

---

## 2. Forecast Horizon and Target Labels
- **Forecast Horizon ($h$):** $15\text{m}$ (3 bars) and $1\text{h}$ (12 bars).
- **Execution Timing & Reference Prices:**
  Prediction is generated at bar $t$ close (completed bar).
  Executable entry fills at bar $t+1$ open ($P_{t+1}^{\text{open}}$).
  Executable exit fills at bar $t+1+h$ open ($P_{t+1+h}^{\text{open}}$).
  $$R_{t \to t+h} = \frac{P_{t+1+h}^{\text{open}} - P_{t+1}^{\text{open}}}{P_{t+1}^{\text{open}}}$$
  
  The three-class target direction $y_t \in \{0, 1, 2\}$ is defined as:
  $$y_t = \begin{cases} 
  2 \text{ (BUY)}, & \text{if } R_{t \to t+h} \ge +\theta_t \\
  0 \text{ (SELL)}, & \text{if } R_{t \to t+h} \le -\theta_t \\
  1 \text{ (NO\_TRADE / NEUTRAL)}, & \text{if } -\theta_t < R_{t \to t+h} < +\theta_t 
  \end{cases}$$
  where the hurdle $\theta_t$ accounts for round-trip friction plus an explicit alpha safety margin:
  $$\theta_t = \frac{\text{Spread} + 2 \times \text{Slippage}}{P_{t+1}^{\text{open}}} + 2 \times \text{Commission Rate} + \text{Margin Rate}$$
  * For EURUSD at $1.1200$: Spread $= 1.5\text{ pips}$ ($0.00015$), Slippage $= 0.20\text{ pip}$ ($0.000020$), Commission $= 0.30\text{ bps}$, Margin $= 1.0\text{ bps}$ $\implies \theta \approx 3.30\text{ bps}$ ($0.000330$).
  * For EURCHF at $0.9400$: Spread $= 2.0\text{ pips}$ ($0.00020$), Slippage $= 0.25\text{ pip}$ ($0.000025$), Commission $= 0.30\text{ bps}$, Margin $= 1.0\text{ bps}$ $\implies \theta \approx 4.26\text{ bps}$ ($0.000426$).

---

## 3. Scale-Invariant Feature Definitions (Version: `FX-FEAT-v1.0.0`)
Features must be strictly scale-invariant and session-gap-aware, computable on completed candles:
1. **Normalized Returns (Session-Gap Aware):**
   - $R_1 = \ln(C_t / C_{t-1})$ within sessions; intra-bar $\ln(C_t / O_t)$ across gaps ($> 300\text{s}$).
   - $R_5 = \ln(C_t / C_{t-5})$ partitioned strictly within contiguous sessions.
   - $R_{12} = \ln(C_t / C_{t-12})$ partitioned strictly within contiguous sessions.
2. **Normalized Volatility (Gap-Protected True Range):**
   - True Range across gaps uses intra-bar $H_t - L_t$, eliminating weekend closure jump corruption.
   - Normalized ATR: $\text{NATR}_{14} = \frac{\text{ATR}_{14}(t)}{C_t}$
   - Normalized Candle Range: $\frac{H_t - L_t}{C_t}$
   - Candle Body Ratio: $\frac{|C_t - O_t|}{H_t - L_t + \epsilon}$
3. **Normalized Momentum & Oscillators:**
   - $\text{RSI}_{14}$ (0 to 100) using gap-filtered returns.
   - Normalized MACD Histogram: $\frac{\text{MACD\_Hist}_t}{C_t}$
   - $\text{ADX}_{14}$, $+DI_{14}$, $-DI_{14}$ with gap protection on directional movement.
   - Bollinger Bands $\%B_{20, 2} = \frac{C_t - \text{Lower}_t}{\text{Upper}_t - \text{Lower}_t}$
4. **Volume Features (Where Genuinely Available):**
   - Relative Volume: $\text{RVOL}_{20} = \frac{V_t}{\text{SMA}(V, 20) + \epsilon}$
   - Volume Dist from VWAP: $\frac{C_t - \text{VWAP}_t}{C_t}$
5. **Strictly Prohibited Features:**
   - Absolute price levels ($C_t$, $O_t$, $H_t$, $L_t$).
   - Absolute Moving Averages ($\text{EMA}_9$, $\text{EMA}_{20}$, $\text{EMA}_{50}$ in nominal currency units).

---

## 4. Historical Dataset Coverage and Sample Requirements
- **Minimum Historical Depth:** 12 consecutive months of verified 5m candles ($\approx 70,000$ observations per pair).
- **Session Filtering:** Active 24/5 global sessions (Sunday 17:00 NY to Friday 17:00 NY).
- **Gap Handling:** Missing intervals must not be interpolated or forward-filled. Gaps $\ge 15\text{m}$ reset rolling indicator windows.

---

## 5. Chronological Train, Validation, and Test Partitioning
- **Strictly Chronological:** Zero random K-Fold splitting.
  - **Training Set (60%):** Oldest contiguous block.
  - **Validation & Calibration Set (20%):** Intermediate contiguous block.
  - **Holdout Test Set (20%):** Genuine out-of-sample final contiguous block.
- **Purging and Embargo:** An embargo period equal to $2 \times h$ bars is placed between splits to completely prevent label overlap leakage.

---

## 6. Calibration and Probability Policy
- **Multiclass Probability Calibration:** Temperature Scaling ($T$) or Vector Scaling fitted strictly on the validation set.
- **Optimization Loss:** Multiclass Cross-Entropy (Log Loss) and Multiclass Brier Score.
- **Unobserved Classes:** Models with unobserved classes ($n\_classes < 3$) cannot be deployed; prior smoothing is permissible only as a fail-safe fallback.

---

## 7. Execution and Cost Accounting
All backtest and forward evaluation must account for realistic market execution:
- **BUY Order:** Entry at $\text{Ask} = P_{\text{mid}} + \frac{1}{2}\text{Spread} + \text{Slippage}$; Exit at $\text{Bid} = P_{\text{mid}} - \frac{1}{2}\text{Spread} - \text{Slippage}$.
- **SELL Order:** Entry at $\text{Bid} = P_{\text{mid}} - \frac{1}{2}\text{Spread} - \text{Slippage}$; Exit at $\text{Ask} = P_{\text{mid}} + \frac{1}{2}\text{Spread} + \text{Slippage}$.
- **Slippage Rate:** $0.25\text{ bps}$ ($0.000025$).
- **Commission Rate:** $0.50\text{ bps}$ ($0.000050$).
- **Overnight Financing / Rollover:** $0.20\text{ bps}$ per 24 hours holding.

---

## 8. Benchmark Baselines
Candidate models must outperform four competitive baselines on the exact same holdout bars:
1. **Majority Class Baseline:** Always predicts the dominant training class (derived solely from training data).
2. **Class-Prior Baseline:** Emits fixed prior class probabilities $(\pi_{\text{sell}}, \pi_{\text{neutral}}, \pi_{\text{buy}})$ derived strictly from training data.
3. **Persistence / Momentum Baseline:** Evaluated on explicit matching test indices. Uses three-class policy: BUY ($2$) if $R_1 \ge +\theta$, SELL ($0$) if $R_1 \le -\theta$, NO_TRADE ($1$) if $|R_1| < \theta$.
4. **No-Trade Baseline:** Returns $0.0\%$ return with $0$ trades and $0$ drawdown.

---

## 9. Promotion and Rollback Gates
Before any model candidate is promoted to paper-trading champion:
1. **Holdout Accuracy:** Must exceed Majority Class baseline by $\ge 2.0\%$ absolute.
2. **Brier Score:** Multiclass Brier Score $< 0.3300$.
3. **Net Return:** Must be strictly positive after all spreads, slippage, and commissions.
4. **Prospective Forward Validation:** Must record $\ge 100$ valid prospective predictions on unseen future market sessions with positive expectancy.
