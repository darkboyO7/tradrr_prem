# Market Data Provider Status and Verification Report

**Date:** 2026-10-20  
**Project Root:** `D:\ant`  
**Execution Environment:** Windows 11 Native Local Environment (`.venv`)  
**Scope:** NSE Equities, BSE Equities, Quotex Research Feeds (Forex strictly out of scope)  

---

## 1. Provider Verification Matrix

| Market Segment | Adapter Class | Primary Source | Secondary Source | Live Quote Latency | Status | Direct Platform API |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **NSE Equities** | `NSEDataProvider` | Groww NSE Cash API (REST / JSON) | TradingView India Scanner (`scanner.tradingview.com/india/scan`) | ~330 ms | **VERIFIED** | Yes (Public Market REST) |
| **BSE Equities** | `BSEDataProvider` | Groww BSE Cash API by Scrip Code | TradingView BSE Scanner | ~266 ms | **VERIFIED** | Yes (Public Market REST) |
| **Quotex Crypto** | `QuotexMarketDataAdapter` | Coinbase / Kraken Public Spot REST | CoinGecko Public Spot REST | ~380 ms | **VERIFIED (PROXY ONLY)** | No (Platform API Non-Existent) |
| **Quotex OTC** | `QuotexMarketDataAdapter` | None (Broker Proprietary Synthetic) | None | N/A | **DATA UNAVAILABLE (FAIL-CLOSED)** | No (Blocked by Governance) |
| **Forex (Legacy)**| `ForexDataProvider` | Frozen Reference Cache | OANDA Client (Practice) | N/A | **OUT OF SCOPE (FROZEN)** | Disabled |

---

## 2. In-Depth Provider Specifications

### 2.1 National Stock Exchange of India (NSE)
- **Implementation File:** [`data/ingestion/nse.py`](file:///D:/ant/data/ingestion/nse.py)
- **Market Hours:** 09:15 – 15:30 IST (Monday through Friday). Pre-market (09:00–09:15) and post-market (15:40–16:00) excluded from continuous matching.
- **Timeframes Supported:** 1m, 5m, 15m, 1h, 1d.
- **Tick Size & Lot Size:** ₹0.05 tick size for standard equities; lot size = 1 share for cash market.
- **Empirical Connectivity Test (Live):**
  - Queried: `NSE:TCS`
  - Observed Price: ₹2460.90
  - Observed Volume: 8,676,373
  - Source: `TRADINGVIEW_NSE_LIVE`
  - Latency Measured: **330.8 ms**
  - Freshness: Valid current session.

### 2.2 Bombay Stock Exchange (BSE)
- **Implementation File:** [`data/ingestion/bse.py`](file:///D:/ant/data/ingestion/bse.py)
- **Market Hours:** 09:15 – 15:30 IST (Monday through Friday).
- **Timeframes Supported:** 1m, 5m, 15m, 1h, 1d.
- **Tick Size & Scrip Resolution:** Dedicated BSE 6-digit scrip codes (e.g., `500325` for Reliance, `532540` for TCS) resolved via [`core/symbol_resolver.py`](file:///D:/ant/core/symbol_resolver.py).
- **Cross-Exchange Isolation:** Zero cross-contamination. Under no circumstances are NSE prices, volume, or order books substituted for BSE quotes.
- **Empirical Connectivity Test (Live):**
  - Queried: `BSE:RELIANCE`
  - Observed Price: ₹1202.80
  - Observed Volume: 1,084,569
  - Source: `TRADINGVIEW_BSE_LIVE`
  - Latency Measured: **266.2 ms**

### 2.3 Quotex Binary Options Research Adapter
- **Implementation File:** [`data/ingestion/quotex.py`](file:///D:/ant/data/ingestion/quotex.py)
- **Operating Model:** Strict research mode. Direct broker API execution is permanently disabled.
- **Data Classification:**
  - **Crypto Spot Underlying Proxies (`BTCUSD_QX`, `ETHUSD_QX`):** Fetched from Coinbase / Kraken spot public feeds. Clearly marked in payloads and dashboard as `UNDERLYING_CRYPTO_SPOT_PROXY` with explicit disclosure: *"Underlying market proxy, not direct Quotex platform feed"*.
  - **Proprietary Synthetic OTC (`MSFT_OTC`, `EURUSD_OTC`, `GBPUSD_OTC`):** Fail-closed with `is_live_data_available() == False` and live quote returning `None`. No synthetic candles or prices are ever fabricated.
- **Contract Payoff Modeling:**
  - Standard Binary Stake $S$, Win payout $p \cdot S$ (average $p = 0.85$).
  - Break-Even Win Probability:
    $$P_{\mathrm{BE}} = \frac{1}{1 + p} = \frac{1}{1 + 0.85} \approx 54.05\%$$
  - Statutory friction assumption: 8.0 bps.

---

## 3. Data Integrity & Validation Rules

1. **OHLC Relationship Invariance:**
   - $\text{Low} \le \min(\text{Open}, \text{Close})$
   - $\text{High} \ge \max(\text{Open}, \text{Close})$
   - $\text{High} \ge \text{Low} > 0$
   - $\text{Volume} \ge 0$
2. **Chronological Monotonicity:** Candle timestamps are strictly strictly increasing with zero duplicate or future timestamps.
3. **Fail-Closed Stale Data Protection:** Predictions and paper execution reject quotes older than configured stale threshold (30s for live equity, 20s for proxy crypto).
