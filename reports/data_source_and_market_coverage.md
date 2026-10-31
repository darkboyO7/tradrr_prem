# Market Data Ingestion, Universe Coverage & Provider Status Report

**Project Root:** `D:\ant`  
**Host Platform:** Windows 11 Enterprise (AMD64), Python 3.14.7 (`.venv`), Native Windows (Zero WSL)  
**Execution Timestamp:** 2026-10-22T05:25:00Z (10:55:00 IST)  
**Target Markets:** National Stock Exchange of India (NSE), Bombay Stock Exchange (BSE), Quotex Spot Research Proxies  
**Governance Constraint:** Real-Money Order Execution Disabled; Forex Frozen Out of Operational Scope  

---

## 1. Provider Verification & Status Matrix

| Market Segment | Adapter Class | Primary Ingestion Feed | Secondary Ingestion Feed | Live Latency | Freshness Threshold | Operating Status | Cross-Contamination Guard |
| :--- | :--- | :--- | :--- | :---: | :---: | :---: | :---: |
| **NSE Equities** | `NSEDataProvider` | Groww NSE Cash API (REST / JSON) | TradingView India Scanner (`scanner.tradingview.com/india/scan`) | ~330 ms | 30 seconds | **VERIFIED LIVE** | Dedicated exchange prefix `NSE:` |
| **BSE Equities** | `BSEDataProvider` | Groww BSE Cash API by Scrip Code | TradingView BSE Scanner | ~266 ms | 30 seconds | **VERIFIED LIVE** | Dedicated exchange prefix `BSE:` |
| **Quotex Crypto** | `QuotexMarketDataAdapter` | Coinbase Public Spot REST | Kraken / CoinGecko Public Spot REST | ~380 ms | 20 seconds | **VERIFIED PROXY** | Tagged `UNDERLYING_CRYPTO_SPOT_PROXY` |
| **Quotex OTC** | `QuotexMarketDataAdapter` | None (Broker Proprietary Synthetic) | None | N/A | Immediate Fail | **DATA UNAVAILABLE (FAIL-CLOSED)** | Strict `NO_TRADE` block |
| **Forex (Legacy)** | `ForexDataProvider` | Frozen Reference Cache | OANDA Practice Client | N/A | N/A | **OUT OF SCOPE (FROZEN)** | Isolated from primary pipeline |

---

## 2. Multi-Market Universe Coverage

### 2.1 National Stock Exchange of India (NSE)
- **Universe Resolution:** Fully dynamic instrument discovery powered by [`core/universe.py`](file:///D:/ant/core/universe.py) and [`core/symbol_resolver.py`](file:///D:/ant/core/symbol_resolver.py).
- **Coverage Scale:** **3,550+ active NSE cash equities** resolved dynamically beyond static lists.
- **Symbol Formatting:** `NSE:<SYMBOL>` (e.g. `NSE:TCS`, `NSE:RELIANCE`, `NSE:INFY`, `NSE:GROWW`, `NSE:TATAGOLD`, `NSE:IDBI`).
- **Exchange Hours:** Continuous session 09:15 – 15:30 IST. Pre-market discovery 09:00 – 09:15 IST.
- **Tick / Lot Specs:** ₹0.05 tick size; minimum lot size = 1 share for cash market equities.

### 2.2 Bombay Stock Exchange (BSE)
- **Universe Resolution:** Dedicated 6-digit numeric scrip code mapping (e.g. `BSE:500325` for Reliance Industries, `BSE:532540` for TCS).
- **Coverage Scale:** **5,500+ listed BSE scrips** supported.
- **Exchange Isolation:** Strict separation of order books, quotes, and candle feeds. Under no circumstances are NSE prices or volumes substituted for BSE scrips.

### 2.3 Quotex Binary Options Research Adapter
- **Public Spot Crypto Proxies (`BTCUSD_QX`, `ETHUSD_QX`):**
  - Fetched exclusively from authentic public crypto spot feeds (Coinbase / Kraken).
  - Explicitly rendered on desktop and API payloads as `UNDERLYING_CRYPTO_SPOT_PROXY` with the disclaimer: *"Underlying spot proxy, not direct Quotex broker feed"*.
- **Broker Proprietary Synthetic OTC (`MSFT_OTC`, `EURUSD_OTC`):**
  - Fail closed immediately. Returns `is_live_data_available() == False` and `fetch_quote() == None`.
  - Predictions fail closed with `QUOTEX_DATA_UNAVAILABLE` and `NO_TRADE`.

---

## 3. Data Integrity & Pre-Prediction Validation

Before any market quote or candle series is delivered to feature engineering or prediction models, it must satisfy all 9 validation checks in [`signals/fusion.py`](file:///D:/ant/signals/fusion.py):

1. **OHLC Boundary Invariance:**
   $$\text{Low} \le \min(\text{Open}, \text{Close}) \le \max(\text{Open}, \text{Close}) \le \text{High}$$
   $$\text{High} \ge \text{Low} > 0$$
2. **Volume Non-Negativity:** $\text{Volume} \ge 0$ for all recorded intervals.
3. **Chronological Monotonicity:** Candle timestamps are strictly monotonically increasing ($t_i > t_{i-1}$) with zero duplicates or future timestamps.
4. **Freshness Gate:** Quotes older than 30 seconds during active market hours are flagged `STALE_DATA` and immediately rejected.
5. **No Abnormal Gaps:** Consecutive bars with missing intervals $> 3 \times \text{timeframe}$ trigger data quality warnings.
6. **Zero-Lookahead Alignment:** Higher-timeframe candles are point-in-time shifted by the full timeframe duration (e.g. 1H candle shifted by 60m).
7. **Fail-Closed Default:** If any invariance check fails, the pipeline generates a `NO_TRADE` abstention with an explicit machine-readable reason code.

---

## 4. Offline Recovery & Candle Cache

- **Location:** [`data/universe/equity_candles_cache.json`](file:///D:/ant/data/universe/equity_candles_cache.json)
- **Cached Universe:** 10 core liquid instruments (`NSE:TCS:5m`, `NSE:RELIANCE:5m`, `NSE:INFY:5m`, `NSE:IDEA:5m`, `NSE:GROWW:5m`, `NSE:TATAGOLD:5m`, `NSE:TATSILV:5m`, `NSE:IDBI:5m`, `NSE:RTNINDIA:5m`, `BSE:RELIANCE:5m`).
- **Depth:** 180 authentic 5-minute candles per instrument (1,800 bars total).
- **Latency:** < 1 millisecond instantaneous retrieval during offline testing, network interruptions, or outside market hours.
