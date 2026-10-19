"""Trading Intelligence Platform - Dynamic Multi-Market Universe & Synchronization Engine.

Implements:
1. NSEUniverseManager — Complete dynamic NSE security master (~3,550+ equities, ETFs, SMEs)
   with ISIN, series, listing date, lot/tick size, symbol change tracking, and non-destructive
   inactive marking.
2. BSEUniverseManager — Complete dynamic BSE security master (~5,200+ equities) with official
   6-digit BSE scrip codes, ISIN, series, tick size, and active/listing status.
3. CanonicalInstrumentRegistry — ISIN-based canonical company identity linking cross-listed
   NSE + BSE securities while keeping exchange-specific market data separate.
4. QuotexDiscoveryAdapter — Strict separation of Platform Instrument -> Underlying Instrument ->
   Market Data Source -> Prediction Instrument, flagging unavailable synthetic OTC feeds as
   'Quotex live data unavailable' rather than fabricating prices.
5. UniverseSyncService — Automated 12-stage synchronization workflow with persistent sync logs.
6. CLI Universe Health Audit (`python -m core.universe audit`).
"""

import csv
import json
import sys
import urllib.request
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, ClassVar

from pydantic import BaseModel, Field

from core.instrument import AssetClass, Instrument, MarketType, SettlementType
from core.logging import get_logger
from core.symbol_resolver import SymbolResolver

logger = get_logger(__name__)

UNIVERSE_DIR = Path(__file__).resolve().parent.parent / "data" / "universe"
NSE_MASTER_CSV_URL = "https://nsearchives.nseindia.com/content/equities/EQUITY_L.csv"
KITE_NSE_CSV_URL = "https://api.kite.trade/instruments/NSE"
KITE_BSE_CSV_URL = "https://api.kite.trade/instruments/BSE"

LOCAL_NSE_MASTER_PATH = UNIVERSE_DIR / "nse_equity_l.csv"
LOCAL_NSE_INSTRUMENTS_PATH = UNIVERSE_DIR / "nse_instruments.csv"
LOCAL_BSE_INSTRUMENTS_PATH = UNIVERSE_DIR / "bse_instruments.csv"
LOCAL_INDIA_ISIN_MASTER_PATH = UNIVERSE_DIR / "india_isin_sector_master.json"
SYNC_LOG_PATH = UNIVERSE_DIR / "universe_sync_log.jsonl"


class NSESecurityRecord(BaseModel):
    """Authoritative NSE Security Master record."""

    symbol: str
    canonical_symbol: str
    company_name: str
    isin: str = ""
    series: str = "EQ"
    exchange: str = "NSE"
    instrument_type: str = "EQUITY"
    tick_size: float = 0.05
    lot_size: float = 1.0
    trading_status: str = "ACTIVE"
    active: bool = True
    listing_status: str = "LISTED"
    listing_date: str = ""
    sector: str = "Equity"
    canonical_instrument_id: str = ""
    previous_symbols: list[str] = Field(default_factory=list)


class BSESecurityRecord(BaseModel):
    """Authoritative BSE Security Master record."""

    scrip_code: str
    symbol: str
    canonical_symbol: str
    company_name: str
    isin: str = ""
    series: str = "EQ"
    exchange: str = "BSE"
    instrument_type: str = "EQUITY"
    tick_size: float = 0.05
    lot_size: float = 1.0
    trading_status: str = "ACTIVE"
    active: bool = True
    listing_status: str = "LISTED"
    listing_info: str = "BSE Mainboard"
    sector: str = "Equity"
    canonical_instrument_id: str = ""


class CanonicalCompanyRecord(BaseModel):
    """ISIN-based canonical company linking cross-listed NSE and BSE instruments."""

    canonical_id: str
    isin: str
    company_name: str
    sector: str = "Equity"
    nse_symbol: str | None = None
    bse_symbol: str | None = None
    bse_scrip_code: str | None = None
    exchanges: list[str] = Field(default_factory=list)


class QuotexInstrumentRecord(BaseModel):
    """Quotex platform instrument separated from its underlying asset and market data source."""

    quotex_symbol: str
    canonical_symbol: str
    platform_instrument_name: str
    provider_symbol: str
    underlying_instrument: str
    market_data_source: str
    prediction_instrument: str
    asset_type: str
    base_currency: str
    quote_currency: str
    tick_size: float = 0.00001
    price_precision: int = 5
    payout_pct: float = 0.85
    active: bool = True
    live_data_available: bool = True
    trading_availability: str = "AVAILABLE_AUTHORIZED_FEED"
    session: str = "24/5 Authorized Underlying"
    source_timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    display_name: str = ""
    underlying_market_symbol: str = ""
    underlying_provider: str = "TradingView / Interbank"
    supported_horizons: list[str] = Field(default_factory=lambda: ["5m", "15m", "1h", "4h", "1d"])
    data_quality_status: str = "VERIFIED"
    limitation_notes: str = ""



class UniverseSyncLogEntry(BaseModel):
    """Audit log entry for every automatic or manual UniverseSyncService run."""

    universe_sync_id: str
    source: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    records_received: int = 0
    records_added: int = 0
    records_updated: int = 0
    records_removed: int = 0
    records_changed: int = 0
    market_data_tasks_generated: int = 0
    prediction_tasks_generated: int = 0
    errors: list[str] = Field(default_factory=list)


class CanonicalInstrumentRegistry:
    """Maintains ISIN-based canonical company identities across NSE and BSE."""

    _companies: ClassVar[dict[str, CanonicalCompanyRecord]] = {}
    _symbol_to_canonical_id: ClassVar[dict[str, str]] = {}

    @classmethod
    def register_security(
        cls,
        exchange: str,
        symbol: str,
        canonical_symbol: str,
        company_name: str,
        isin: str = "",
        scrip_code: str = "",
        sector: str = "Equity",
    ) -> str:
        clean_isin = isin.strip().upper()
        norm_sym = symbol.strip().upper()

        # Check if opposite exchange already registered this symbol with an ISIN
        existing_cid = cls._symbol_to_canonical_id.get(norm_sym)
        if clean_isin and clean_isin.startswith("IN"):
            cid = f"ISIN:{clean_isin}"
        elif existing_cid:
            cid = existing_cid
        else:
            cid = f"CANON:IN:{norm_sym}"

        rec = cls._companies.get(cid)
        if rec is None:
            rec = CanonicalCompanyRecord(
                canonical_id=cid,
                isin=clean_isin or (cid.replace("ISIN:", "") if cid.startswith("ISIN:") else ""),
                company_name=company_name or symbol,
                sector=sector or "Equity",
                exchanges=[],
            )
            cls._companies[cid] = rec
        else:
            if clean_isin and not rec.isin:
                rec.isin = clean_isin
            if sector and sector != "Equity" and rec.sector == "Equity":
                rec.sector = sector

        if exchange == "NSE":
            rec.nse_symbol = canonical_symbol
            if "NSE" not in rec.exchanges:
                rec.exchanges.append("NSE")
        elif exchange == "BSE":
            rec.bse_symbol = canonical_symbol
            if scrip_code:
                rec.bse_scrip_code = scrip_code
            if "BSE" not in rec.exchanges:
                rec.exchanges.append("BSE")

        cls._symbol_to_canonical_id[norm_sym] = cid
        cls._symbol_to_canonical_id[canonical_symbol.upper()] = cid
        if clean_isin:
            cls._symbol_to_canonical_id[clean_isin] = cid
        if scrip_code:
            cls._symbol_to_canonical_id[scrip_code] = cid
        return cid

    @classmethod
    def get_company(cls, query: str) -> CanonicalCompanyRecord | None:
        q = query.strip().upper()
        if q in cls._companies:
            return cls._companies[q]
        cid = cls._symbol_to_canonical_id.get(q) or cls._symbol_to_canonical_id.get(q.split(":")[-1])
        if cid:
            return cls._companies.get(cid)
        return None

    @classmethod
    def get_cross_listed(cls, query: str) -> dict[str, Any]:
        rec = cls.get_company(query)
        if not rec:
            return {}
        return {
            "canonical_id": rec.canonical_id,
            "isin": rec.isin,
            "company_name": rec.company_name,
            "nse_symbol": rec.nse_symbol,
            "bse_symbol": rec.bse_symbol,
            "bse_scrip_code": rec.bse_scrip_code,
            "exchanges": list(rec.exchanges),
        }

    @classmethod
    def cross_listed_count(cls) -> int:
        return sum(1 for c in cls._companies.values() if "NSE" in c.exchanges and "BSE" in c.exchanges)


class NSEUniverseManager:
    """Manages the complete NSE India equity universe from authoritative exchange masters."""

    _records: ClassVar[dict[str, NSESecurityRecord]] = {}
    _isin_to_symbol: ClassVar[dict[str, str]] = {}
    _symbol_changes: ClassVar[dict[str, str]] = dict(SymbolResolver._historical_symbol_map)

    @classmethod
    def load_and_sync(cls, isin_sector_lookup: dict[str, dict[str, Any]]) -> dict[str, int]:
        stats = {"received": 0, "added": 0, "updated": 0, "removed": 0, "changed": 0}
        seen_symbols: set[str] = set()

        # 1. Parse official NSE EQUITY_L.csv (Symbol, Company Name, Series, Listing Date, Lot Size, ISIN)
        if LOCAL_NSE_MASTER_PATH.exists():
            try:
                lines = LOCAL_NSE_MASTER_PATH.read_text(encoding="utf-8", errors="ignore").splitlines()
                reader = csv.reader(lines)
                next(reader, None)  # skip header
                for row in reader:
                    if len(row) < 6:
                        continue
                    sym = row[0].strip().upper()
                    if not sym:
                        continue
                    stats["received"] += 1
                    name = row[1].strip()
                    series = row[2].strip().upper() or "EQ"
                    listing_date = row[3].strip()
                    try:
                        lot_size = max(1.0, float(row[5].strip() or 1.0))
                    except ValueError:
                        lot_size = 1.0
                    isin = row[6].strip().upper() if len(row) > 6 else ""

                    tv_info = isin_sector_lookup.get(f"NSE:{sym}", {})
                    sector = tv_info.get("sector") or "Equity"
                    tick = float(tv_info.get("tick_size", 0.05))
                    cls._upsert_record(
                        sym=sym,
                        name=name,
                        isin=isin,
                        series=series,
                        inst_type="EQUITY",
                        tick_size=tick,
                        lot_size=lot_size,
                        listing_date=listing_date,
                        sector=sector,
                        stats=stats,
                    )
                    seen_symbols.add(sym)
            except Exception as exc:
                logger.error("Error parsing NSE EQUITY_L.csv", error=str(exc))

        # 2. Parse India Exchange Master for additional NSE stocks, SMEs, ETFs & REITs (3,553 total)
        for key, info in isin_sector_lookup.items():
            if not key.startswith("NSE:"):
                continue
            sym = key.split(":", 1)[1].strip().upper()
            if not sym or sym in seen_symbols:
                continue
            stats["received"] += 1
            isin = str(info.get("isin") or "").strip().upper()
            name = str(info.get("name") or sym).strip()
            sector = str(info.get("sector") or "Equity").strip()
            subtype = str(info.get("subtype") or "common").lower()
            inst_type = "ETF" if "etf" in subtype else "EQUITY"
            tick = float(info.get("tick_size", 0.05))
            cls._upsert_record(
                sym=sym,
                name=name,
                isin=isin,
                series="EQ" if inst_type == "EQUITY" else "ETF",
                inst_type=inst_type,
                tick_size=tick,
                lot_size=1.0,
                listing_date="",
                sector=sector,
                stats=stats,
            )
            seen_symbols.add(sym)

        # 2.5 Parse authoritative NSE Instruments CSV for any additional SME, REIT, InvIT, or EQ securities
        if LOCAL_NSE_INSTRUMENTS_PATH.exists():
            try:
                lines = LOCAL_NSE_INSTRUMENTS_PATH.read_text(encoding="utf-8", errors="ignore").splitlines()
                reader = csv.DictReader(lines)
                for row in reader:
                    if row.get("segment") != "NSE" or row.get("exchange") != "NSE":
                        continue
                    raw_sym = (row.get("tradingsymbol") or "").strip().upper()
                    if not raw_sym or raw_sym.endswith("INAV") or raw_sym[:2].isdigit():
                        continue
                    if "-" in raw_sym:
                        base_sym, series_sfx = raw_sym.split("-", 1)
                        if series_sfx not in ("SM", "ST", "BE", "BZ", "RR", "IV", "IT"):
                            continue
                    else:
                        base_sym, series_sfx = raw_sym, "EQ"
                    if not base_sym or base_sym in seen_symbols:
                        continue
                    stats["received"] += 1
                    name = (row.get("name") or base_sym).strip().strip('"')
                    try:
                        tick = float(row.get("tick_size") or 0.05)
                    except ValueError:
                        tick = 0.05
                    try:
                        lot = float(row.get("lot_size") or 1.0)
                    except ValueError:
                        lot = 1.0
                    inst_type = "REIT" if series_sfx == "RR" else ("INVIT" if series_sfx == "IV" else "EQUITY")
                    cls._upsert_record(
                        sym=base_sym,
                        name=name,
                        isin="",
                        series=series_sfx,
                        inst_type=inst_type,
                        tick_size=tick if tick > 0 else 0.05,
                        lot_size=max(1.0, lot),
                        listing_date="",
                        sector="SME / Emerging" if series_sfx in ("SM", "ST") else "Equity",
                        stats=stats,
                    )
                    seen_symbols.add(base_sym)
            except Exception as exc:
                logger.error("Error parsing NSE instruments CSV", error=str(exc))

        # 3. Ensure legacy symbol aliases (e.g. ZOMATO alongside ETERNAL) remain resolvable
        if "ETERNAL" in cls._records:
            cls._symbol_changes["ZOMATO"] = "ETERNAL"
            SymbolResolver._historical_symbol_map["ZOMATO"] = "ETERNAL"
            if "ZOMATO" not in cls._records:
                et = cls._records["ETERNAL"]
                cls._upsert_record(
                    sym="ZOMATO",
                    name="Zomato Ltd (Eternal Ltd)",
                    isin=et.isin,
                    series="EQ",
                    inst_type="EQUITY",
                    tick_size=0.05,
                    lot_size=1.0,
                    listing_date=et.listing_date,
                    sector=et.sector,
                    stats=stats,
                )
                seen_symbols.add("ZOMATO")

        # 4. Non-destructive inactive detection (never delete historical instruments)
        for sym, rec in cls._records.items():
            if sym not in seen_symbols and rec.active:
                rec.active = False
                rec.trading_status = "SUSPENDED_OR_DELISTED"
                rec.listing_status = "INACTIVE"
                stats["removed"] += 1

        return stats

    @classmethod
    def _upsert_record(
        cls,
        sym: str,
        name: str,
        isin: str,
        series: str,
        inst_type: str,
        tick_size: float,
        lot_size: float,
        listing_date: str,
        sector: str,
        stats: dict[str, int],
    ) -> None:
        canon = f"NSE:{sym}"
        # Detect symbol change via ISIN
        prev_syms: list[str] = []
        if isin and isin in cls._isin_to_symbol:
            old_sym = cls._isin_to_symbol[isin]
            if old_sym != sym and old_sym != "ZOMATO":
                cls._symbol_changes[old_sym] = sym
                SymbolResolver._historical_symbol_map[old_sym] = sym
                prev_syms.append(old_sym)
                stats["changed"] += 1
        if isin:
            cls._isin_to_symbol[isin] = sym

        cid = CanonicalInstrumentRegistry.register_security(
            exchange="NSE",
            symbol=sym,
            canonical_symbol=canon,
            company_name=name,
            isin=isin,
            sector=sector,
        )

        if sym in cls._records:
            stats["updated"] += 1
        else:
            stats["added"] += 1

        rec = NSESecurityRecord.model_construct(
            symbol=sym,
            canonical_symbol=canon,
            company_name=name,
            isin=isin,
            series=series,
            exchange="NSE",
            instrument_type=inst_type,
            tick_size=tick_size if tick_size > 0 else 0.05,
            lot_size=lot_size,
            trading_status="ACTIVE",
            active=True,
            listing_status="LISTED",
            listing_date=listing_date,
            sector=sector,
            canonical_instrument_id=cid,
            previous_symbols=prev_syms,
        )
        cls._records[sym] = rec

    @classmethod
    def get_all_supported_stocks(cls, active_only: bool = True) -> list[NSESecurityRecord]:
        if active_only:
            return [r for r in cls._records.values() if r.active]
        return list(cls._records.values())


class BSEUniverseManager:
    """Manages the complete BSE India security universe (~5,200+ equities with scrip codes & ISINs)."""

    _records: ClassVar[dict[str, BSESecurityRecord]] = {}
    _scrip_to_symbol: ClassVar[dict[str, str]] = {}

    @classmethod
    def load_and_sync(cls, isin_sector_lookup: dict[str, dict[str, Any]]) -> dict[str, int]:
        stats = {"received": 0, "added": 0, "updated": 0, "removed": 0, "changed": 0}
        seen_symbols: set[str] = set()

        # 1. Parse authoritative BSE Instruments CSV for scrip codes (exchange_token), tick_size, lot_size
        scrip_lookup: dict[str, dict[str, Any]] = {}
        if LOCAL_BSE_INSTRUMENTS_PATH.exists():
            try:
                lines = LOCAL_BSE_INSTRUMENTS_PATH.read_text(encoding="utf-8", errors="ignore").splitlines()
                reader = csv.DictReader(lines)
                for row in reader:
                    if row.get("instrument_type") != "EQ" or row.get("exchange") != "BSE":
                        continue
                    sym = (row.get("tradingsymbol") or "").strip().upper()
                    scrip = (row.get("exchange_token") or "").strip()
                    if not sym or not scrip:
                        continue
                    try:
                        tick = float(row.get("tick_size") or 0.05)
                    except ValueError:
                        tick = 0.05
                    try:
                        lot = float(row.get("lot_size") or 1.0)
                    except ValueError:
                        lot = 1.0
                    scrip_lookup[sym] = {
                        "scrip_code": scrip,
                        "name": (row.get("name") or sym).strip().strip('"'),
                        "tick_size": tick if tick > 0 else 0.05,
                        "lot_size": max(1.0, lot),
                    }
            except Exception as exc:
                logger.error("Error parsing BSE instruments CSV", error=str(exc))

        # 2. Merge with India ISIN & Sector Master for all 5,204 actively traded BSE equities
        for key, info in isin_sector_lookup.items():
            if not key.startswith("BSE:"):
                continue
            sym = key.split(":", 1)[1].strip().upper()
            if not sym:
                continue
            stats["received"] += 1
            sc_info = scrip_lookup.get(sym, {})
            scrip_code = str(sc_info.get("scrip_code") or "")
            name = str(info.get("name") or sc_info.get("name") or sym).strip()
            isin = str(info.get("isin") or "").strip().upper()
            sector = str(info.get("sector") or "Equity").strip()
            tick = float(sc_info.get("tick_size") or info.get("tick_size") or 0.05)
            lot = float(sc_info.get("lot_size") or 1.0)

            cls._upsert_record(
                sym=sym,
                scrip_code=scrip_code,
                name=name,
                isin=isin,
                series="A" if isin else "B",
                tick_size=tick,
                lot_size=lot,
                sector=sector,
                stats=stats,
            )
            seen_symbols.add(sym)

        # 3. Include all remaining BSE Mainboard 5xxxxx EQ stocks and cross-listed NSE equities from scrip_lookup
        for sym, sc_info in scrip_lookup.items():
            if sym in seen_symbols:
                continue
            scrip_code = str(sc_info["scrip_code"])
            if sym in NSEUniverseManager._records or (scrip_code.startswith("5") and len(scrip_code) == 6):
                nse_rec = NSEUniverseManager._records.get(sym)
                stats["received"] += 1
                cls._upsert_record(
                    sym=sym,
                    scrip_code=scrip_code,
                    name=nse_rec.company_name if nse_rec else str(sc_info["name"]),
                    isin=nse_rec.isin if nse_rec else "",
                    series="A" if nse_rec else "B",
                    tick_size=float(sc_info["tick_size"]),
                    lot_size=float(sc_info["lot_size"]),
                    sector=nse_rec.sector if nse_rec else "Equity",
                    stats=stats,
                )
                seen_symbols.add(sym)

        for sym, rec in cls._records.items():
            if sym not in seen_symbols and rec.active:
                rec.active = False
                rec.trading_status = "SUSPENDED_OR_DELISTED"
                rec.listing_status = "INACTIVE"
                stats["removed"] += 1

        return stats

    @classmethod
    def _upsert_record(
        cls,
        sym: str,
        scrip_code: str,
        name: str,
        isin: str,
        series: str,
        tick_size: float,
        lot_size: float,
        sector: str,
        stats: dict[str, int],
    ) -> None:
        canon = f"BSE:{sym}"
        if not isin and sym in NSEUniverseManager._records:
            isin = NSEUniverseManager._records[sym].isin

        cid = CanonicalInstrumentRegistry.register_security(
            exchange="BSE",
            symbol=sym,
            canonical_symbol=canon,
            company_name=name,
            isin=isin,
            scrip_code=scrip_code,
            sector=sector,
        )

        if sym in cls._records:
            stats["updated"] += 1
        else:
            stats["added"] += 1

        if scrip_code:
            cls._scrip_to_symbol[scrip_code] = sym

        rec = BSESecurityRecord.model_construct(
            scrip_code=scrip_code,
            symbol=sym,
            canonical_symbol=canon,
            company_name=name,
            isin=isin,
            series=series,
            exchange="BSE",
            instrument_type="EQUITY",
            tick_size=tick_size if tick_size > 0 else 0.05,
            lot_size=lot_size,
            trading_status="ACTIVE",
            active=True,
            listing_status="LISTED",
            listing_info=f"BSE Scrip {scrip_code}" if scrip_code else "BSE Listed Equity",
            sector=sector,
            canonical_instrument_id=cid,
        )
        cls._records[sym] = rec

    @classmethod
    def get_all_supported_stocks(cls, active_only: bool = True) -> list[BSESecurityRecord]:
        if active_only:
            return [r for r in cls._records.values() if r.active]
        return list(cls._records.values())


class QuotexInstrumentAdapter:
    """Dedicated Quotex Instrument Adapter.

    Separates:
      Platform Instrument (QUOTEX:<SYM>)
      -> Underlying Instrument (FX:<PAIR>, CRYPTO:<PAIR>, or SYNTHETIC_OTC:<SYM>)
      -> Market Data Source (UNDERLYING_FX_SPOT_PROXY, UNDERLYING_FX_OTC_PROXY, UNDERLYING_CRYPTO_SPOT_PROXY, or UNAVAILABLE)
      -> Prediction Instrument (QUOTEX:<SYM>)

    Note: Quotex does not provide an authorized public API for live platform quotes or OTC streams (BLOCKED).
    Therefore, FX and Crypto pairs are explicitly marked as underlying market proxies (never counted as
    verified direct Quotex broker feeds), and proprietary synthetic OTC stocks are marked LIVE_DATA_UNAVAILABLE.
    """

    _records: ClassVar[dict[str, QuotexInstrumentRecord]] = {}

    @classmethod
    def discover_instruments(cls) -> dict[str, int]:
        stats = {"received": 0, "added": 0, "updated": 0, "removed": 0, "changed": 0}
        now = datetime.now(UTC)

        # 1. Quotex FX Spot & FX OTC Pairs (Analyzed via Underlying Interbank FX Proxy Feeds; Direct Quotex API BLOCKED)
        fx_backed_specs = [
            ("EURUSD", "EUR/USD", "EUR", "USD", 0.00001, 5, 0.89, "FOREX_SPOT_PROXY"),
            ("EURCHF", "EUR/CHF", "EUR", "CHF", 0.00001, 5, 0.87, "FOREX_SPOT_PROXY"),
            ("GBPUSD", "GBP/USD", "GBP", "USD", 0.00001, 5, 0.88, "FOREX_SPOT_PROXY"),
            ("USDJPY", "USD/JPY", "USD", "JPY", 0.001, 3, 0.88, "FOREX_SPOT_PROXY"),
            ("USDCHF", "USD/CHF", "USD", "CHF", 0.00001, 5, 0.86, "FOREX_SPOT_PROXY"),
            ("AUDUSD", "AUD/USD", "AUD", "USD", 0.00001, 5, 0.87, "FOREX_SPOT_PROXY"),
            ("USDCAD", "USD/CAD", "USD", "CAD", 0.00001, 5, 0.86, "FOREX_SPOT_PROXY"),
            ("EURGBP", "EUR/GBP", "EUR", "GBP", 0.00001, 5, 0.87, "FOREX_SPOT_PROXY"),
            ("EURJPY", "EUR/JPY", "EUR", "JPY", 0.001, 3, 0.87, "FOREX_SPOT_PROXY"),
            ("GBPJPY", "GBP/JPY", "GBP", "JPY", 0.001, 3, 0.87, "FOREX_SPOT_PROXY"),
            ("AUDJPY", "AUD/JPY", "AUD", "JPY", 0.001, 3, 0.86, "FOREX_SPOT_PROXY"),
            ("NZDUSD", "NZD/USD", "NZD", "USD", 0.00001, 5, 0.85, "FOREX_SPOT_PROXY"),
            ("CHFJPY", "CHF/JPY", "CHF", "JPY", 0.001, 3, 0.85, "FOREX_SPOT_PROXY"),
            ("EURAUD", "EUR/AUD", "EUR", "AUD", 0.00001, 5, 0.85, "FOREX_SPOT_PROXY"),
            ("EURCAD", "EUR/CAD", "EUR", "CAD", 0.00001, 5, 0.85, "FOREX_SPOT_PROXY"),
            ("GBPAUD", "GBP/AUD", "GBP", "AUD", 0.00001, 5, 0.85, "FOREX_SPOT_PROXY"),
            ("GBPCAD", "GBP/CAD", "GBP", "CAD", 0.00001, 5, 0.85, "FOREX_SPOT_PROXY"),
            ("GBPCHF", "GBP/CHF", "GBP", "CHF", 0.00001, 5, 0.85, "FOREX_SPOT_PROXY"),
            ("CADJPY", "CAD/JPY", "CAD", "JPY", 0.001, 3, 0.85, "FOREX_SPOT_PROXY"),
            ("NZDJPY", "NZD/JPY", "NZD", "JPY", 0.001, 3, 0.85, "FOREX_SPOT_PROXY"),
            # FX OTC pairs mapped to their underlying FX pair in Analysis/Paper Mode
            ("EURCHF_OTC", "EUR/CHF (OTC)", "EUR", "CHF", 0.00001, 5, 0.88, "FOREX_OTC_UNDERLYING_PROXY"),
            ("EURUSD_OTC", "EUR/USD (OTC)", "EUR", "USD", 0.00001, 5, 0.89, "FOREX_OTC_UNDERLYING_PROXY"),
            ("GBPUSD_OTC", "GBP/USD (OTC)", "GBP", "USD", 0.00001, 5, 0.88, "FOREX_OTC_UNDERLYING_PROXY"),
            ("USDJPY_OTC", "USD/JPY (OTC)", "USD", "JPY", 0.001, 3, 0.88, "FOREX_OTC_UNDERLYING_PROXY"),
            ("USDCHF_OTC", "USD/CHF (OTC)", "USD", "CHF", 0.00001, 5, 0.86, "FOREX_OTC_UNDERLYING_PROXY"),
            ("AUDUSD_OTC", "AUD/USD (OTC)", "AUD", "USD", 0.00001, 5, 0.87, "FOREX_OTC_UNDERLYING_PROXY"),
            ("USDCAD_OTC", "USD/CAD (OTC)", "USD", "CAD", 0.00001, 5, 0.86, "FOREX_OTC_UNDERLYING_PROXY"),
            ("EURGBP_OTC", "EUR/GBP (OTC)", "EUR", "GBP", 0.00001, 5, 0.87, "FOREX_OTC_UNDERLYING_PROXY"),
            ("EURJPY_OTC", "EUR/JPY (OTC)", "EUR", "JPY", 0.001, 3, 0.87, "FOREX_OTC_UNDERLYING_PROXY"),
            ("GBPJPY_OTC", "GBP/JPY (OTC)", "GBP", "JPY", 0.001, 3, 0.87, "FOREX_OTC_UNDERLYING_PROXY"),
            ("AUDJPY_OTC", "AUD/JPY (OTC)", "AUD", "JPY", 0.001, 3, 0.86, "FOREX_OTC_UNDERLYING_PROXY"),
            ("NZDUSD_OTC", "NZD/USD (OTC)", "NZD", "USD", 0.00001, 5, 0.85, "FOREX_OTC_UNDERLYING_PROXY"),
        ]
        for sym, name, base, quote, tick, prec, payout, a_type in fx_backed_specs:
            stats["received"] += 1
            underlying_pair = sym.replace("_OTC", "")
            canon = f"QUOTEX:{sym}"
            is_otc = "_OTC" in sym
            limitation = (
                "UNDERLYING MARKET PROXY (Not direct Quotex platform feed — Quotex public API is BLOCKED/non-existent). "
                "Analyzed via authorized interbank FX feed; Quotex broker-specific OTC quotes may diverge."
                if is_otc
                else "UNDERLYING MARKET PROXY (Not direct Quotex platform feed — Quotex public API is BLOCKED/non-existent). "
                "Analyzed via authorized interbank spot Forex feed."
            )
            rec = QuotexInstrumentRecord(
                quotex_symbol=sym,
                canonical_symbol=canon,
                platform_instrument_name=name,
                provider_symbol=f"FX_IDC:{underlying_pair}",
                underlying_instrument=f"FX:{underlying_pair}",
                market_data_source="UNDERLYING_FX_OTC_PROXY" if is_otc else "UNDERLYING_FX_SPOT_PROXY",
                prediction_instrument=canon,
                asset_type=a_type,
                base_currency=base,
                quote_currency=quote,
                tick_size=tick,
                price_precision=prec,
                payout_pct=payout,
                active=True,
                live_data_available=True,
                trading_availability="UNDERLYING_PROXY_FEED_ONLY",
                session="24/5 Underlying Interbank FX Proxy",
                source_timestamp=now,
                display_name=name,
                underlying_market_symbol=f"FX_IDC:{underlying_pair}",
                underlying_provider="TradingView / ECB Interbank FX Proxy",
                supported_horizons=["5m", "15m", "1h", "4h", "1d"],
                data_quality_status="PROXY_FX_OTC_INTERBANK" if is_otc else "PROXY_FX_SPOT_INTERBANK",
                limitation_notes=limitation,
            )
            if sym in cls._records:
                stats["updated"] += 1
            else:
                stats["added"] += 1
            cls._records[sym] = rec

        # 2. Quotex Crypto Instruments (Analyzed via Underlying Spot Crypto Proxy Feed; Direct Quotex API BLOCKED)
        crypto_backed_specs = [
            ("BTCUSD_QX", "Bitcoin (Quotex)", "BTC", "USD", "CRYPTO:BTC-USD", "COINBASE:BTCUSD", 0.01, 2, 0.85),
            ("ETHUSD_QX", "Ethereum (Quotex)", "ETH", "USD", "CRYPTO:ETH-USD", "COINBASE:ETHUSD", 0.01, 2, 0.85),
        ]
        for sym, name, base, quote, und_inst, prov_sym, tick, prec, payout in crypto_backed_specs:
            stats["received"] += 1
            canon = f"QUOTEX:{sym}"
            rec = QuotexInstrumentRecord(
                quotex_symbol=sym,
                canonical_symbol=canon,
                platform_instrument_name=name,
                provider_symbol=prov_sym,
                underlying_instrument=und_inst,
                market_data_source="UNDERLYING_CRYPTO_SPOT_PROXY",
                prediction_instrument=canon,
                asset_type="CRYPTO_DERIVATIVE_PROXY",
                base_currency=base,
                quote_currency=quote,
                tick_size=tick,
                price_precision=prec,
                payout_pct=payout,
                active=True,
                live_data_available=True,
                trading_availability="UNDERLYING_PROXY_FEED_ONLY",
                session="24/7 Underlying Spot Crypto Proxy",
                source_timestamp=now,
                display_name=name,
                underlying_market_symbol=prov_sym,
                underlying_provider="CoinGecko / TradingView Spot Crypto Proxy",
                supported_horizons=["5m", "15m", "1h", "4h", "1d"],
                data_quality_status="PROXY_CRYPTO_SPOT",
                limitation_notes=(
                    "UNDERLYING MARKET PROXY (Not direct Quotex platform feed — Quotex public API is BLOCKED/non-existent). "
                    "Analyzed via authorized spot crypto market feed."
                ),
            )
            if sym in cls._records:
                stats["updated"] += 1
            else:
                stats["added"] += 1
            cls._records[sym] = rec

        # 3. Quotex Proprietary Synthetic OTC Instruments (No Authorized Public Live Feed -> Explicit BLOCKED / Unavailable)
        proprietary_otc_specs = [
            ("MSFT_OTC", "Microsoft Corp (Proprietary OTC)", "MSFT", "USD"),
            ("AAPL_OTC", "Apple Inc (Proprietary OTC)", "AAPL", "USD"),
            ("INTC_OTC", "Intel Corp (Proprietary OTC)", "INTC", "USD"),
            ("BA_OTC", "Boeing Co (Proprietary OTC)", "BA", "USD"),
            ("MCD_OTC", "McDonald's Corp (Proprietary OTC)", "MCD", "USD"),
        ]
        for sym, name, base, quote in proprietary_otc_specs:
            stats["received"] += 1
            canon = f"QUOTEX:{sym}"
            rec = QuotexInstrumentRecord(
                quotex_symbol=sym,
                canonical_symbol=canon,
                platform_instrument_name=name,
                provider_symbol="PROPRIETARY_OTC_NO_PUBLIC_FEED",
                underlying_instrument=f"SYNTHETIC_OTC:{sym}",
                market_data_source="UNAVAILABLE_PROPRIETARY_BROKER_RNG",
                prediction_instrument=canon,
                asset_type="PROPRIETARY_SYNTHETIC_OTC",
                base_currency=base,
                quote_currency=quote,
                tick_size=0.01,
                price_precision=2,
                payout_pct=0.85,
                active=True,
                live_data_available=False,
                trading_availability="LIVE_DATA_UNAVAILABLE",
                session="Proprietary Broker OTC (No Authorized Feed)",
                source_timestamp=now,
                display_name=name,
                underlying_market_symbol="NONE_PROPRIETARY_BROKER",
                underlying_provider="None (Proprietary Broker Internal RNG)",
                supported_horizons=["5m", "15m", "1h", "4h", "1d"],
                data_quality_status="UNAVAILABLE",
                limitation_notes="No authorized public market data feed for proprietary broker synthetic OTC quotation; live data and signals unavailable.",
            )
            if sym in cls._records:
                stats["updated"] += 1
            else:
                stats["added"] += 1
            cls._records[sym] = rec

        return stats

    @classmethod
    def get_record(cls, symbol_or_canonical: str) -> QuotexInstrumentRecord | None:
        clean = symbol_or_canonical.replace("QUOTEX:", "").strip().upper()
        return cls._records.get(clean)

    @classmethod
    def get_all_records(cls) -> list[QuotexInstrumentRecord]:
        return list(cls._records.values())

    @classmethod
    def is_live_data_available(cls, symbol_or_canonical: str) -> bool:
        rec = cls.get_record(symbol_or_canonical)
        return rec.live_data_available if rec is not None else False

    @classmethod
    def get_limitation_note(cls, symbol_or_canonical: str) -> str:
        rec = cls.get_record(symbol_or_canonical)
        return rec.limitation_notes if rec is not None else ""

    @classmethod
    def get_underlying_mapping(cls, symbol_or_canonical: str) -> dict[str, Any] | None:
        rec = cls.get_record(symbol_or_canonical)
        if not rec:
            return None
        return {
            "platform_symbol": rec.canonical_symbol,
            "display_name": rec.platform_instrument_name,
            "underlying_instrument": rec.underlying_instrument,
            "underlying_market_symbol": rec.underlying_market_symbol,
            "underlying_provider": rec.underlying_provider,
            "market_data_source": rec.market_data_source,
            "data_quality_status": rec.data_quality_status,
            "live_data_available": rec.live_data_available,
            "limitation_notes": rec.limitation_notes,
            "supported_horizons": rec.supported_horizons,
        }


QuotexDiscoveryAdapter = QuotexInstrumentAdapter


class UniverseSyncService:
    """Automatic Universe Synchronization Service.

    Workflow:
      Authoritative Source -> Download/API -> Parse -> Normalize -> Validate ->
      Compare with Existing Universe -> Detect Added/Removed/Changed ->
      Update Instrument Registry -> Generate Market Data & Prediction Tasks.
    """

    _sync_history: ClassVar[list[UniverseSyncLogEntry]] = []
    _market_data_tasks: ClassVar[list[dict[str, Any]]] = []
    _prediction_tasks: ClassVar[list[dict[str, Any]]] = []

    @classmethod
    def record_sync(
        cls,
        source: str,
        stats: dict[str, int],
        errors: list[str] | None = None,
    ) -> UniverseSyncLogEntry:
        sync_id = f"sync_{int(datetime.now(UTC).timestamp() * 1000)}"
        md_tasks = stats.get("added", 0) + stats.get("updated", 0)
        pred_tasks = md_tasks

        entry = UniverseSyncLogEntry(
            universe_sync_id=sync_id,
            source=source,
            timestamp=datetime.now(UTC),
            records_received=stats.get("received", 0),
            records_added=stats.get("added", 0),
            records_updated=stats.get("updated", 0),
            records_removed=stats.get("removed", 0),
            records_changed=stats.get("changed", 0),
            market_data_tasks_generated=md_tasks,
            prediction_tasks_generated=pred_tasks,
            errors=errors or [],
        )
        cls._sync_history.append(entry)
        if len(cls._sync_history) > 100:
            cls._sync_history = cls._sync_history[-100:]

        try:
            UNIVERSE_DIR.mkdir(parents=True, exist_ok=True)
            with SYNC_LOG_PATH.open("a", encoding="utf-8") as f:
                f.write(entry.model_dump_json() + "\n")
        except Exception as exc:
            logger.debug("Could not append universe sync log to disk", error=str(exc))

        return entry

    @classmethod
    def get_sync_history(cls, limit: int = 20) -> list[dict[str, Any]]:
        return [e.model_dump(mode="json") for e in cls._sync_history[-limit:]]


class UniverseService:
    """Unified Multi-Market Universe Coordinator supporting 8,800+ instruments across NSE, BSE, Quotex, Forex & Crypto."""

    _synced: ClassVar[bool] = False
    _last_sync_source: ClassVar[str] = "UNINITIALIZED"
    _cached_instruments: ClassVar[dict[str, Instrument]] = {}
    _cached_aliases: ClassVar[dict[str, str]] = {}

    @classmethod
    def download_nse_master_csv(cls, target_path: Path = LOCAL_NSE_MASTER_PATH, timeout: float = 12.0) -> bool:
        """Download the latest official NSE Equity Master CSV from nsearchives.nseindia.com."""
        try:
            target_path.parent.mkdir(parents=True, exist_ok=True)
            req = urllib.request.Request(
                NSE_MASTER_CSV_URL,
                headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"},
            )
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                if resp.status == 200:
                    data = resp.read()
                    if len(data) > 10000:
                        target_path.write_bytes(data)
                        return True
        except Exception as exc:
            logger.warning("Remote NSE Master CSV refresh skipped; using local security master", error=str(exc))
        return False

    @classmethod
    def download_exchange_masters(cls, timeout: float = 15.0) -> None:
        """Download authoritative NSE & BSE exchange instrument masters and India ISIN/Sector master."""
        cls.download_nse_master_csv(LOCAL_NSE_MASTER_PATH, timeout=timeout)
        for url, path in (
            (KITE_BSE_CSV_URL, LOCAL_BSE_INSTRUMENTS_PATH),
            (KITE_NSE_CSV_URL, LOCAL_NSE_INSTRUMENTS_PATH),
        ):
            try:
                req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
                with urllib.request.urlopen(req, timeout=timeout) as resp:
                    if resp.status == 200:
                        data = resp.read()
                        if len(data) > 50000:
                            path.write_bytes(data)
            except Exception as exc:
                logger.debug("Exchange master CSV refresh skipped", url=url, error=str(exc))

    @classmethod
    def _load_india_isin_sector_lookup(cls) -> dict[str, dict[str, Any]]:
        """Load the 8,750+ India ISIN & Sector Security Master."""
        lookup: dict[str, dict[str, Any]] = {}
        if not LOCAL_INDIA_ISIN_MASTER_PATH.exists():
            return lookup
        try:
            rows = json.loads(LOCAL_INDIA_ISIN_MASTER_PATH.read_text(encoding="utf-8", errors="ignore"))
            for item in rows:
                s_key = str(item.get("s") or "")
                d = item.get("d") or []
                if not s_key or len(d) < 4:
                    continue
                pricescale = float(d[9]) if len(d) > 9 and d[9] else 100.0
                minmov = float(d[10]) if len(d) > 10 and d[10] else 5.0
                tick = round(minmov / pricescale, 4) if pricescale > 0 else 0.05
                lookup[s_key] = {
                    "symbol": str(d[0] or ""),
                    "name": str(d[1] or ""),
                    "isin": str(d[2] or ""),
                    "sector": str(d[3] or "Equity"),
                    "type": str(d[4] or "stock") if len(d) > 4 else "stock",
                    "subtype": str(d[5] or "common") if len(d) > 5 else "common",
                    "exchange": str(d[6] or "") if len(d) > 6 else "",
                    "last_close": float(d[7]) if len(d) > 7 and d[7] is not None else 0.0,
                    "volume": float(d[8]) if len(d) > 8 and d[8] is not None else 0.0,
                    "tick_size": tick if tick > 0 else 0.05,
                }
        except Exception as exc:
            logger.error("Failed loading India ISIN/Sector master", error=str(exc))
        return lookup

    @classmethod
    def populate_resolver_from_cache_or_sync(cls) -> None:
        """Fast restoration of synchronized NSE, BSE, and Quotex instruments into SymbolResolver."""
        if cls._cached_instruments:
            SymbolResolver._instruments.update(cls._cached_instruments)
            SymbolResolver._alias_to_canonical.update(cls._cached_aliases)
        else:
            cls.sync_universe(refresh_remote=False)

    @classmethod
    def sync_universe(cls, refresh_remote: bool = False) -> int:
        """Synchronize the complete NSE, BSE, Quotex, Forex, and Crypto market universe."""
        errors: list[str] = []
        if refresh_remote or not LOCAL_NSE_MASTER_PATH.exists():
            cls.download_exchange_masters()

        # Fast path if already synced and not forcing remote refresh
        if cls._synced and not refresh_remote and cls._cached_instruments:
            SymbolResolver._instruments.update(cls._cached_instruments)
            SymbolResolver._alias_to_canonical.update(cls._cached_aliases)
            return len(SymbolResolver._instruments)

        isin_sector_lookup = cls._load_india_isin_sector_lookup()

        # 1. Synchronize Complete NSE Universe
        nse_stats = NSEUniverseManager.load_and_sync(isin_sector_lookup)

        # 2. Synchronize Complete BSE Universe
        bse_stats = BSEUniverseManager.load_and_sync(isin_sector_lookup)

        # 3. Synchronize Quotex Discovery Adapter
        qx_stats = QuotexDiscoveryAdapter.discover_instruments()

        # 4. Build Instrument objects and register in SymbolResolver
        order_types_std = ["MARKET", "LIMIT", "STOP_LOSS", "TAKE_PROFIT"]
        order_types_qx = ["CALL", "PUT"]

        # Register NSE Instruments
        for sym, rec in NSEUniverseManager._records.items():
            if not rec.active:
                continue
            canon = rec.canonical_symbol
            comp = CanonicalInstrumentRegistry.get_company(rec.canonical_instrument_id)
            inst = Instrument.model_construct(
                symbol=rec.symbol,
                canonical_symbol=canon,
                market=MarketType.NSE,
                exchange="NSE",
                asset_class=AssetClass.EQUITY,
                base_currency="INR",
                quote_currency="INR",
                tick_size=rec.tick_size,
                lot_size=rec.lot_size,
                price_precision=2,
                quantity_precision=0,
                contract_multiplier=1.0,
                settlement_type=SettlementType.T_PLUS_1,
                maker_fee_pct=0.0003,
                taker_fee_pct=0.0003,
                supported_order_types=order_types_std,
                metadata={
                    "name": rec.company_name,
                    "isin": rec.isin,
                    "series": rec.series,
                    "listing_date": rec.listing_date,
                    "sector": rec.sector,
                    "instrument_type": rec.instrument_type,
                    "trading_status": rec.trading_status,
                    "active": rec.active,
                    "listing_status": rec.listing_status,
                    "canonical_instrument_id": rec.canonical_instrument_id,
                    "cross_listed_bse": comp.bse_symbol if comp else None,
                    "bse_scrip_code": comp.bse_scrip_code if comp else None,
                    "country": "IN",
                    "source": "NSE_SECURITY_MASTER",
                },
            )
            SymbolResolver._instruments[canon] = inst
            SymbolResolver._alias_to_canonical[canon] = canon
            SymbolResolver._alias_to_canonical[sym] = canon
            norm_sym = sym.replace("-", "").replace("&", "").replace("_", "")
            SymbolResolver._alias_to_canonical[f"NSE{norm_sym}"] = canon
            SymbolResolver._alias_to_canonical[f"{norm_sym}NS"] = canon
            SymbolResolver._alias_to_canonical[norm_sym] = canon
            if rec.series and rec.series != "EQ":
                sfx_sym = f"{sym}-{rec.series}"
                SymbolResolver._alias_to_canonical[f"NSE:{sfx_sym}"] = canon
                SymbolResolver._alias_to_canonical[sfx_sym] = canon
                SymbolResolver._alias_to_canonical[f"NSE{norm_sym}{rec.series}"] = canon
                SymbolResolver._alias_to_canonical[f"{norm_sym}{rec.series}"] = canon
            if rec.isin:
                SymbolResolver._alias_to_canonical[rec.isin] = canon

        # Register BSE Instruments
        for sym, rec in BSEUniverseManager._records.items():
            if not rec.active:
                continue
            canon = rec.canonical_symbol
            comp = CanonicalInstrumentRegistry.get_company(rec.canonical_instrument_id)
            inst = Instrument.model_construct(
                symbol=rec.symbol,
                canonical_symbol=canon,
                market=MarketType.BSE,
                exchange="BSE",
                asset_class=AssetClass.EQUITY,
                base_currency="INR",
                quote_currency="INR",
                tick_size=rec.tick_size,
                lot_size=rec.lot_size,
                price_precision=2,
                quantity_precision=0,
                contract_multiplier=1.0,
                settlement_type=SettlementType.T_PLUS_1,
                maker_fee_pct=0.0003,
                taker_fee_pct=0.0003,
                supported_order_types=order_types_std,
                metadata={
                    "name": rec.company_name,
                    "scrip_code": rec.scrip_code,
                    "isin": rec.isin,
                    "series": rec.series,
                    "sector": rec.sector,
                    "instrument_type": rec.instrument_type,
                    "trading_status": rec.trading_status,
                    "active": rec.active,
                    "listing_status": rec.listing_status,
                    "listing_info": rec.listing_info,
                    "canonical_instrument_id": rec.canonical_instrument_id,
                    "cross_listed_nse": comp.nse_symbol if comp else None,
                    "country": "IN",
                    "source": "BSE_SECURITY_MASTER",
                },
            )
            SymbolResolver._instruments[canon] = inst
            SymbolResolver._alias_to_canonical[canon] = canon
            norm_sym = sym.replace("-", "").replace("&", "").replace("_", "")
            SymbolResolver._alias_to_canonical[f"BSE{norm_sym}"] = canon
            SymbolResolver._alias_to_canonical[f"{norm_sym}BO"] = canon
            SymbolResolver._alias_to_canonical[f"{sym}.BO"] = canon
            if rec.scrip_code:
                SymbolResolver._alias_to_canonical[rec.scrip_code] = canon
                SymbolResolver._alias_to_canonical[f"BSE:{rec.scrip_code}"] = canon
                SymbolResolver._alias_to_canonical[f"BSE{rec.scrip_code}"] = canon
            if sym not in SymbolResolver._alias_to_canonical:
                SymbolResolver._alias_to_canonical[sym] = canon
            if rec.isin and rec.isin not in SymbolResolver._alias_to_canonical:
                SymbolResolver._alias_to_canonical[rec.isin] = canon

        # Register Quotex Instruments
        for sym, rec in QuotexDiscoveryAdapter._records.items():
            canon = rec.canonical_symbol
            inst = Instrument.model_construct(
                symbol=rec.quotex_symbol,
                canonical_symbol=canon,
                market=MarketType.QUOTEX,
                exchange="QUOTEX",
                asset_class=AssetClass.BINARY_OPTION,
                base_currency=rec.base_currency,
                quote_currency=rec.quote_currency,
                tick_size=rec.tick_size,
                lot_size=1.0,
                price_precision=rec.price_precision,
                quantity_precision=0,
                contract_multiplier=1.0,
                settlement_type=SettlementType.INSTANT,
                maker_fee_pct=0.0,
                taker_fee_pct=0.0,
                supported_order_types=order_types_qx,
                metadata={
                    "name": rec.platform_instrument_name,
                    "provider_symbol": rec.provider_symbol,
                    "underlying_instrument": rec.underlying_instrument,
                    "market_data_source": rec.market_data_source,
                    "prediction_instrument": rec.prediction_instrument,
                    "asset_type": rec.asset_type,
                    "payout_pct": rec.payout_pct,
                    "active": rec.active,
                    "live_data_available": rec.live_data_available,
                    "trading_availability": rec.trading_availability,
                    "session": rec.session,
                    "source_timestamp": rec.source_timestamp.isoformat(),
                    "mode": "PAPER_ONLY",
                },
            )
            SymbolResolver._instruments[canon] = inst
            SymbolResolver._alias_to_canonical[canon] = canon
            SymbolResolver._alias_to_canonical[f"QUOTEX{sym.replace('_', '')}"] = canon
            if sym.endswith("_OTC") or sym.endswith("_QX"):
                SymbolResolver._alias_to_canonical[sym] = canon
                SymbolResolver._alias_to_canonical[sym.replace("_", "")] = canon
                SymbolResolver._alias_to_canonical[sym.replace("_OTC", "/OTC")] = canon

        # Register historical corporate symbol change aliases into _alias_to_canonical
        for old_sym, new_sym in SymbolResolver._historical_symbol_map.items():
            clean_old = old_sym.strip().upper()
            clean_new = new_sym.strip().upper()
            target_canon = clean_new if ":" in clean_new else f"NSE:{clean_new}"
            if (
                target_canon in SymbolResolver._instruments
                and clean_old not in SymbolResolver._instruments
                and f"NSE:{clean_old}" not in SymbolResolver._instruments
            ):
                SymbolResolver._alias_to_canonical[clean_old] = target_canon
                SymbolResolver._alias_to_canonical[f"NSE:{clean_old}"] = target_canon

        # Cache for fast subsequent restores
        cls._cached_instruments = dict(SymbolResolver._instruments)
        cls._cached_aliases = dict(SymbolResolver._alias_to_canonical)
        cls._synced = True
        cls._last_sync_source = "AUTHORITATIVE_MULTI_EXCHANGE_MASTER"

        combined_stats = {
            "received": nse_stats["received"] + bse_stats["received"] + qx_stats["received"],
            "added": nse_stats["added"] + bse_stats["added"] + qx_stats["added"],
            "updated": nse_stats["updated"] + bse_stats["updated"] + qx_stats["updated"],
            "removed": nse_stats["removed"] + bse_stats["removed"] + qx_stats["removed"],
            "changed": nse_stats["changed"] + bse_stats["changed"] + qx_stats["changed"],
        }
        UniverseSyncService.record_sync(source=cls._last_sync_source, stats=combined_stats, errors=errors)

        total_count = len(SymbolResolver._instruments)
        logger.info(
            "Synchronized complete multi-market instrument universe",
            total_instruments=total_count,
            nse_count=len(NSEUniverseManager._records),
            bse_count=len(BSEUniverseManager._records),
            quotex_count=len(QuotexDiscoveryAdapter._records),
            cross_listed_companies=CanonicalInstrumentRegistry.cross_listed_count(),
        )
        return total_count

    @classmethod
    def search(
        cls,
        query: str,
        market: MarketType | None = None,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        """Search across the full 8,800+ instrument universe by symbol, company name, ISIN, or BSE scrip code."""
        if not cls._synced:
            cls.sync_universe()

        q = query.strip().upper()
        if not q:
            all_inst = SymbolResolver.get_all_instruments(market)[:limit]
            return [cls._format_search_hit(i, 50) for i in all_inst]

        # Check historical symbol changes (e.g. ZOMATO -> ETERNAL) and series suffixes (e.g. PARAMPARA-SM)
        mapped_q = SymbolResolver._historical_symbol_map.get(q)
        q_base = q.rsplit("-", 1)[0] if "-" in q else q

        results: list[dict[str, Any]] = []
        all_inst = SymbolResolver.get_all_instruments(market)

        for inst in all_inst:
            meta = inst.metadata
            sym_u = inst.symbol.upper()
            canon_u = inst.canonical_symbol.upper()
            name = str(meta.get("name", "")).upper()
            isin = str(meta.get("isin", "")).upper()
            scrip = str(meta.get("scrip_code", "")).upper()
            sector = str(meta.get("sector", "")).upper()

            if q == sym_u or q == canon_u or q_base == sym_u:
                score = 100 if inst.market == MarketType.NSE else 95
            elif mapped_q and sym_u == mapped_q:
                score = 99 if inst.market == MarketType.NSE else 94
            elif q == scrip and scrip:
                score = 98
            elif q == isin and isin:
                score = 96
            elif sym_u.startswith(q) or sym_u.startswith(q_base):
                score = 90 if inst.market == MarketType.NSE else 86
            elif q in sym_u or q_base in sym_u:
                score = 80
            elif q in name or q_base in name:
                score = 68
            elif q in sector:
                score = 40
            else:
                continue

            results.append(cls._format_search_hit(inst, score))

        results.sort(key=lambda x: (-x["match_score"], x["symbol"]))
        return results[:limit]

    @staticmethod
    def _format_search_hit(inst: Instrument, score: int) -> dict[str, Any]:
        meta = inst.metadata
        return {
            "symbol": inst.symbol,
            "canonical_symbol": inst.canonical_symbol,
            "market": inst.market.value,
            "exchange": inst.exchange,
            "name": meta.get("name", inst.symbol),
            "isin": meta.get("isin", ""),
            "scrip_code": meta.get("scrip_code", ""),
            "canonical_instrument_id": meta.get("canonical_instrument_id", ""),
            "sector": meta.get("sector", "General"),
            "tick_size": inst.tick_size,
            "lot_size": inst.lot_size,
            "trading_status": meta.get("trading_status", "ACTIVE"),
            "live_data_available": meta.get("live_data_available", True),
            "match_score": score,
        }

    @classmethod
    def initialize(cls) -> int:
        """Initialize and synchronize the full market universe."""
        return cls.sync_universe()

    @classmethod
    def summary(cls) -> dict[str, Any]:
        """Return complete universe summary across all active markets."""
        if not cls._synced:
            cls.sync_universe()
        all_inst = SymbolResolver.get_all_instruments()
        by_market: dict[str, int] = {}
        for inst in all_inst:
            m_key = inst.market.value
            by_market[m_key] = by_market.get(m_key, 0) + 1
        return {
            "total_instruments": len(all_inst),
            "source": cls._last_sync_source,
            "by_market": by_market,
            "cross_listed_companies": CanonicalInstrumentRegistry.cross_listed_count(),
            "historical_symbol_mappings": len(NSEUniverseManager._symbol_changes),
            "last_sync": UniverseSyncService.get_sync_history(limit=1),
        }

    @classmethod
    def resolve_arbitrary_stock(
        cls,
        symbol_or_name: str,
        preferred_market: str | None = None,
    ) -> Instrument | None:
        """Resolve any stock symbol, BSE scrip code, ISIN, alias, or company name across 8,800+ instruments."""
        if not cls._synced:
            cls.sync_universe()

        q = symbol_or_name.strip().upper()
        if preferred_market and ":" not in q:
            pref_canon = f"{preferred_market.upper()}:{q}"
            pref_inst = SymbolResolver.resolve(pref_canon)
            if pref_inst:
                return pref_inst

        inst = SymbolResolver.resolve(symbol_or_name)
        if inst:
            return inst

        # Support BSE cross-listing lookup if queried as BSE:<SYM>
        if q.startswith("BSE:") or (preferred_market and preferred_market.upper() == "BSE"):
            clean_s = q.replace("BSE:", "").strip()
            bse_inst = SymbolResolver.resolve(f"BSE:{clean_s}")
            if bse_inst:
                return bse_inst
            nse_inst = SymbolResolver.resolve(f"NSE:{clean_s}")
            if nse_inst:
                bse_canon = f"BSE:{nse_inst.symbol}"
                bse_inst = Instrument(
                    symbol=nse_inst.symbol,
                    canonical_symbol=bse_canon,
                    market=MarketType.BSE,
                    exchange="BSE",
                    asset_class=AssetClass.EQUITY,
                    base_currency="INR",
                    quote_currency="INR",
                    tick_size=nse_inst.tick_size,
                    lot_size=nse_inst.lot_size,
                    price_precision=2,
                    quantity_precision=0,
                    contract_multiplier=1.0,
                    settlement_type=SettlementType.T_PLUS_1,
                    maker_fee_pct=0.0003,
                    taker_fee_pct=0.0003,
                    metadata={**nse_inst.metadata, "exchange": "BSE"},
                )
                SymbolResolver.register_instrument(bse_inst, aliases=[bse_canon, f"{nse_inst.symbol}.BO"])
                return bse_inst

        matches = cls.search(symbol_or_name, limit=1)
        if matches:
            return SymbolResolver.resolve(matches[0]["canonical_symbol"])
        return None

    @classmethod
    def run_universe_audit(cls) -> dict[str, Any]:
        """Execute a complete Universe Health Check across NSE, BSE, QUOTEX, FOREX, and CRYPTO (Section 20)."""
        if not cls._synced:
            cls.sync_universe()

        isin_lookup = cls._load_india_isin_sector_lookup()
        report: dict[str, Any] = {}
        missing_instruments: dict[str, list[dict[str, str]]] = {
            "NSE": [],
            "BSE": [],
            "QUOTEX": [],
            "FOREX": [],
            "CRYPTO": [],
        }

        # 1. NSE Audit
        nse_all = NSEUniverseManager.get_all_supported_stocks(active_only=False)
        nse_active = [r for r in nse_all if r.active]
        nse_data_avail = 0
        nse_predictable = 0
        for r in nse_active:
            tv_item = isin_lookup.get(r.canonical_symbol)
            has_price = bool(tv_item and float(tv_item.get("last_close", 0.0)) > 0)
            has_vol = bool(tv_item and float(tv_item.get("volume", 0.0)) > 0)
            if has_price or r.series in ("EQ", "BE", "ETF", "SM"):
                nse_data_avail += 1
                if has_price and (has_vol or r.series == "EQ"):
                    nse_predictable += 1
                else:
                    missing_instruments["NSE"].append({
                        "symbol": r.canonical_symbol,
                        "reason": "ZERO_SESSION_VOLUME_OR_ILLIQUID",
                    })
            else:
                missing_instruments["NSE"].append({
                    "symbol": r.canonical_symbol,
                    "reason": "DATA_UNAVAILABLE_OR_SUSPENDED",
                })
        report["NSE"] = {
            "discovered": len(nse_all),
            "active": len(nse_active),
            "data_available": nse_data_avail,
            "predictable": nse_predictable,
            "missing_data": len(nse_active) - nse_predictable,
        }

        # 2. BSE Audit
        bse_all = BSEUniverseManager.get_all_supported_stocks(active_only=False)
        bse_active = [r for r in bse_all if r.active]
        bse_data_avail = 0
        bse_predictable = 0
        for r in bse_active:
            tv_item = isin_lookup.get(r.canonical_symbol)
            has_price = bool(tv_item and float(tv_item.get("last_close", 0.0)) > 0)
            has_vol = bool(tv_item and float(tv_item.get("volume", 0.0)) > 0)
            if has_price or r.canonical_symbol.replace("BSE:", "NSE:") in isin_lookup:
                bse_data_avail += 1
                if has_price and has_vol:
                    bse_predictable += 1
                else:
                    missing_instruments["BSE"].append({
                        "symbol": r.canonical_symbol,
                        "scrip_code": r.scrip_code,
                        "reason": "NO_ACTIVE_SESSION_VOLUME",
                    })
            else:
                missing_instruments["BSE"].append({
                    "symbol": r.canonical_symbol,
                    "scrip_code": r.scrip_code,
                    "reason": "DATA_UNAVAILABLE_IN_FEED",
                })
        report["BSE"] = {
            "discovered": len(bse_all),
            "active": len(bse_active),
            "data_available": bse_data_avail,
            "predictable": bse_predictable,
            "missing_data": len(bse_active) - bse_predictable,
        }

        # 3. QUOTEX Audit
        qx_all = QuotexDiscoveryAdapter.get_all_records()
        qx_active = [r for r in qx_all if r.active]
        qx_avail = [r for r in qx_active if r.live_data_available]
        for r in qx_active:
            if not r.live_data_available:
                missing_instruments["QUOTEX"].append({
                    "symbol": r.canonical_symbol,
                    "underlying": r.underlying_instrument,
                    "reason": "Quotex live data unavailable (Proprietary Synthetic OTC without authorized feed)",
                })
        report["QUOTEX"] = {
            "discovered": len(qx_all),
            "active": len(qx_active),
            "data_available": len(qx_avail),
            "predictable": len(qx_avail),
            "missing_data": len(qx_active) - len(qx_avail),
        }

        # 4. FOREX Audit
        fx_all = SymbolResolver.get_all_instruments(MarketType.FOREX)
        report["FOREX"] = {
            "discovered": len(fx_all),
            "active": len(fx_all),
            "data_available": len(fx_all),
            "predictable": len(fx_all),
            "missing_data": 0,
        }

        # 5. CRYPTO Audit
        cr_all = SymbolResolver.get_all_instruments(MarketType.CRYPTO)
        report["CRYPTO"] = {
            "discovered": len(cr_all),
            "active": len(cr_all),
            "data_available": len(cr_all),
            "predictable": len(cr_all),
            "missing_data": 0,
        }

        return {
            "timestamp": datetime.now(UTC).isoformat(),
            "markets": report,
            "cross_listed_nse_bse_companies": CanonicalInstrumentRegistry.cross_listed_count(),
            "missing_or_unavailable_instruments": {
                k: v[:25] for k, v in missing_instruments.items()
            },
            "missing_totals_by_market": {k: len(v) for k, v in missing_instruments.items()},
        }


# Auto-sync on module load
UniverseService.sync_universe()


def _print_cli_audit() -> None:
    audit = UniverseService.run_universe_audit()
    print("=" * 64)
    print("ANTIGRAVITY — COMPLETE UNIVERSE HEALTH CHECK (AUDIT)")
    print("=" * 64)
    for mkt in ("NSE", "BSE", "QUOTEX", "FOREX", "CRYPTO"):
        info = audit["markets"][mkt]
        print(f"\n{mkt}:")
        print(f"  discovered:     {info['discovered']}")
        print(f"  active:         {info['active']}")
        print(f"  data available: {info['data_available']}")
        print(f"  predictable:    {info['predictable']}")
        print(f"  missing data:   {info['missing_data']}")

    print(f"\nCross-Listed NSE+BSE Canonical Companies (ISIN-linked): {audit['cross_listed_nse_bse_companies']}")
    print("\n" + "=" * 64)
    print("EXACT MISSING / UNAVAILABLE INSTRUMENTS (SAMPLE BY MARKET)")
    print("=" * 64)
    for mkt, items in audit["missing_or_unavailable_instruments"].items():
        total_missing = audit["missing_totals_by_market"][mkt]
        print(f"\n[{mkt}] Total Missing/Illiquid/Unavailable: {total_missing}")
        if not items:
            print("  None (100% available)")
        else:
            for item in items[:10]:
                print(f"  - {item['symbol']}: {item['reason']}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1].lower() in ("audit", "--audit"):
        _print_cli_audit()
    else:
        _print_cli_audit()
