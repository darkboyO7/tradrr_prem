"""Trading Intelligence Platform - Symbol & Instrument Resolution Engine.

Resolves broker aliases, exchange tickers, ISINs, BSE scrip codes, historical symbol changes,
and user search queries to canonical instruments across NSE, BSE, Forex, Quotex, and Crypto markets.
"""

import re
from typing import ClassVar

from core.instrument import AssetClass, Instrument, MarketType, SettlementType, TradingSession
from core.logging import get_logger

logger = get_logger(__name__)


class SymbolResolver:
    """Central repository and resolver for canonical instruments and aliases across venues."""

    _instruments: ClassVar[dict[str, Instrument]] = {}
    _alias_to_canonical: ClassVar[dict[str, str]] = {}
    _sessions: ClassVar[dict[MarketType, TradingSession]] = {}
    _historical_symbol_map: ClassVar[dict[str, str]] = {
        # Authoritative NSE/BSE corporate symbol change history (old -> current)
        "ZOMATO": "ETERNAL",
        "NSE:ZOMATO": "NSE:ETERNAL",
        "BSE:ZOMATO": "BSE:ETERNAL",
        "MINDTREE": "LTM",
        "LTIM": "LTM",
        "INFOSYSTCH": "INFY",
        "TATAGLOBAL": "TATACONSUM",
        "MOTHERSUMI": "MOTHERSON",
        "WELSPUNIND": "WELSPUNLIV",
        "AMARAJABAT": "ARE&M",
        "SRTRANSFIN": "SHRIRAMFIN",
        "HDFC": "HDFCBANK",
        "CADILAHC": "ZYDUSLIFE",
        " Piramal": "PEL",
        "UTIBANK": "AXISBANK",
        "ICICI": "ICICIBANK",
        "BHARTI": "BHARTIARTL",
        "HEROHONDA": "HEROMOTOCO",
        "BAJAJAUTO": "BAJAJ-AUTO",
        "SATYAMCOMP": "TECHM",
        "SESAGOA": "VEDL",
        "STER": "VEDL",
        "RANBAXY": "SUNPHARMA",
    }

    @classmethod
    def _init_sessions(cls) -> None:
        """Configure authoritative exchange trading sessions."""
        cls._sessions[MarketType.NSE] = TradingSession(
            market=MarketType.NSE,
            timezone="Asia/Kolkata",
            start_time="09:15",
            end_time="15:30",
            trading_days=[0, 1, 2, 3, 4],
            is_24_7=False,
        )
        cls._sessions[MarketType.BSE] = TradingSession(
            market=MarketType.BSE,
            timezone="Asia/Kolkata",
            start_time="09:15",
            end_time="15:30",
            trading_days=[0, 1, 2, 3, 4],
            is_24_7=False,
        )
        cls._sessions[MarketType.FOREX] = TradingSession(
            market=MarketType.FOREX,
            timezone="America/New_York",
            start_time="17:00",  # Sunday 5pm to Friday 5pm EST
            end_time="17:00",
            trading_days=[0, 1, 2, 3, 4],
            is_24_7=False,
        )
        cls._sessions[MarketType.QUOTEX] = TradingSession(
            market=MarketType.QUOTEX,
            timezone="UTC",
            start_time="00:00",
            end_time="23:59",
            trading_days=[0, 1, 2, 3, 4, 5, 6],
            is_24_7=True,
        )
        cls._sessions[MarketType.CRYPTO] = TradingSession(
            market=MarketType.CRYPTO,
            timezone="UTC",
            start_time="00:00",
            end_time="23:59",
            trading_days=[0, 1, 2, 3, 4, 5, 6],
            is_24_7=True,
        )

    @classmethod
    def initialize_defaults(cls) -> None:
        """Initialize trading sessions and populate from the dynamic exchange universe cache."""
        cls._instruments.clear()
        cls._alias_to_canonical.clear()
        cls._sessions.clear()
        cls._init_sessions()

        # 1. Forex Institutional Universe (28 Majors, Crosses & EM Pairs)
        forex_pairs = [
            ("EURCHF", "EUR", "CHF", 0.00001, 1000.0, 5, "Euro / Swiss Franc"),
            ("EURUSD", "EUR", "USD", 0.00001, 1000.0, 5, "Euro / US Dollar"),
            ("GBPUSD", "GBP", "USD", 0.00001, 1000.0, 5, "British Pound / US Dollar"),
            ("USDJPY", "USD", "JPY", 0.001, 1000.0, 3, "US Dollar / Japanese Yen"),
            ("USDCHF", "USD", "CHF", 0.00001, 1000.0, 5, "US Dollar / Swiss Franc"),
            ("AUDUSD", "AUD", "USD", 0.00001, 1000.0, 5, "Australian Dollar / US Dollar"),
            ("USDCAD", "USD", "CAD", 0.00001, 1000.0, 5, "US Dollar / Canadian Dollar"),
            ("NZDUSD", "NZD", "USD", 0.00001, 1000.0, 5, "New Zealand Dollar / US Dollar"),
            ("EURGBP", "EUR", "GBP", 0.00001, 1000.0, 5, "Euro / British Pound"),
            ("EURJPY", "EUR", "JPY", 0.001, 1000.0, 3, "Euro / Japanese Yen"),
            ("GBPJPY", "GBP", "JPY", 0.001, 1000.0, 3, "British Pound / Japanese Yen"),
            ("CHFJPY", "CHF", "JPY", 0.001, 1000.0, 3, "Swiss Franc / Japanese Yen"),
            ("GBPCHF", "GBP", "CHF", 0.00001, 1000.0, 5, "British Pound / Swiss Franc"),
            ("AUDJPY", "AUD", "JPY", 0.001, 1000.0, 3, "Australian Dollar / Japanese Yen"),
            ("EURAUD", "EUR", "AUD", 0.00001, 1000.0, 5, "Euro / Australian Dollar"),
            ("EURCAD", "EUR", "CAD", 0.00001, 1000.0, 5, "Euro / Canadian Dollar"),
            ("GBPAUD", "GBP", "AUD", 0.00001, 1000.0, 5, "British Pound / Australian Dollar"),
            ("GBPCAD", "GBP", "CAD", 0.00001, 1000.0, 5, "British Pound / Canadian Dollar"),
            ("AUDCHF", "AUD", "CHF", 0.00001, 1000.0, 5, "Australian Dollar / Swiss Franc"),
            ("AUDCAD", "AUD", "CAD", 0.00001, 1000.0, 5, "Australian Dollar / Canadian Dollar"),
            ("AUDNZD", "AUD", "NZD", 0.00001, 1000.0, 5, "Australian Dollar / New Zealand Dollar"),
            ("EURNZD", "EUR", "NZD", 0.00001, 1000.0, 5, "Euro / New Zealand Dollar"),
            ("GBPNZD", "GBP", "NZD", 0.00001, 1000.0, 5, "British Pound / New Zealand Dollar"),
            ("NZDJPY", "NZD", "JPY", 0.001, 1000.0, 3, "New Zealand Dollar / Japanese Yen"),
            ("CADJPY", "CAD", "JPY", 0.001, 1000.0, 3, "Canadian Dollar / Japanese Yen"),
            ("CADCHF", "CAD", "CHF", 0.00001, 1000.0, 5, "Canadian Dollar / Swiss Franc"),
            ("USDINR", "USD", "INR", 0.0025, 1000.0, 4, "US Dollar / Indian Rupee"),
            ("EURINR", "EUR", "INR", 0.0025, 1000.0, 4, "Euro / Indian Rupee"),
        ]
        for pair, base, quote, tick, lot, prec, name in forex_pairs:
            canon = f"FX:{pair}"
            inst = Instrument(
                symbol=pair,
                canonical_symbol=canon,
                market=MarketType.FOREX,
                exchange="FOREX",
                asset_class=AssetClass.FOREX,
                base_currency=base,
                quote_currency=quote,
                tick_size=tick,
                lot_size=lot,
                price_precision=prec,
                quantity_precision=0,
                contract_multiplier=1.0,
                settlement_type=SettlementType.T_PLUS_2,
                maker_fee_pct=0.00002,
                taker_fee_pct=0.00002,
                metadata={
                    "name": name,
                    "pip_size": tick * 10 if prec in (3, 5) else tick,
                    "active": True,
                    "trading_status": "ACTIVE",
                    "listing_status": "LISTED",
                    "source": "FOREX_INTERBANK_MASTER",
                },
            )
            cls.register_instrument(
                inst,
                aliases=[
                    canon,
                    pair,
                    f"{base}/{quote}",
                    f"{base}-{quote}",
                    f"FX_{pair}",
                    f"OANDA:{pair}",
                ],
            )

        # 2. Crypto Spot & Perp Universe
        crypto_pairs = [
            ("BTC-USD", "BTC", "USD", 0.01, 0.00001, 2, 5),
            ("ETH-USD", "ETH", "USD", 0.01, 0.0001, 2, 4),
            ("SOL-USD", "SOL", "USD", 0.01, 0.01, 2, 2),
            ("BNB-USD", "BNB", "USD", 0.01, 0.01, 2, 2),
            ("XRP-USD", "XRP", "USD", 0.0001, 1.0, 4, 0),
            ("ADA-USD", "ADA", "USD", 0.0001, 1.0, 4, 0),
            ("DOGE-USD", "DOGE", "USD", 0.00001, 1.0, 5, 0),
            ("AVAX-USD", "AVAX", "USD", 0.01, 0.1, 2, 1),
            ("LINK-USD", "LINK", "USD", 0.001, 0.1, 3, 1),
            ("DOT-USD", "DOT", "USD", 0.001, 0.1, 3, 1),
            ("LTC-USD", "LTC", "USD", 0.01, 0.01, 2, 2),
            ("MATIC-USD", "MATIC", "USD", 0.0001, 1.0, 4, 0),
            ("UNI-USD", "UNI", "USD", 0.001, 0.1, 3, 1),
            ("BTCUSDT", "BTC", "USDT", 0.01, 0.00001, 2, 5),
            ("ETHUSDT", "ETH", "USDT", 0.01, 0.0001, 2, 4),
        ]
        for sym, base, quote, tick, lot, p_prec, q_prec in crypto_pairs:
            canon = f"CRYPTO:{sym}"
            inst = Instrument(
                symbol=sym,
                canonical_symbol=canon,
                market=MarketType.CRYPTO,
                exchange="CRYPTO",
                asset_class=AssetClass.CRYPTO,
                base_currency=base,
                quote_currency=quote,
                tick_size=tick,
                lot_size=lot,
                price_precision=p_prec,
                quantity_precision=q_prec,
                contract_multiplier=1.0,
                settlement_type=SettlementType.INSTANT,
                maker_fee_pct=0.001,
                taker_fee_pct=0.002,
                metadata={
                    "name": f"{base}/{quote} Crypto",
                    "active": True,
                    "trading_status": "ACTIVE",
                    "listing_status": "LISTED",
                    "source": "CRYPTO_EXCHANGE_MASTER",
                },
            )
            aliases = [canon, sym, sym.replace("-", "/"), sym.replace("-", "")]
            cls.register_instrument(inst, aliases=aliases)

        # 3. Restore or synchronize dynamic NSE, BSE, and Quotex universes via UniverseService
        try:
            from core.universe import UniverseService

            UniverseService.populate_resolver_from_cache_or_sync()
        except Exception as exc:
            logger.debug("UniverseService deferred initialization during bootstrap", error=str(exc))

    @classmethod
    def register_instrument(cls, instrument: Instrument, aliases: list[str] | None = None) -> None:
        """Register an instrument and its aliases."""
        cls._instruments[instrument.canonical_symbol] = instrument
        cls._alias_to_canonical[instrument.canonical_symbol.upper()] = instrument.canonical_symbol

        # Give priority to NSE for unqualified Indian equity symbols
        sym_key = instrument.symbol.upper()
        if sym_key not in cls._alias_to_canonical or instrument.market == MarketType.NSE:
            cls._alias_to_canonical[sym_key] = instrument.canonical_symbol

        if aliases:
            for alias in aliases:
                if not alias:
                    continue
                upper_alias = alias.upper()
                if upper_alias not in cls._alias_to_canonical or instrument.market == MarketType.NSE:
                    cls._alias_to_canonical[upper_alias] = instrument.canonical_symbol
                norm_key = cls._normalize_key(alias)
                if norm_key and (norm_key not in cls._alias_to_canonical or instrument.market == MarketType.NSE):
                    cls._alias_to_canonical[norm_key] = instrument.canonical_symbol

    @classmethod
    def _normalize_key(cls, key: str) -> str:
        """Strip delimiters for robust case-insensitive lookup."""
        return re.sub(r"[\s\-_/:]", "", key.upper())

    @classmethod
    def resolve(cls, query: str) -> Instrument | None:
        """Resolve any ticker string, alias, ISIN, BSE scrip code, or historical symbol to a canonical Instrument."""
        if not cls._instruments:
            cls.initialize_defaults()

        if not query:
            return None

        clean_q = query.strip()
        upper_query = clean_q.upper()

        # Direct canonical lookup
        if upper_query in cls._instruments:
            return cls._instruments[upper_query]
        if clean_q in cls._instruments:
            return cls._instruments[clean_q]

        # Exact upper alias lookup
        if upper_query in cls._alias_to_canonical:
            canon = cls._alias_to_canonical[upper_query]
            if canon in cls._instruments:
                return cls._instruments[canon]

        # Normalized key lookup
        norm_query = cls._normalize_key(clean_q)
        if norm_query in cls._alias_to_canonical:
            canon = cls._alias_to_canonical[norm_query]
            if canon in cls._instruments:
                return cls._instruments[canon]

        # Check historical corporate symbol changes (e.g. ZOMATO -> ETERNAL, MINDTREE -> LTIM)
        if upper_query in cls._historical_symbol_map:
            mapped = cls._historical_symbol_map[upper_query]
            if mapped in cls._instruments:
                return cls._instruments[mapped]
            if mapped in cls._alias_to_canonical:
                return cls._instruments.get(cls._alias_to_canonical[mapped])

        # Fallback scan
        for inst in cls._instruments.values():
            if inst.symbol.upper() == upper_query or inst.symbol.replace("-", "").upper() == norm_query:
                return inst

        return None

    @classmethod
    def get_session(cls, market: MarketType) -> TradingSession | None:
        """Get the trading session definition for a given market."""
        if not cls._sessions:
            cls._init_sessions()
        return cls._sessions.get(market)

    @classmethod
    def get_all_instruments(cls, market: MarketType | None = None) -> list[Instrument]:
        """List all registered instruments, optionally filtered by market segment."""
        if not cls._instruments:
            cls.initialize_defaults()
        if market is None:
            return list(cls._instruments.values())
        return [inst for inst in cls._instruments.values() if inst.market == market]


# Auto-initialize on module load
SymbolResolver.initialize_defaults()
