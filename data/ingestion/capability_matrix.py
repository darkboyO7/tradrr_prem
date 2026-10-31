"""Trading Intelligence Platform - Market Data Provider Capability Matrix.

Institutional reference and runtime inspection for supported exchange feeds,
data sources, interval coverage, session rules, and integration constraints across:
1. NSE (National Stock Exchange of India)
2. BSE (Bombay Stock Exchange)
3. Quotex (Binary Options Platform & Underlying Proxy Feeds)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from core.instrument import MarketType


@dataclass(frozen=True)
class ProviderCapability:
    """Capability specification for an exchange data provider."""

    market: MarketType
    exchange_name: str
    primary_data_source: str
    secondary_data_source: str
    direct_platform_api: bool
    is_underlying_proxy: bool
    supported_timeframes: list[str]
    native_session_timezone: str
    trading_hours: str
    is_24_7: bool
    status: str
    limitation_notes: str
    break_even_win_rate_pct: float | None = None
    transaction_cost_bps: float = 6.0


PROVIDER_CAPABILITY_MATRIX: dict[MarketType, ProviderCapability] = {
    MarketType.NSE: ProviderCapability(
        market=MarketType.NSE,
        exchange_name="National Stock Exchange of India (NSE)",
        primary_data_source="Groww NSE Cash API (REST / JSON)",
        secondary_data_source="TradingView India Scanner & Moneycontrol TechCharts",
        direct_platform_api=True,
        is_underlying_proxy=False,
        supported_timeframes=["1m", "5m", "15m", "1h", "1d"],
        native_session_timezone="Asia/Kolkata (IST, UTC+05:30)",
        trading_hours="09:15 - 15:30 IST (Mon-Fri)",
        is_24_7=False,
        status="VERIFIED",
        limitation_notes="Pre-market 09:00-09:15 IST and post-market 15:40-16:00 IST are not continuous trading. Indian statutory costs (STT, SEBI, GST, stamp duty) total ~6 bps.",
        break_even_win_rate_pct=50.06,  # 50% + friction
        transaction_cost_bps=6.0,
    ),
    MarketType.BSE: ProviderCapability(
        market=MarketType.BSE,
        exchange_name="Bombay Stock Exchange (BSE)",
        primary_data_source="Groww BSE Cash API by Scrip Code & Tradingsymbol",
        secondary_data_source="TradingView BSE India Scanner",
        direct_platform_api=True,
        is_underlying_proxy=False,
        supported_timeframes=["1m", "5m", "15m", "1h", "1d"],
        native_session_timezone="Asia/Kolkata (IST, UTC+05:30)",
        trading_hours="09:15 - 15:30 IST (Mon-Fri)",
        is_24_7=False,
        status="VERIFIED",
        limitation_notes="BSE order books operate with independent liquidity and scrip codes (5xxxxx). Under no circumstances are NSE prices, volume, or candles substituted for BSE.",
        break_even_win_rate_pct=50.06,
        transaction_cost_bps=6.0,
    ),
    MarketType.QUOTEX: ProviderCapability(
        market=MarketType.QUOTEX,
        exchange_name="Quotex Binary Options Platform",
        primary_data_source="Coinbase / Kraken Spot Public REST (Crypto Proxy)",
        secondary_data_source="None (Direct Quotex platform execution API is BLOCKED / Non-Existent)",
        direct_platform_api=False,
        is_underlying_proxy=True,
        supported_timeframes=["1m", "5m", "15m", "1h"],
        native_session_timezone="UTC",
        trading_hours="24/7 (Crypto) / Instrument Specific",
        is_24_7=True,
        status="VERIFIED_UNDERLYING_PROXY_ONLY",
        limitation_notes=(
            "Direct Quotex platform feed is BLOCKED (no public API). Underlying spot crypto feeds are authorized proxies. "
            "Proprietary synthetic broker OTC assets (MSFT_OTC, etc.) are strictly marked QUOTEX DATA UNAVAILABLE. "
            "Binary options contract: Win payout = +P*stake (avg 85%), Loss = -1.0*stake. Break-even win rate is 54.05%."
        ),
        break_even_win_rate_pct=54.05,  # 1 / (1 + 0.85) = 54.05%
        transaction_cost_bps=8.0,
    ),
}


def get_provider_capability(market: MarketType | str) -> ProviderCapability | None:
    """Retrieve capability specification for a target market."""
    if isinstance(market, str):
        market_str = market.upper().replace("MARKETTYPE.", "")
        try:
            m_enum = MarketType[market_str]
        except KeyError:
            return None
        return PROVIDER_CAPABILITY_MATRIX.get(m_enum)
    return PROVIDER_CAPABILITY_MATRIX.get(market)
