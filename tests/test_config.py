"""Tests for system configuration engine."""

from core.config import Settings


def test_settings_defaults():
    settings = Settings()
    assert settings.ENVIRONMENT in ("development", "test", "staging", "production")
    assert "BTC-USD" in settings.DEFAULT_SYMBOLS
    assert "1m" in settings.ENABLED_TIMEFRAMES
    assert settings.STALE_DATA_THRESHOLD_SECONDS > 0
    assert settings.MAX_RISK_PER_TRADE_PCT == 0.01


def test_settings_custom_symbols_parsing():
    settings = Settings(DEFAULT_SYMBOLS="btc-usd, eth-usd, sol-usd")
    assert settings.DEFAULT_SYMBOLS == ["BTC-USD", "ETH-USD", "SOL-USD"]


def test_settings_timeframes_parsing():
    settings = Settings(ENABLED_TIMEFRAMES="1m, 5m, 15m, 1h")
    assert settings.ENABLED_TIMEFRAMES == ["1m", "5m", "15m", "1h"]
