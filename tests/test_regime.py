"""Tests for Market Regime Engine."""

import pandas as pd

from analysis.regime import (
    MarketRegimeDetector,
    TrendRegime,
    VolatilityRegime,
    VolumeRegime,
)


def test_classify_strong_bull_bar():
    detector = MarketRegimeDetector()
    row = pd.Series({
        "close": 110.0,
        "ema_20": 105.0,
        "ema_50": 100.0,
        "adx_14": 32.0,
        "plus_di_14": 28.0,
        "minus_di_14": 14.0,
        "volatility_ratio": 1.05,
        "bb_squeeze": 0,
        "rvol_20": 1.8,
    })

    state = detector.classify_bar(row)
    assert state.trend == TrendRegime.STRONG_BULL
    assert state.volatility == VolatilityRegime.NORMAL
    assert state.volume == VolumeRegime.VOLUME_EXPANSION
    assert state.confidence >= 0.85
    assert "STRONG_BULL" in state.regime_code


def test_classify_strong_bear_bar():
    detector = MarketRegimeDetector()
    row = pd.Series({
        "close": 90.0,
        "ema_20": 95.0,
        "ema_50": 100.0,
        "adx_14": 30.0,
        "plus_di_14": 12.0,
        "minus_di_14": 32.0,
        "volatility_ratio": 2.2,
        "bb_squeeze": 0,
        "rvol_20": 3.5,
    })

    state = detector.classify_bar(row)
    assert state.trend == TrendRegime.STRONG_BEAR
    assert state.volatility == VolatilityRegime.EXTREME
    assert state.volume == VolumeRegime.VOLUME_CLIMAX


def test_classify_ranging_compressed_bar():
    detector = MarketRegimeDetector()
    row = pd.Series({
        "close": 100.0,
        "ema_20": 100.0,
        "ema_50": 100.0,
        "adx_14": 12.0,
        "volatility_ratio": 0.60,
        "bb_squeeze": 1,
        "rvol_20": 0.5,
    })

    state = detector.classify_bar(row)
    assert state.trend == TrendRegime.RANGING
    assert state.volatility == VolatilityRegime.COMPRESSED
    assert state.volume == VolumeRegime.LOW_VOLUME


def test_classify_dataframe():
    detector = MarketRegimeDetector()
    # 50 rows synthetic
    df = pd.DataFrame({
        "close": [100.0 + i for i in range(50)],
        "open": [99.5 + i for i in range(50)],
        "high": [101.0 + i for i in range(50)],
        "low": [99.0 + i for i in range(50)],
        "volume": [100.0] * 50,
    })

    classified = detector.classify_dataframe(df)
    assert "trend_regime" in classified.columns
    assert "volatility_regime" in classified.columns
    assert "volume_regime" in classified.columns
    assert "regime_code" in classified.columns
    assert len(classified) == 50
