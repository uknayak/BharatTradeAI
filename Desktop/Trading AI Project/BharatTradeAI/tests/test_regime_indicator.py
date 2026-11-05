import pytest

from app.agents.market_agent import analyze_market_regime
from app.indicators.regime import classify_market_regime


def test_regime_identifies_bullish_high_vol_trend():
    prices = [100, 101, 100, 103, 102, 106, 105, 109, 108, 113, 112, 117, 116, 121, 120, 125]
    result = classify_market_regime(prices, period=5)

    assert result["direction"] == "BULLISH"
    assert result["volatility"] == "HIGH VOL"
    assert result["structure"] == "TRENDING"
    assert result["metrics"]["period_return"] == pytest.approx((125 / 112) - 1)


def test_regime_identifies_bearish_low_vol_trend_from_ohlcv():
    candles = [{"close": 100 - index, "high": 100 - index + 0.3, "low": 100 - index - 0.3} for index in range(16)]
    result = classify_market_regime(candles, period=5)

    assert result["direction"] == "BEARISH"
    assert result["volatility"] == "LOW VOL"
    assert result["structure"] == "TRENDING"


def test_regime_identifies_neutral_range_and_agent_delegates():
    prices = [100, 101, 99, 100, 101, 99, 100, 101, 99, 100, 101, 99, 100, 101, 99, 100]
    result = analyze_market_regime(prices, period=5)

    assert result["analysis"]["direction"] == "NEUTRAL"
    assert result["analysis"]["structure"] == "RANGE"
    assert analyze_market_regime() == {"agent": "market_agent", "status": "ready"}