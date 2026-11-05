import pytest

from app.agents.technical_agent import analyze_technicals
from app.indicators.technical import compute_technical_indicators


def _candles(count=21):
    return [
        {"high": 101 + index, "low": 99 + index, "close": 100 + index, "volume": 100 if index < count - 1 else 250}
        for index in range(count)
    ]


def test_technical_engine_calculates_all_requested_indicators():
    result = compute_technical_indicators(_candles(), period=5)

    assert result["rsi"][-1] == 100.0
    assert result["ema"][-1] == pytest.approx(118.0)
    assert result["macd"]["macd"][-1] is None or isinstance(result["macd"]["macd"][-1], float)
    assert result["vwap"][-1] is not None
    assert result["atr"][-1] == pytest.approx(2.0)
    assert result["adx"]["adx"][-1] is not None
    assert result["bollinger"]["upper"][-1] is not None
    assert result["supertrend"]["direction"][-1] == "bullish"
    assert result["support_resistance"]["support"][-1] == 115
    assert result["support_resistance"]["resistance"][-1] == 121
    assert result["volume_breakout"][-1] is True


def test_technical_agent_delegates_and_preserves_ready_response():
    assert analyze_technicals() == {"agent": "technical_agent", "status": "ready"}
    result = analyze_technicals(_candles(), period=5)
    assert result["status"] == "ready"
    assert result["analysis"]["ema"][-1] == pytest.approx(118.0)