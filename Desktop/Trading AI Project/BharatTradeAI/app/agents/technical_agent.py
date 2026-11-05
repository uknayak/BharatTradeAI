"""Technical analysis agent backed by the indicator engine."""

from collections.abc import Iterable
from typing import Any, Mapping

from app.indicators.technical import compute_technical_indicators


def analyze_technicals(candles: Iterable[Mapping[str, Any]] | None = None, period: int = 14) -> dict[str, Any]:
    if candles is not None:
        return {"agent": "technical_agent", "status": "ready", "analysis": compute_technical_indicators(candles, period)}
    return {"agent": "technical_agent", "status": "ready"}
