"""Market regime agent backed by the regime classifier."""

from collections.abc import Iterable, Sequence
from typing import Any, Mapping

from app.indicators.regime import classify_market_regime


def analyze_market_regime(data: Iterable[Mapping[str, Any]] | Sequence[float] | None = None, period: int = 14) -> dict[str, Any]:
    if data is not None:
        return {"agent": "market_agent", "status": "ready", "analysis": classify_market_regime(data, period)}
    return {"agent": "market_agent", "status": "ready"}
