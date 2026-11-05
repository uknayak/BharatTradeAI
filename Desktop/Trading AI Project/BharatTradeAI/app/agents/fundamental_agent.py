"""Fundamental analysis agent backed by the fundamentals engine."""

from typing import Any, Mapping

from app.indicators.fundamental import analyze_fundamentals as calculate_fundamentals


def analyze_fundamentals(current: Mapping[str, Any] | None = None, previous: Mapping[str, Any] | None = None) -> dict[str, Any]:
    if current is not None:
        return {"agent": "fundamental_agent", "status": "ready", "analysis": calculate_fundamentals(current, previous)}
    return {"agent": "fundamental_agent", "status": "ready"}
