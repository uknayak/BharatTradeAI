"""Deterministic risk gate for the agentic pipeline."""

from typing import Any, Mapping


def evaluate_risk(state: Mapping[str, Any] | None = None) -> dict[str, Any]:
    if state is None:
        return {"agent": "risk_agent", "status": "ready"}
    prediction = state.get("ml", {}).get("prediction", {})
    probability = prediction.get("probability")
    approved = probability is None or 0.55 <= probability <= 0.95
    return {
        "agent": "risk_agent",
        "status": "ready",
        "approved": approved,
        "risk_level": "LOW" if approved else "HIGH",
        "reason": "prediction probability outside risk band" if not approved else "within configured probability band",
    }
