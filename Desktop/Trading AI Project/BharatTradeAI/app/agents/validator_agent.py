"""Final deterministic validation gate for agentic outputs."""

from typing import Any, Mapping

from app.agentic.scoring import score_signal


def validate_signal(state: Mapping[str, Any] | None = None) -> dict[str, Any]:
    if state is None:
        return {"agent": "validator_agent", "status": "ready"}
    risk = state.get("risk", {})
    errors = []
    if risk.get("approved") is False:
        errors.append(risk.get("reason", "risk rejected"))
    return {
        "agent": "validator_agent",
        "status": "ready",
        "valid": not errors,
        "errors": errors,
        **score_signal(state, state.get("signal_weights")),
    }
