"""Prediction helpers for trained classical models."""

from typing import Any, Sequence


def predict_signal(model_result: dict[str, Any] | None = None, features: Sequence[float] | None = None) -> dict[str, Any]:
    if model_result is None or features is None:
        return {"module": "predict", "status": "not_implemented"}
    estimator = model_result.get("estimator")
    if estimator is None:
        raise ValueError("model_result must contain a trained estimator")
    probability = float(estimator.predict_proba([features])[0][1])
    return {"signal": "BUY" if probability >= 0.5 else "SELL", "probability": probability, "model": model_result.get("model")}
