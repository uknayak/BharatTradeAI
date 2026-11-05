"""Machine learning agent backed by the classical prediction pipeline."""

from collections.abc import Sequence
from typing import Any

from app.ml.predict import predict_signal


def run_ml_prediction(model_result: dict[str, Any] | None = None, features: Sequence[float] | None = None) -> dict[str, Any]:
    if model_result is not None or features is not None:
        return {"agent": "ml_agent", "status": "ready", "prediction": predict_signal(model_result, features)}
    return {"agent": "ml_agent", "status": "ready"}
