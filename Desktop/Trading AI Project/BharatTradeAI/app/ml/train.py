"""Chronological classical-model training and probability calibration."""

from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any


def _estimator(model_name: str, random_state: int = 42) -> Any:
    try:
        from sklearn.dummy import DummyClassifier
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.linear_model import LogisticRegression
    except ImportError as exc:
        raise RuntimeError("scikit-learn is required for model training") from exc
    if model_name == "baseline":
        return DummyClassifier(strategy="prior")
    if model_name == "logistic":
        return LogisticRegression(max_iter=1000, random_state=random_state)
    if model_name == "random_forest":
        return RandomForestClassifier(n_estimators=200, min_samples_leaf=2, random_state=random_state, n_jobs=-1)
    if model_name == "xgboost":
        try:
            from importlib import import_module
            XGBClassifier = import_module("xgboost").XGBClassifier
        except ImportError as exc:
            raise RuntimeError("xgboost is not installed; install it to use model='xgboost'") from exc
        return XGBClassifier(
            n_estimators=120,
            max_depth=3,
            learning_rate=0.05,
            eval_metric="logloss",
            random_state=random_state,
            n_jobs=1,
            tree_method="hist",
        )
    if model_name == "lightgbm":
        try:
            from importlib import import_module
            LGBMClassifier = import_module("lightgbm").LGBMClassifier
        except ImportError as exc:
            raise RuntimeError("lightgbm is not installed; install it to use model='lightgbm'") from exc
        return LGBMClassifier(n_estimators=200, learning_rate=0.05, random_state=random_state, verbosity=-1)
    raise ValueError("model must be baseline, logistic, random_forest, xgboost, or lightgbm")


def _matrix(features: Sequence[dict[str, float]] | Sequence[Sequence[float]]) -> Any:
    import numpy as np
    if features and isinstance(features[0], Mapping):
        names = list(features[0])
        return np.asarray([[row[name] for name in names] for row in features], dtype=float)
    return np.asarray(features, dtype=float)


def train_model(
    features: Sequence[dict[str, float]] | Sequence[Sequence[float]],
    labels: Sequence[int],
    model: str = "baseline",
    validation_size: float = 0.2,
    calibrate: bool = True,
    feature_names: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Fit a model using a chronological holdout and optional calibration."""
    if not features or len(features) != len(labels):
        raise ValueError("features and labels must be non-empty and aligned")
    if not 0 <= validation_size < 1:
        raise ValueError("validation_size must be between 0 and 1")
    matrix = _matrix(features)
    split = len(labels) - int(len(labels) * validation_size) if validation_size else len(labels)
    if split < 2 or len(set(labels[:split])) < 2:
        raise ValueError("training window must contain at least two classes")
    estimator = _estimator(model)
    if calibrate and model != "baseline":
        from sklearn.calibration import CalibratedClassifierCV
        from sklearn.model_selection import TimeSeriesSplit
        minimum_class_count = min(Counter(labels[:split]).values())
        if minimum_class_count >= 2:
            estimator = CalibratedClassifierCV(
                estimator,
                cv=TimeSeriesSplit(n_splits=min(3, max(2, minimum_class_count))),
                method="sigmoid",
            )
    estimator.fit(matrix[:split], labels[:split])
    metrics: dict[str, float | None] = {"accuracy": None, "brier_score": None}
    if split < len(labels):
        from sklearn.metrics import accuracy_score, brier_score_loss
        probabilities = estimator.predict_proba(matrix[split:])[:, 1]
        metrics = {"accuracy": float(accuracy_score(labels[split:], estimator.predict(matrix[split:]))), "brier_score": float(brier_score_loss(labels[split:], probabilities))}
    return {"model": model, "estimator": estimator, "feature_names": list(feature_names or []), "metrics": metrics, "train_size": split, "validation_size": len(labels) - split}
