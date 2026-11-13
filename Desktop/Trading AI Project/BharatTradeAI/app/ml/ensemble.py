"""Chronological XGBoost/LSTM evaluation, calibration, and model selection."""

from collections.abc import Sequence
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, f1_score, precision_score, recall_score

from app.ml.lstm import (
    build_ohlcv_sequences,
    predict_lstm_probabilities,
    train_lstm_model,
)
from app.ml.features import NEWS_DIRECTION_FEATURE_NAMES
from app.ml.train import _estimator, _matrix

MIN_SAMPLES = 200
MIN_TRAIN_SAMPLES = 120
MIN_CALIBRATION_SAMPLES = 30
MIN_SELECTION_SAMPLES = 20
MIN_TEST_SAMPLES = 30
MIN_NEWS_EVALUATION_SAMPLES = 30
SEQUENCE_LENGTH = 30


def _split_sizes(sample_count: int) -> tuple[int, int, int]:
    train_end = int(sample_count * 0.60)
    calibration_end = int(sample_count * 0.75)
    selection_end = int(sample_count * 0.85)
    return train_end, calibration_end, selection_end


def _fit_probability_calibrator(
    probabilities: Sequence[float], labels: Sequence[int]
) -> LogisticRegression:
    values = np.clip(np.asarray(probabilities, dtype=float), 1e-6, 1 - 1e-6)
    logits = np.log(values / (1 - values)).reshape(-1, 1)
    calibrator = LogisticRegression(solver="lbfgs", max_iter=500)
    calibrator.fit(logits, np.asarray(labels, dtype=int))
    return calibrator


def _calibrate(calibrator: LogisticRegression, probabilities: Sequence[float]) -> np.ndarray:
    values = np.clip(np.asarray(probabilities, dtype=float), 1e-6, 1 - 1e-6)
    logits = np.log(values / (1 - values)).reshape(-1, 1)
    return calibrator.predict_proba(logits)[:, 1]


def _metrics(labels: Sequence[int], probabilities: Sequence[float]) -> dict[str, float | int | None]:
    actual = np.asarray(labels, dtype=int)
    scores = np.asarray(probabilities, dtype=float)
    predicted = (scores >= 0.5).astype(int)
    calibration_error = 0.0
    for lower, upper in zip(np.linspace(0, 0.9, 10), np.linspace(0.1, 1, 10)):
        in_bin = (scores >= lower) & (scores < upper if upper < 1 else scores <= upper)
        if np.any(in_bin):
            calibration_error += float(np.mean(in_bin)) * abs(
                float(np.mean(scores[in_bin])) - float(np.mean(actual[in_bin]))
            )
    return {
        "sample_count": len(actual),
        "precision": float(precision_score(actual, predicted, zero_division=0)),
        "recall": float(recall_score(actual, predicted, zero_division=0)),
        "f1": float(f1_score(actual, predicted, zero_division=0)),
        "brier_score": float(brier_score_loss(actual, scores)),
        "expected_calibration_error": calibration_error,
    }


def _fit_walk_forward_xgboost(
    matrix: Any,
    labels: Sequence[int],
    *,
    start_index: int,
    step: int = 60,
) -> dict[str, Any]:
    actual: list[int] = []
    predicted_probabilities: list[float] = []
    for block_start in range(start_index, len(labels), step):
        block_end = min(block_start + step, len(labels))
        historical_end = int(block_start * 0.85)
        if historical_end < MIN_TRAIN_SAMPLES or block_start - historical_end < MIN_CALIBRATION_SAMPLES:
            continue
        estimator = _estimator("xgboost")
        estimator.fit(matrix[:historical_end], labels[:historical_end])
        calibration_raw = estimator.predict_proba(matrix[historical_end:block_start])[:, 1]
        calibration_labels = labels[historical_end:block_start]
        if len(set(calibration_labels)) < 2:
            continue
        calibrator = _fit_probability_calibrator(calibration_raw, calibration_labels)
        raw_scores = estimator.predict_proba(matrix[block_start:block_end])[:, 1]
        predicted_probabilities.extend(_calibrate(calibrator, raw_scores).tolist())
        actual.extend(labels[block_start:block_end])
    return {
        "method": "expanding_window",
        "model": "xgboost",
        "step": step,
        **(_metrics(actual, predicted_probabilities) if actual else {
            "sample_count": 0,
            "precision": None,
            "recall": None,
            "f1": None,
            "brier_score": None,
            "expected_calibration_error": None,
        }),
    }


def compare_news_feature_walk_forward(
    features: Sequence[dict[str, float]],
    labels: Sequence[int],
    *,
    start_index: int,
    step: int = 60,
) -> dict[str, Any]:
    """Compare XGBoost with/without archived news features on identical future folds."""
    if len(features) != len(labels):
        raise ValueError("News-feature rows and labels must have equal lengths.")
    if not 0 <= start_index <= len(labels) or step <= 0:
        raise ValueError("The walk-forward start index and step must be valid.")
    if not features or not all(name in features[0] for name in NEWS_DIRECTION_FEATURE_NAMES):
        return {
            "status": "unavailable",
            "message": "The feature set does not include the expected news inputs.",
        }
    news_observations = sum(
        row["news_available"] > 0 for row in features[start_index:]
    )
    if news_observations < MIN_NEWS_EVALUATION_SAMPLES:
        return {
            "status": "insufficient_news_history",
            "news_available_sample_count": news_observations,
            "required_news_sample_count": MIN_NEWS_EVALUATION_SAMPLES,
            "message": (
                "The point-in-time news archive does not cover enough out-of-sample "
                "observations to compare variants."
            ),
        }

    baseline_features = [
        {name: value for name, value in row.items() if name not in NEWS_DIRECTION_FEATURE_NAMES}
        for row in features
    ]
    baseline = _fit_walk_forward_xgboost(
        _matrix(baseline_features),
        labels,
        start_index=start_index,
        step=step,
    )
    with_news = _fit_walk_forward_xgboost(
        _matrix(features),
        labels,
        start_index=start_index,
        step=step,
    )
    if baseline["sample_count"] < MIN_TEST_SAMPLES or with_news["sample_count"] != baseline["sample_count"]:
        return {
            "status": "insufficient_out_of_sample_data",
            "news_available_sample_count": news_observations,
            "without_news": baseline,
            "with_news": with_news,
            "message": "The paired chronological test folds did not produce enough matching outcomes.",
        }

    return {
        "status": "ready",
        "method": "paired_expanding_window_xgboost",
        "step": step,
        "news_available_sample_count": news_observations,
        "without_news": baseline,
        "with_news": with_news,
        "metric_delta_with_news_minus_without_news": {
            metric: (
                with_news[metric] - baseline[metric]
                if with_news[metric] is not None and baseline[metric] is not None
                else None
            )
            for metric in (
                "brier_score",
                "expected_calibration_error",
                "precision",
                "recall",
                "f1",
            )
        },
        "interpretation": (
            "Negative Brier-score and calibration-error deltas favor the news-feature variant; "
            "review sample size and all metrics before changing production features."
        ),
    }


def train_direction_ensemble(
    features: Sequence[dict[str, float]] | Sequence[Sequence[float]],
    labels: Sequence[int],
    candles: list[dict[str, Any]],
    indices: Sequence[int],
    *,
    latest_features: dict[str, float],
    lstm_epochs: int = 30,
) -> dict[str, Any]:
    """Train on the earliest window, calibrate next, select on validation, test last."""
    sample_count = len(labels)
    train_end, calibration_end, selection_end = _split_sizes(sample_count)
    required = train_end >= MIN_TRAIN_SAMPLES and (
        calibration_end - train_end >= MIN_CALIBRATION_SAMPLES
        and selection_end - calibration_end >= MIN_SELECTION_SAMPLES
        and sample_count - selection_end >= MIN_TEST_SAMPLES
    )
    if sample_count < MIN_SAMPLES or not required:
        return {
            "status": "insufficient_data",
            "validation_status": "insufficient_data",
            "message": (
                f"At least {MIN_SAMPLES} labeled observations are required, with "
                "separate chronological training, calibration, selection, and test windows."
            ),
            "sample_count": sample_count,
            "required_sample_count": MIN_SAMPLES,
        }

    y = np.asarray(labels, dtype=int)
    if len(set(y[:train_end])) < 2 or len(set(y[train_end:calibration_end])) < 2:
        return {
            "status": "insufficient_class_history",
            "validation_status": "insufficient_class_history",
            "message": "Training and calibration windows must each contain up and down outcomes.",
            "sample_count": sample_count,
            "required_sample_count": MIN_SAMPLES,
        }
    matrix = _matrix(features)
    candle_indices = list(indices)
    sequence_values = build_ohlcv_sequences(candles, candle_indices, SEQUENCE_LENGTH)

    xgb_model = _estimator("xgboost")
    xgb_model.fit(matrix[:train_end], y[:train_end])
    xgb_calibration_raw = xgb_model.predict_proba(matrix[train_end:calibration_end])[:, 1]
    xgb_calibrator = _fit_probability_calibrator(
        xgb_calibration_raw, y[train_end:calibration_end]
    )

    lstm_model = train_lstm_model(
        sequence_values[:train_end],
        y[:train_end].tolist(),
        epochs=lstm_epochs,
    )
    lstm_calibration_raw = predict_lstm_probabilities(
        lstm_model, sequence_values[train_end:calibration_end]
    )
    lstm_calibrator = _fit_probability_calibrator(
        lstm_calibration_raw, y[train_end:calibration_end]
    )

    selection_labels = y[calibration_end:selection_end]
    xgb_selection = _calibrate(
        xgb_calibrator,
        xgb_model.predict_proba(matrix[calibration_end:selection_end])[:, 1],
    )
    lstm_selection = _calibrate(
        lstm_calibrator,
        predict_lstm_probabilities(lstm_model, sequence_values[calibration_end:selection_end]),
    )
    xgb_selection_brier = float(brier_score_loss(selection_labels, xgb_selection))
    lstm_selection_brier = float(brier_score_loss(selection_labels, lstm_selection))
    best_individual_brier = min(xgb_selection_brier, lstm_selection_brier)
    ensemble_choice = (best_individual_brier, 1.0 if xgb_selection_brier <= lstm_selection_brier else 0.0)
    for xgb_weight in np.linspace(0.1, 0.9, 9):
        combined = xgb_weight * xgb_selection + (1 - xgb_weight) * lstm_selection
        score = float(brier_score_loss(selection_labels, combined))
        if score < ensemble_choice[0] - 1e-6:
            ensemble_choice = (score, float(xgb_weight))
    selected_xgb_weight = ensemble_choice[1]
    ensemble_selected = selected_xgb_weight not in (0.0, 1.0)

    test_labels = y[selection_end:]
    xgb_test = _calibrate(
        xgb_calibrator,
        xgb_model.predict_proba(matrix[selection_end:])[:, 1],
    )
    lstm_test = _calibrate(
        lstm_calibrator,
        predict_lstm_probabilities(lstm_model, sequence_values[selection_end:]),
    )
    combined_test = selected_xgb_weight * xgb_test + (1 - selected_xgb_weight) * lstm_test
    walk_forward = _fit_walk_forward_xgboost(
        matrix,
        y.tolist(),
        start_index=selection_end,
    )

    latest_sequence = build_ohlcv_sequences(
        candles,
        [len(candles) - 1],
        SEQUENCE_LENGTH,
    )
    latest_xgb_raw = float(
        xgb_model.predict_proba([[latest_features[name] for name in features[0]]])[:, 1][0]
    )
    latest_lstm_raw = float(predict_lstm_probabilities(lstm_model, latest_sequence)[0])
    latest_xgb = float(_calibrate(xgb_calibrator, [latest_xgb_raw])[0])
    latest_lstm = float(_calibrate(lstm_calibrator, [latest_lstm_raw])[0])
    latest_up_probability = selected_xgb_weight * latest_xgb + (1 - selected_xgb_weight) * latest_lstm

    return {
        "status": "ready",
        "validation_status": "out_of_sample_evaluated",
        "calibration_status": "sigmoid_calibrated_on_chronological_holdout",
        "sample_count": sample_count,
        "split_sizes": {
            "training": train_end,
            "calibration": calibration_end - train_end,
            "selection": selection_end - calibration_end,
            "test": sample_count - selection_end,
        },
        "metrics": {
            "xgboost": _metrics(test_labels, xgb_test),
            "lstm": _metrics(test_labels, lstm_test),
            "selected": _metrics(test_labels, combined_test),
            "walk_forward_xgboost": walk_forward,
        },
        "selection": {
            "xgboost_validation_brier": xgb_selection_brier,
            "lstm_validation_brier": lstm_selection_brier,
            "selected_validation_brier": ensemble_choice[0],
            "ensemble_used": ensemble_selected,
            "xgboost_weight": selected_xgb_weight,
            "lstm_weight": 1 - selected_xgb_weight,
        },
        "latest": {
            "upward_probability": latest_up_probability,
            "downward_probability": 1 - latest_up_probability,
            "xgboost_upward_probability": latest_xgb,
            "lstm_upward_probability": latest_lstm,
        },
        "artifacts": {
            "xgboost": xgb_model,
            "xgboost_calibrator": xgb_calibrator,
            "lstm": lstm_model,
            "lstm_calibrator": lstm_calibrator,
            "xgboost_weight": selected_xgb_weight,
            "feature_names": list(features[0].keys()) if isinstance(features[0], dict) else [],
            "sequence_length": SEQUENCE_LENGTH,
        },
    }
