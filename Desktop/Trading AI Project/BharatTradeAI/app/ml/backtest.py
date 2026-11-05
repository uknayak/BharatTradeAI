"""Walk-forward and historical performance evaluation utilities."""

from collections.abc import Iterable, Mapping
from math import sqrt
from statistics import mean, pstdev
from typing import Any, Sequence

from app.ml.train import _estimator, _matrix


def walk_forward_test(features: Sequence[dict[str, float]] | Sequence[Sequence[float]], labels: Sequence[int], model: str = "logistic", initial_train_size: int = 30, step: int = 1) -> dict[str, Any]:
    if len(features) != len(labels) or initial_train_size < 2 or initial_train_size >= len(labels):
        raise ValueError("features and labels must align and leave a test window")
    if step <= 0:
        raise ValueError("step must be positive")
    matrix = _matrix(features)
    predictions: list[int] = []
    probabilities: list[float] = []
    actual: list[int] = []
    for index in range(initial_train_size, len(labels), step):
        estimator = _estimator(model)
        estimator.fit(matrix[:index], labels[:index])
        batch = matrix[index:min(index + step, len(labels))]
        predictions.extend(estimator.predict(batch).tolist())
        probabilities.extend(estimator.predict_proba(batch)[:, 1].tolist())
        actual.extend(labels[index:min(index + step, len(labels))])
    accuracy = sum(prediction == target for prediction, target in zip(predictions, actual)) / len(actual) if actual else None
    return {"model": model, "predictions": predictions, "probabilities": probabilities, "actual": actual, "accuracy": accuracy, "test_size": len(actual)}


def _year(row: Mapping[str, Any]) -> int:
    value = row.get("year")
    if value is None and row.get("date") is not None:
        value = str(row["date"])[:4]
    try:
        return int(value)
    except (TypeError, ValueError) as error:
        raise ValueError("each backtest row requires a numeric year or date") from error


def _pnl(row: Mapping[str, Any]) -> float:
    value = row.get("pnl", row.get("profit", row.get("return")))
    try:
        return float(value)
    except (TypeError, ValueError) as error:
        raise ValueError("each backtest row requires pnl, profit, or return") from error


def _classification_metrics(rows: list[Mapping[str, Any]]) -> dict[str, float | None]:
    classified = [(row.get("prediction"), row.get("actual")) for row in rows if row.get("prediction") is not None and row.get("actual") is not None]
    if not classified:
        return {"accuracy": None, "precision": None, "recall": None}
    true_positive = sum(prediction == actual == 1 for prediction, actual in classified)
    false_positive = sum(prediction == 1 and actual == 0 for prediction, actual in classified)
    false_negative = sum(prediction == 0 and actual == 1 for prediction, actual in classified)
    return {
        "accuracy": sum(prediction == actual for prediction, actual in classified) / len(classified),
        "precision": true_positive / (true_positive + false_positive) if true_positive + false_positive else None,
        "recall": true_positive / (true_positive + false_negative) if true_positive + false_negative else None,
    }


def _trading_metrics(rows: list[Mapping[str, Any]]) -> dict[str, float | int | None]:
    profits = [_pnl(row) for row in rows]
    wins = [value for value in profits if value > 0]
    losses = [value for value in profits if value < 0]
    gross_profit = sum(wins)
    gross_loss = sum(losses)
    equity = 0.0
    peak = 0.0
    max_drawdown = 0.0
    for value in profits:
        equity += value
        peak = max(peak, equity)
        max_drawdown = max(max_drawdown, peak - equity)
    average_profit = mean(wins) if wins else 0.0
    average_loss = mean(losses) if losses else 0.0
    win_rate = len(wins) / len(profits) if profits else None
    return {
        "count": len(profits),
        "win_rate": win_rate,
        "profit_factor": gross_profit / abs(gross_loss) if gross_loss else None,
        "sharpe": mean(profits) / pstdev(profits) * sqrt(252) if len(profits) > 1 and pstdev(profits) else None,
        "max_drawdown": max_drawdown,
        "average_profit": average_profit,
        "average_loss": average_loss,
        "expectancy": (win_rate * average_profit + (1 - win_rate) * average_loss) if win_rate is not None else None,
    }


def _metrics(rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    return {**_classification_metrics(rows), **_trading_metrics(rows)}


def evaluate_backtest(records: Iterable[Mapping[str, Any]], years: Sequence[int] = (2022, 2023, 2024, 2025, 2026)) -> dict[str, Any]:
    """Evaluate dated signal outcomes and return yearly and aggregate metrics.

    Each record needs ``year`` or ``date`` and a numeric ``pnl``, ``profit``,
    or ``return``. ``prediction`` and ``actual`` are optional classification
    labels; ``signal`` or ``strength`` is optional for the signal breakdown.
    """
    rows = list(records)
    grouped = {year: [row for row in rows if _year(row) == year] for year in years}
    signal_rows: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        label = row.get("signal", row.get("strength"))
        if label is not None:
            signal_rows.setdefault(str(label), []).append(row)
    signal_breakdown = {
        label: {"count": len(items), "win_rate": sum(_pnl(row) > 0 for row in items) / len(items) if items else None}
        for label, items in signal_rows.items()
    }
    return {
        "module": "backtest",
        "status": "ready",
        "years": {str(year): _metrics(grouped[year]) for year in years},
        "overall": _metrics(rows),
        "signal_breakdown": signal_breakdown,
    }


def run_backtest(records: Iterable[Mapping[str, Any]] | None = None) -> dict[str, Any]:
    """Run historical evaluation when signal outcomes are supplied."""
    if records is None:
        return {"module": "backtest", "status": "no_data"}
    return evaluate_backtest(records)
