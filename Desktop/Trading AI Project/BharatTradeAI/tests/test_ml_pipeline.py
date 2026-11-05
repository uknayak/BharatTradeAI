import pytest

from app.ml.features import build_features, build_labels, build_training_set

sklearn = pytest.importorskip("sklearn")

from app.ml.backtest import walk_forward_test
from app.ml.predict import predict_signal
from app.ml.train import train_model


def _candles(count=90):
    return [{"high": 100 + index + 3, "low": 100 + index - 3, "close": 100 + index + (2 if index % 4 < 2 else -2), "volume": 100 + (index % 5) * 10} for index in range(count)]


def test_features_labels_and_training_set_are_aligned():
    candles = _candles()
    assert len(build_features(candles, lookback=5)) == len(candles)
    assert build_labels([row["close"] for row in candles], horizon=2)[-2:] == [None, None]
    training = build_training_set(candles, lookback=5, horizon=1)
    assert len(training["features"]) == len(training["labels"])
    assert training["feature_names"]


def test_model_ladder_calibration_prediction_and_walk_forward():
    training = build_training_set(_candles(), lookback=5, horizon=1)
    result = train_model(training["features"], training["labels"], model="logistic", validation_size=0.2, feature_names=training["feature_names"])
    prediction = predict_signal(result, list(training["features"][-1].values()))
    walk_forward = walk_forward_test(training["features"], training["labels"], initial_train_size=30)

    assert result["model"] == "logistic"
    assert 0 <= prediction["probability"] <= 1
    assert walk_forward["test_size"] > 0
    assert len(walk_forward["predictions"]) == walk_forward["test_size"]