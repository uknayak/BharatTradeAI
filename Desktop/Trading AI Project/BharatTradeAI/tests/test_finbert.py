from contextlib import nullcontext
from types import SimpleNamespace

import pytest

from app.services.sentiment.base import SentimentModelUnavailable
from app.services.sentiment.finbert import BATCH_SIZE, FinBertSentimentModel


class _ProbabilityRows:
    def __init__(self, rows):
        self.rows = rows

    def tolist(self):
        return self.rows


class _FakeTorch:
    @staticmethod
    def inference_mode():
        return nullcontext()

    @staticmethod
    def softmax(logits, dim):
        assert dim == -1
        return _ProbabilityRows(logits)


class _FakeTokenizer:
    def __init__(self):
        self.calls = []

    def __call__(self, texts, **kwargs):
        self.calls.append((texts, kwargs))
        return {"input_ids": texts}


class _FakeModel:
    config = SimpleNamespace(id2label={0: "negative", 1: "neutral", 2: "positive"})

    def __init__(self):
        self.batch_sizes = []
        self.device = None
        self.eval_called = False

    def to(self, device):
        self.device = device

    def eval(self):
        self.eval_called = True

    def __call__(self, input_ids):
        self.batch_sizes.append(len(input_ids))
        return SimpleNamespace(logits=[[0.2, 0.3, 0.5] for _ in input_ids])


def test_finbert_adapter_returns_normalized_class_probabilities(monkeypatch):
    tokenizer = _FakeTokenizer()
    model = _FakeModel()

    class AutoTokenizer:
        @staticmethod
        def from_pretrained(model_id):
            assert model_id == "ProsusAI/finbert"
            return tokenizer

    class AutoModelForSequenceClassification:
        @staticmethod
        def from_pretrained(model_id):
            assert model_id == "ProsusAI/finbert"
            return model

    modules = {
        "torch": _FakeTorch(),
        "transformers": SimpleNamespace(
            AutoTokenizer=AutoTokenizer,
            AutoModelForSequenceClassification=AutoModelForSequenceClassification,
        ),
    }
    monkeypatch.setattr("app.services.sentiment.finbert.importlib.import_module", modules.__getitem__)

    predictions = FinBertSentimentModel().predict([f"headline {index}" for index in range(BATCH_SIZE + 1)])

    assert len(predictions) == BATCH_SIZE + 1
    assert model.batch_sizes == [BATCH_SIZE, 1]
    assert model.device == "cpu"
    assert model.eval_called
    assert predictions[0].positive == pytest.approx(0.5)
    assert predictions[0].negative == pytest.approx(0.2)
    assert predictions[0].neutral == pytest.approx(0.3)
    assert predictions[0].label == "POSITIVE"
    assert sum((predictions[0].positive, predictions[0].negative, predictions[0].neutral)) == pytest.approx(1)
    assert tokenizer.calls[0][1]["truncation"] is True
    assert tokenizer.calls[0][1]["max_length"] == 512


def test_finbert_adapter_surfaces_missing_dependencies(monkeypatch):
    def missing_module(name):
        raise ImportError(name)

    monkeypatch.setattr("app.services.sentiment.finbert.importlib.import_module", missing_module)

    with pytest.raises(SentimentModelUnavailable, match="install the configured model dependencies"):
        FinBertSentimentModel().predict(["Company reports higher earnings."])
