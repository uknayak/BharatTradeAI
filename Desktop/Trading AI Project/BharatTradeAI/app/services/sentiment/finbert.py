"""Lazy CPU inference adapter for ProsusAI/finbert."""

import importlib
import math
import threading
from typing import Any

from app.services.sentiment.base import (
    SentimentModelUnavailable,
    SentimentProbabilities,
)

MODEL_ID = "ProsusAI/finbert"
BATCH_SIZE = 8
MAX_TOKENS = 512
_load_lock = threading.Lock()


class FinBertSentimentModel:
    model_id = MODEL_ID

    def __init__(self) -> None:
        self._torch: Any = None
        self._tokenizer: Any = None
        self._model: Any = None
        self._label_ids: dict[str, int] | None = None

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        with _load_lock:
            if self._model is not None:
                return
            try:
                torch = importlib.import_module("torch")
                transformers = importlib.import_module("transformers")
                tokenizer = transformers.AutoTokenizer.from_pretrained(self.model_id)
                model = transformers.AutoModelForSequenceClassification.from_pretrained(self.model_id)
                model.to("cpu")
                model.eval()
                labels = {
                    str(label).strip().lower(): int(index)
                    for index, label in model.config.id2label.items()
                }
                if not {"positive", "negative", "neutral"}.issubset(labels):
                    raise SentimentModelUnavailable(
                        f"{self.model_id} does not expose positive, negative, and neutral labels."
                    )
            except SentimentModelUnavailable:
                raise
            except Exception as exc:
                raise SentimentModelUnavailable(
                    f"Could not load {self.model_id}; install the configured model dependencies "
                    "and check Hugging Face model access."
                ) from exc

            self._torch = torch
            self._tokenizer = tokenizer
            self._model = model
            self._label_ids = labels

    def predict(self, texts: list[str]) -> list[SentimentProbabilities]:
        if not texts:
            return []
        self._ensure_loaded()
        predictions: list[SentimentProbabilities] = []
        if self._label_ids is None:
            raise SentimentModelUnavailable(f"{self.model_id} is not initialized.")

        try:
            for start in range(0, len(texts), BATCH_SIZE):
                batch = texts[start : start + BATCH_SIZE]
                tokens = self._tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=MAX_TOKENS,
                    return_tensors="pt",
                )
                with self._torch.inference_mode():
                    logits = self._model(**tokens).logits
                    probability_rows = self._torch.softmax(logits, dim=-1).tolist()

                for row in probability_rows:
                    values = {
                        label: float(row[index])
                        for label, index in self._label_ids.items()
                    }
                    probabilities = SentimentProbabilities(
                        positive=values["positive"],
                        negative=values["negative"],
                        neutral=values["neutral"],
                    )
                    numbers = (
                        probabilities.positive,
                        probabilities.negative,
                        probabilities.neutral,
                    )
                    if any(not math.isfinite(number) or number < 0 for number in numbers):
                        raise SentimentModelUnavailable(
                            f"{self.model_id} returned invalid sentiment probabilities."
                        )
                    total = sum(numbers)
                    if total <= 0:
                        raise SentimentModelUnavailable(
                            f"{self.model_id} returned an empty probability distribution."
                        )
                    predictions.append(
                        SentimentProbabilities(
                            positive=probabilities.positive / total,
                            negative=probabilities.negative / total,
                            neutral=probabilities.neutral / total,
                        )
                    )
        except SentimentModelUnavailable:
            raise
        except Exception as exc:
            raise SentimentModelUnavailable(
                f"{self.model_id} could not analyze the supplied financial news."
            ) from exc

        if len(predictions) != len(texts):
            raise SentimentModelUnavailable(
                f"{self.model_id} returned an unexpected number of sentiment results."
            )
        return predictions
