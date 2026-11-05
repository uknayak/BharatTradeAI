from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class SentimentProbabilities:
    positive: float
    negative: float
    neutral: float

    @property
    def label(self) -> str:
        return max(
            (
                ("POSITIVE", self.positive),
                ("NEGATIVE", self.negative),
                ("NEUTRAL", self.neutral),
            ),
            key=lambda item: item[1],
        )[0]

    @property
    def score(self) -> float:
        return self.positive - self.negative


class FinancialSentimentModel(Protocol):
    model_id: str

    def predict(self, texts: list[str]) -> list[SentimentProbabilities]:
        """Return one normalized positive/negative/neutral probability set per text."""


class SentimentModelUnavailable(RuntimeError):
    """Raised when a configured sentiment model cannot load or run."""
