from functools import lru_cache

from app.services.sentiment.base import FinancialSentimentModel, SentimentModelUnavailable
from app.services.sentiment.finbert import MODEL_ID, FinBertSentimentModel

DEFAULT_MODEL_ID = MODEL_ID
MODEL_FACTORIES = {MODEL_ID: FinBertSentimentModel}


@lru_cache(maxsize=4)
def get_sentiment_model(model_id: str = DEFAULT_MODEL_ID) -> FinancialSentimentModel:
    factory = MODEL_FACTORIES.get(model_id)
    if factory is None:
        raise SentimentModelUnavailable(f"Unsupported financial sentiment model: {model_id}.")
    return factory()
