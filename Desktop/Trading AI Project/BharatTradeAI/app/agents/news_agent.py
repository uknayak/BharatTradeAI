"""News analysis agent backed by the news enrichment pipeline."""

from collections.abc import Iterable
from typing import Any, Mapping

from app.data.news import collect_news


def analyze_news(records: Iterable[Mapping[str, Any]] | None = None) -> dict[str, Any]:
    if records is not None:
        return {"agent": "news_agent", "status": "ready", "articles": collect_news(records)}
    return {"agent": "news_agent", "status": "ready"}
