from datetime import datetime, timedelta, timezone
import json

import pytest
from langchain_core.messages import AIMessage
from fastapi.testclient import TestClient

from app.main import app
from app.services.ai_picks import rag


def _citation(
    article_id: str,
    title: str,
    sentiment: str,
    *,
    published_at: datetime,
    retrieved_at: datetime,
) -> dict:
    return {
        "article_id": article_id,
        "title": title,
        "source": "Example Finance",
        "provider": "GDELT",
        "published_at": published_at.isoformat(),
        "retrieved_at": retrieved_at.isoformat(),
        "url": f"https://news.example/{article_id}",
        "excerpt": title,
        "event_category": "GENERAL",
        "provider_sentiment": sentiment,
    }


def test_rag_retrieval_constrains_company_publication_and_retrieval_cutoffs(monkeypatch):
    cutoff = datetime.now(timezone.utc) - timedelta(hours=1)
    captured = {}
    documents = [
        type(
            "RetrievedDocument",
            (),
            {
                "page_content": "TCS reports an earnings increase.",
                "metadata": {
                    "article_id": "positive",
                    "title": "TCS reports an earnings increase.",
                    "source": "Example Finance",
                    "provider": "GDELT",
                    "url": "https://news.example/positive",
                    "published_at": (cutoff - timedelta(hours=2)).isoformat(),
                    "retrieved_at": (cutoff - timedelta(hours=1)).isoformat(),
                    "event_category": "QUARTERLY_RESULTS",
                    "sentiment_label": "POSITIVE",
                },
            },
        )(),
        type(
            "RetrievedDocument",
            (),
            {
                "page_content": "TCS reports lower revenue.",
                "metadata": {
                    "article_id": "negative",
                    "title": "TCS reports lower revenue.",
                    "source": "Another Finance",
                    "provider": "GDELT",
                    "url": "https://news.example/negative",
                    "published_at": (cutoff - timedelta(hours=3)).isoformat(),
                    "retrieved_at": (cutoff - timedelta(hours=2)).isoformat(),
                    "event_category": "QUARTERLY_RESULTS",
                    "sentiment_label": "NEGATIVE",
                },
            },
        )(),
    ]

    class FakeVectorStore:
        def similarity_search(self, query, *, k, filter):
            captured.update(query=query, k=k, filter=filter)
            return documents

    monkeypatch.setattr(rag, "index_stored_news", lambda: {
        "status": "ready",
        "provider": "Chroma + Ollama embeddings",
        "embedding_model": "nomic-embed-text",
        "article_count": 2,
        "indexed_article_count": 0,
        "indexed_chunk_count": 0,
        "total_chunk_count": 2,
        "skipped_article_count": 0,
        "indexed_at": cutoff.isoformat(),
    })
    monkeypatch.setattr(rag, "_get_vector_store", lambda: FakeVectorStore())

    result = rag.retrieve_company_news(
        symbol="tcs",
        query="earnings outlook",
        cutoff=cutoff,
        limit=3,
    )

    assert captured["query"] == "earnings outlook"
    assert captured["k"] == 3
    assert captured["filter"]["$and"] == [
        {"symbol": "TCS"},
        {"published_timestamp": {"$lt": cutoff.timestamp()}},
        {"retrieved_timestamp": {"$lt": cutoff.timestamp()}},
    ]
    assert result["evidence_status"] == "mixed_provider_sentiment"
    assert {item["article_id"] for item in result["citations"]} == {"positive", "negative"}


def test_rag_rejects_unknown_symbols_future_cutoffs_and_bad_questions():
    with pytest.raises(ValueError, match="Unsupported"):
        rag.retrieve_company_news(symbol="UNKNOWN", query="news")
    with pytest.raises(ValueError, match="between 1 and 500"):
        rag.retrieve_company_news(symbol="TCS", query="   ")
    with pytest.raises(ValueError, match="future"):
        rag.retrieve_company_news(
            symbol="TCS",
            query="earnings",
            cutoff=datetime.now(timezone.utc) + timedelta(minutes=5),
        )


def test_grounded_answer_returns_no_evidence_without_calling_llm(monkeypatch):
    monkeypatch.setattr(rag, "retrieve_company_news", lambda **kwargs: {
        "status": "ready",
        "symbol": "TCS",
        "citations": [],
        "evidence_status": "no_evidence",
        "cutoff": datetime.now(timezone.utc).isoformat(),
    })
    monkeypatch.setattr(
        rag,
        "ChatOllama",
        lambda **kwargs: pytest.fail("The LLM must not answer without retrieved evidence."),
    )

    result = rag.answer_company_news_question(symbol="TCS", question="What changed?")

    assert result["citation_ids"] == []
    assert "No matching archived" in result["answer"]


def test_grounded_answer_rejects_unknown_citation_ids(monkeypatch):
    now = datetime.now(timezone.utc)
    citation = _citation(
        "known",
        "TCS earnings update",
        "NEUTRAL",
        published_at=now - timedelta(hours=2),
        retrieved_at=now - timedelta(hours=1),
    )
    monkeypatch.setattr(rag, "retrieve_company_news", lambda **kwargs: {
        "status": "ready",
        "symbol": "TCS",
        "citations": [citation],
        "evidence_status": "limited",
    })

    def fake_chat(values):
        return AIMessage(content=json.dumps({
                "answer": "Unsupported claim.",
                "citation_ids": ["unknown"],
            }))

    monkeypatch.setattr(rag, "ChatOllama", lambda **kwargs: fake_chat)

    with pytest.raises(rag.NewsRagUnavailable, match="Grounded news analysis"):
        rag.answer_company_news_question(symbol="TCS", question="What happened?")


def test_grounded_news_api_validates_question_and_surfaces_rag_unavailable(monkeypatch):
    from app.api.routes import ai_stock_picks

    def unavailable(**kwargs):
        raise rag.NewsRagUnavailable("Embedding model is unavailable.")

    monkeypatch.setattr(ai_stock_picks, "answer_company_news_question", unavailable)
    client = TestClient(app)
    invalid = client.post(
        "/api/v1/ai-stock-picks/news/grounded-analysis",
        json={"symbol": "TCS", "question": " "},
    )
    unavailable_response = client.post(
        "/api/v1/ai-stock-picks/news/grounded-analysis",
        json={"symbol": "TCS", "question": "What are the recent earnings updates?"},
    )

    assert invalid.status_code == 422
    assert unavailable_response.status_code == 503
    assert unavailable_response.json()["detail"] == "Embedding model is unavailable."
