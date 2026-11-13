import requests
from fastapi import APIRouter, HTTPException, Query
from sqlalchemy.exc import SQLAlchemyError

from app.data.news import fetch_market_news
from app.services.ai_picks.news import archive_news_intelligence_articles
from config.settings import settings

router = APIRouter(prefix="/api/v1", tags=["news"])


@router.get("/news")
def search_news(
    query: str = Query(default="", max_length=100),
    limit: int = Query(default=25, ge=1, le=50),
) -> dict:
    try:
        result = fetch_market_news(
            query=query,
            limit=limit,
            newsdata_api_key=settings.news_data_api_key,
        )
        result.update(archive_news_intelligence_articles(result))
        return result
    except requests.HTTPError as exc:
        provider = "NewsData.io" if settings.news_data_api_key else "GDELT"
        if exc.response is not None and exc.response.status_code == 429:
            retry_after = exc.response.headers.get("Retry-After", "60")
            raise HTTPException(
                status_code=503,
                detail=f"{provider} is temporarily rate-limiting searches. Wait before trying again.",
                headers={"Retry-After": retry_after},
            ) from exc
        raise HTTPException(
            status_code=502,
            detail=f"{provider} did not respond to this search. Try again shortly.",
        ) from exc
    except requests.RequestException as exc:
        provider = "NewsData.io" if settings.news_data_api_key else "GDELT"
        raise HTTPException(
            status_code=502,
            detail=f"{provider} did not respond to this search. Try again shortly.",
        ) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except SQLAlchemyError as exc:
        raise HTTPException(
            status_code=503,
            detail="News was retrieved, but the point-in-time news archive could not be updated.",
        ) from exc