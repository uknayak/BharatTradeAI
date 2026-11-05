from fastapi import APIRouter
from sqlalchemy import text

from app.core.redis_client import check_redis
from app.db.database import SessionLocal

router = APIRouter(prefix="/api/v1", tags=["health"])


@router.get("/health")
def health_check() -> dict:
    return {
        "status": "ok",
        "service": "BharatTradeAI",
        "environment": "development",
    }


@router.get("/ready")
def readiness_check() -> dict:
    db_status = {"status": "pending", "detail": "not checked"}
    redis_status = {"status": "pending", "detail": "not checked"}

    try:
        db = SessionLocal()
        db.execute(text("SELECT 1"))
        db_status = {"status": "connected", "detail": "postgresql ping successful"}
        db.close()
    except Exception as exc:  # pragma: no cover
        db_status = {"status": "unavailable", "detail": str(exc)}

    redis_status = check_redis()

    if db_status["status"] == "connected" and redis_status["status"] == "connected":
        overall = "ready"
    elif db_status["status"] == "unavailable" or redis_status["status"] == "unavailable":
        overall = "degraded"
    else:
        overall = "starting"

    return {
        "status": overall,
        "checks": {
            "database": db_status,
            "redis": redis_status,
            "market_data": {"status": "pending", "detail": "not configured yet"},
        },
    }
