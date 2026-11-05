from fastapi import APIRouter
from sqlalchemy import text

from app.db.database import SessionLocal

router = APIRouter(prefix="/api/v1", tags=["database"])


@router.get("/db-status")
def db_status() -> dict:
    try:
        db = SessionLocal()
        db.execute(text("SELECT 1"))
        db.close()
        return {"status": "connected", "database": "postgresql"}
    except Exception as exc:  # pragma: no cover
        return {"status": "error", "database": "postgresql", "detail": str(exc)}
