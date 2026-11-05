from sqlalchemy.exc import SQLAlchemyError

from app.db.database import Base, engine


def init_db() -> dict:
    try:
        Base.metadata.create_all(bind=engine)
        return {"status": "initialized", "database": "postgresql"}
    except SQLAlchemyError as exc:
        return {"status": "unavailable", "detail": str(exc)}
    except Exception as exc:  # pragma: no cover
        return {"status": "error", "detail": str(exc)}
