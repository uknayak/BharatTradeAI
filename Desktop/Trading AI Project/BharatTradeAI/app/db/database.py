from sqlalchemy import create_engine
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import declarative_base, sessionmaker
from sqlalchemy.pool import StaticPool

from config.settings import settings


def _build_engine():
    primary_url = settings.database_url
    try:
        engine = create_engine(primary_url, pool_pre_ping=True)
        with engine.connect() as connection:
            connection.execute("SELECT 1")
        return engine
    except (OperationalError, Exception):
        fallback_url = "sqlite:///./bharattradeai.db"
        return create_engine(
            fallback_url,
            connect_args={"check_same_thread": False},
            poolclass=StaticPool,
        )


engine = _build_engine()
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
