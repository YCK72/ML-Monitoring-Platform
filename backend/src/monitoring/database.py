"""
database.py
-----------
SQLAlchemy engine + session factory, shared by Alembic migrations,
the repository layer, and the FastAPI dependency injection system.
"""

import os

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

DATABASE_URL: str = os.getenv(
    "DATABASE_URL",
    "postgresql+psycopg2://mluser:mlpassword@localhost:5433/ml_monitoring",
)

# pool_pre_ping avoids "stale connection" errors after Postgres restarts
engine = create_engine(DATABASE_URL, pool_pre_ping=True, future=True)

SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)


def get_db() -> Session:
    """
    FastAPI dependency — yields a Session and guarantees it's closed
    after the request completes, even on exceptions.

    Usage:
        @app.get("/drift/reports")
        def list_reports(db: Session = Depends(get_db)):
            ...
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()