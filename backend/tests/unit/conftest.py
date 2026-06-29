"""
conftest.py
-----------
Shared pytest fixtures for the unit test suite.

Uses an in-memory SQLite database instead of Postgres so unit tests run
fast and require no external services. SQLite supports everything our
models need (JSON columns, foreign keys, datetime) well enough for
isolated CRUD testing — integration tests (Day 11) cover real Postgres
behavior separately.
"""

import pytest
from sqlalchemy import create_engine, event
from sqlalchemy.orm import Session, sessionmaker

from src.monitoring.models import Base


@pytest.fixture()
def db() -> Session:
    """
    Yields a fresh in-memory SQLite session with all tables created.
    Each test gets a brand-new database — no shared state between tests.
    """
    engine = create_engine("sqlite:///:memory:", future=True)

    # SQLite disables foreign key enforcement by default; turn it on so
    # ON DELETE CASCADE behavior matches what Postgres would actually do.
    @event.listens_for(engine, "connect")
    def _enable_fk(dbapi_connection, _):
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()

    Base.metadata.create_all(engine)
    SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)
    session = SessionLocal()

    try:
        yield session
    finally:
        session.close()
        engine.dispose()