# backend/app/db.py
from __future__ import annotations

import os
import time
from contextlib import contextmanager
from typing import Any, Generator, Iterator

from sqlalchemy import create_engine
from sqlalchemy.exc import OperationalError, ProgrammingError
from sqlalchemy.orm import Session, declarative_base, sessionmaker

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql+psycopg://postgres:postgres@db:5432/postgres",
)

Base = declarative_base()


def _connect_args(url: str) -> dict[str, Any]:
    if url.startswith("postgresql"):
        return {
            "connect_timeout": int(os.getenv("DB_CONNECT_TIMEOUT", "5")),
            "options": "-c statement_timeout=5000",
        }
    return {}


engine = create_engine(
    DATABASE_URL,
    pool_pre_ping=True,
    pool_size=5,
    max_overflow=10,
    future=True,
    connect_args=_connect_args(DATABASE_URL),
)

SessionLocal = sessionmaker(
    bind=engine,
    autoflush=False,
    autocommit=False,
    expire_on_commit=False,
    future=True,
)

def init_db() -> None:
    max_attempts = int(os.getenv("DB_INIT_MAX_RETRIES", "10"))
    backoff = float(os.getenv("DB_INIT_RETRY_DELAY", "0.5"))
    last_error: OperationalError | None = None

    for attempt in range(max_attempts):
        try:
            with engine.begin() as conn:
                if conn.dialect.name == "postgresql":
                    try:
                        conn.exec_driver_sql("CREATE EXTENSION IF NOT EXISTS vector")
                    except ProgrammingError:
                        conn.rollback()
                Base.metadata.create_all(bind=conn)
            return
        except OperationalError as exc:
            last_error = exc
            time.sleep(min(backoff * (2 ** attempt), 5.0))

    raise last_error


@contextmanager
def session_scope() -> Iterator[Session]:
    db = SessionLocal()
    try:
        yield db
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


def get_db() -> Generator[Session, None, None]:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
