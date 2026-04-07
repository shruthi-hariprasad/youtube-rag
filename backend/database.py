"""Database setup for the backend.

Reads DATABASE_URL from .env, creates the SQLAlchemy engine and a
sessionmaker, and exposes a `get_db` generator for FastAPI dependencies.
"""
from typing import Generator
import os

from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.orm import declarative_base


load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    raise RuntimeError(
        "DATABASE_URL environment variable not set. Set it in .env or the environment."
    )


# Create the SQLAlchemy engine. The project uses PostgreSQL in production,
# but any SQLAlchemy URL (including sqlite://) will work for local testing.
engine = create_engine(DATABASE_URL, future=True)

# Create a configured "Session" class
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)

# Declarative base for models to inherit from. Models import `Base` from here.
Base = declarative_base()


def create_tables() -> None:
    """Create database tables from SQLAlchemy models.

    Useful for quick local development. In production use migrations instead.
    """
    Base.metadata.create_all(bind=engine)


def get_db() -> Generator[Session, None, None]:
    """Yield a database session and ensure it is closed after use.

    Typical usage in FastAPI endpoints:

        def endpoint(db: Session = Depends(get_db)):
            ...
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
