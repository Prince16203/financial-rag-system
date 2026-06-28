"""
database/db.py

Database connection and session management.
Creates the SQLite database file and all tables on first run.

Usage:
    from database.db import get_session, init_db

    init_db()                    # creates tables if they don't exist
    session = get_session()      # get a session to query or insert
    session.close()              # always close when done
"""

import logging
from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session

from database.models import Base

logger = logging.getLogger(__name__)

# SQLite file lives inside data/ folder which is already in .gitignore
DB_PATH = "data/financial_metrics.db"
DB_URL  = f"sqlite:///{DB_PATH}"

# Create engine once at module level — reused across all sessions
engine = create_engine(
    DB_URL,
    connect_args={"check_same_thread": False},  # needed for SQLite + Streamlit
    echo=False,  # set True to log all SQL queries for debugging
)

SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)


def init_db() -> None:
    """
    Create all tables defined in models.py if they do not exist.
    Safe to call multiple times — will not drop existing data.
    """
    Path("data").mkdir(exist_ok=True)
    Base.metadata.create_all(bind=engine)
    logger.info(f"Database initialised at: {DB_PATH}")


def get_session() -> Session:
    """
    Return a new SQLAlchemy session.
    Caller is responsible for closing it.

    Usage:
        session = get_session()
        try:
            session.add(metric)
            session.commit()
        except Exception:
            session.rollback()
        finally:
            session.close()
    """
    return SessionLocal()


def get_metric_count() -> int:
    """Return total number of rows in financial_metrics table."""
    session = get_session()
    try:
        from database.models import FinancialMetric
        return session.query(FinancialMetric).count()
    finally:
        session.close()


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)

    init_db()

    from database.models import FinancialMetric
    from sqlalchemy.exc import IntegrityError

    session = get_session()
    try:
        # Insert a test row
        test = FinancialMetric(
            company="Apple",
            metric_name="revenue",
            value=391.04,
            unit="billion USD",
            fiscal_year=2024,
            source="aapl-20240928.pdf",
            page=32,
        )
        session.add(test)
        session.commit()
        print("Test row inserted successfully")

        # Query it back
        result = session.query(FinancialMetric).filter_by(company="Apple").first()
        print(f"Retrieved: {result}")
        print(f"Total rows: {get_metric_count()}")

    except IntegrityError:
        print("Row already exists — unique constraint working correctly")
        session.rollback()
    finally:
        session.close()