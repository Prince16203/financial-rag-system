"""
database/models.py

SQLAlchemy table definitions for the financial metrics database.
Stores structured metrics extracted from 10-K filings by metric_extractor.py

Table: financial_metrics
    One row = one metric for one company for one year from one source page.

Example rows:
    Apple | revenue | 391.04 | billion USD | 2024 | aapl-20240928.pdf | 32
    Goldman | ficc_revenue | 14.52 | billion USD | 2025 | gs-20251231.pdf | 193
"""

from sqlalchemy import (
    Column,
    Integer,
    String,
    Float,
    DateTime,
    UniqueConstraint,
)
from sqlalchemy.orm import declarative_base
from datetime import datetime

Base = declarative_base()


class FinancialMetric(Base):
    """
    One extracted financial metric from one PDF page.

    Unique constraint on (company, metric_name, fiscal_year, source)
    prevents duplicate rows when re-running the extractor.
    """
    __tablename__ = "financial_metrics"

    id            = Column(Integer, primary_key=True, autoincrement=True)
    company       = Column(String(100), nullable=False)
    metric_name   = Column(String(100), nullable=False)
    value         = Column(Float, nullable=False)
    unit          = Column(String(50), nullable=True)
    fiscal_year   = Column(Integer, nullable=True)
    source        = Column(String(200), nullable=False)
    page          = Column(Integer, nullable=False)
    created_at    = Column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        UniqueConstraint(
            "company", "metric_name", "fiscal_year", "source",
            name="uq_metric_per_company_year_source",
        ),
    )

    def __repr__(self):
        return (
            f"<FinancialMetric "
            f"company={self.company!r} "
            f"metric={self.metric_name!r} "
            f"value={self.value} {self.unit} "
            f"year={self.fiscal_year}>"
        )

    def to_dict(self) -> dict:
        """Convert to plain dict for easy JSON serialisation."""
        return {
            "id":          self.id,
            "company":     self.company,
            "metric_name": self.metric_name,
            "value":       self.value,
            "unit":        self.unit,
            "fiscal_year": self.fiscal_year,
            "source":      self.source,
            "page":        self.page,
        }