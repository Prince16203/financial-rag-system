"""
scripts/seed_db.py

Manually seeds the database with known financial metrics for
companies the extractor hasn't reached yet (Goldman, JPMorgan, Microsoft).

This exists because Groq's free tier daily token limit makes running
the extractor over all 11405 chunks impractical. These values are
real figures pulled directly from the source PDFs for testing the
SQL agent and ReAct agent across multiple companies.

Run once:
    python scripts/seed_db.py
"""

from database.db import get_session, init_db
from database.models import FinancialMetric
from sqlalchemy.exc import IntegrityError

init_db()
session = get_session()

test_rows = [
    # Goldman Sachs
    FinancialMetric(company="Goldman Sachs", metric_name="ficc_revenue", value=14.52, unit="billion USD", fiscal_year=2025, source="gs-20251231.pdf", page=193),
    FinancialMetric(company="Goldman Sachs", metric_name="ficc_revenue", value=13.34, unit="billion USD", fiscal_year=2024, source="gs-20251231.pdf", page=193),
    FinancialMetric(company="Goldman Sachs", metric_name="net_income", value=14.28, unit="billion USD", fiscal_year=2025, source="gs-20251231.pdf", page=50),

    # JPMorgan
    FinancialMetric(company="JPMorgan", metric_name="revenue", value=177.0, unit="billion USD", fiscal_year=2024, source="jpm-20241231.pdf", page=40),
    FinancialMetric(company="JPMorgan", metric_name="net_income", value=58.5, unit="billion USD", fiscal_year=2024, source="jpm-20241231.pdf", page=42),

    # Microsoft
    FinancialMetric(company="Microsoft", metric_name="revenue", value=245.1, unit="billion USD", fiscal_year=2025, source="MSFT-2025.pdf", page=50),
    FinancialMetric(company="Microsoft", metric_name="net_income", value=88.1, unit="billion USD", fiscal_year=2025, source="MSFT-2025.pdf", page=52),

    # BlackRock (already partially there, adding clean entry)
    FinancialMetric(company="BlackRock", metric_name="revenue", value=20.4, unit="billion USD", fiscal_year=2025, source="BLK-2026.pdf", page=88),
]

saved = 0
for row in test_rows:
    try:
        session.add(row)
        session.commit()
        saved += 1
    except IntegrityError:
        session.rollback()  # already exists, skip

session.close()
print(f"Seeded {saved} rows (duplicates skipped)")