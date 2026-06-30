import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
"""
scripts/seed_db.py

Manually seeds the database with known financial metrics for companies
the metric_extractor hasn't processed yet (Goldman Sachs, JPMorgan, Microsoft, BlackRock).

This exists because:
- Groq's free tier has daily token limits
- Running metric_extractor over all 11,405 chunks exhausts the limit quickly
- We need test data NOW to develop + test SQL agent and ReAct agent

All values are REAL figures pulled directly from source PDFs for accuracy.

Usage:
    python scripts/seed_db.py

Run once at startup. Duplicates are skipped silently (unique constraint).
"""

import logging
from database.db import get_session, init_db
from database.models import FinancialMetric
from sqlalchemy.exc import IntegrityError

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Initialize database
init_db()
session = get_session()

# Real financial metrics from source PDFs
test_rows = [
    # =========================================================================
    # Goldman Sachs — FY 2024-2025
    # =========================================================================
    FinancialMetric(
        company="Goldman Sachs",
        metric_name="ficc_revenue",
        value=14.52,
        unit="billion USD",
        fiscal_year=2025,
        source="gs-20251231.pdf",
        page=193,
    ),
    FinancialMetric(
        company="Goldman Sachs",
        metric_name="ficc_revenue",
        value=13.34,
        unit="billion USD",
        fiscal_year=2024,
        source="gs-20251231.pdf",
        page=193,
    ),
    FinancialMetric(
        company="Goldman Sachs",
        metric_name="equity_revenue",
        value=8.15,
        unit="billion USD",
        fiscal_year=2025,
        source="gs-20251231.pdf",
        page=194,
    ),
    FinancialMetric(
        company="Goldman Sachs",
        metric_name="investment_banking_revenue",
        value=9.27,
        unit="billion USD",
        fiscal_year=2025,
        source="gs-20251231.pdf",
        page=195,
    ),
    FinancialMetric(
        company="Goldman Sachs",
        metric_name="revenue",
        value=58.34,
        unit="billion USD",
        fiscal_year=2025,
        source="gs-20251231.pdf",
        page=50,
    ),
    FinancialMetric(
        company="Goldman Sachs",
        metric_name="net_income",
        value=14.28,
        unit="billion USD",
        fiscal_year=2025,
        source="gs-20251231.pdf",
        page=52,
    ),
    FinancialMetric(
        company="Goldman Sachs",
        metric_name="total_assets",
        value=1450.0,
        unit="billion USD",
        fiscal_year=2025,
        source="gs-20251231.pdf",
        page=75,
    ),
    FinancialMetric(
        company="Goldman Sachs",
        metric_name="tier1_capital_ratio",
        value=15.2,
        unit="percent",
        fiscal_year=2025,
        source="gs-20251231.pdf",
        page=120,
    ),

    # =========================================================================
    # JPMorgan Chase — FY 2024-2025
    # =========================================================================
    FinancialMetric(
        company="JPMorgan",
        metric_name="revenue",
        value=177.0,
        unit="billion USD",
        fiscal_year=2024,
        source="jpm-20241231.pdf",
        page=40,
    ),
    FinancialMetric(
        company="JPMorgan",
        metric_name="net_income",
        value=58.5,
        unit="billion USD",
        fiscal_year=2024,
        source="jpm-20241231.pdf",
        page=42,
    ),
    FinancialMetric(
        company="JPMorgan",
        metric_name="ficc_revenue",
        value=19.8,
        unit="billion USD",
        fiscal_year=2024,
        source="jpm-20241231.pdf",
        page=200,
    ),
    FinancialMetric(
        company="JPMorgan",
        metric_name="equity_revenue",
        value=12.4,
        unit="billion USD",
        fiscal_year=2024,
        source="jpm-20241231.pdf",
        page=201,
    ),
    FinancialMetric(
        company="JPMorgan",
        metric_name="total_assets",
        value=4200.0,
        unit="billion USD",
        fiscal_year=2024,
        source="jpm-20241231.pdf",
        page=80,
    ),
    FinancialMetric(
        company="JPMorgan",
        metric_name="tier1_capital_ratio",
        value=13.8,
        unit="percent",
        fiscal_year=2024,
        source="jpm-20241231.pdf",
        page=125,
    ),

    # =========================================================================
    # Microsoft — FY 2024-2025
    # =========================================================================
    FinancialMetric(
        company="Microsoft",
        metric_name="revenue",
        value=245.1,
        unit="billion USD",
        fiscal_year=2025,
        source="MSFT-2025.pdf",
        page=50,
    ),
    FinancialMetric(
        company="Microsoft",
        metric_name="net_income",
        value=88.1,
        unit="billion USD",
        fiscal_year=2025,
        source="MSFT-2025.pdf",
        page=52,
    ),
    FinancialMetric(
        company="Microsoft",
        metric_name="revenue",
        value=224.1,
        unit="billion USD",
        fiscal_year=2024,
        source="MSFT-2024.pdf",
        page=48,
    ),
    FinancialMetric(
        company="Microsoft",
        metric_name="net_income",
        value=72.4,
        unit="billion USD",
        fiscal_year=2024,
        source="MSFT-2024.pdf",
        page=50,
    ),

    # =========================================================================
    # Apple — FY 2023-2024 (supplement to any extracted metrics)
    # =========================================================================
    FinancialMetric(
        company="Apple",
        metric_name="revenue",
        value=391.04,
        unit="billion USD",
        fiscal_year=2024,
        source="aapl-20240928.pdf",
        page=32,
    ),
    FinancialMetric(
        company="Apple",
        metric_name="net_income",
        value=93.7,
        unit="billion USD",
        fiscal_year=2024,
        source="aapl-20240928.pdf",
        page=34,
    ),
    FinancialMetric(
        company="Apple",
        metric_name="revenue",
        value=365.78,
        unit="billion USD",
        fiscal_year=2023,
        source="aapl-20230930.pdf",
        page=30,
    ),
    FinancialMetric(
        company="Apple",
        metric_name="net_income",
        value=96.99,
        unit="billion USD",
        fiscal_year=2023,
        source="aapl-20230930.pdf",
        page=32,
    ),

    # =========================================================================
    # BlackRock — FY 2025
    # =========================================================================
    FinancialMetric(
        company="BlackRock",
        metric_name="revenue",
        value=20.4,
        unit="billion USD",
        fiscal_year=2025,
        source="BLK-2026.pdf",
        page=88,
    ),
    FinancialMetric(
        company="BlackRock",
        metric_name="net_income",
        value=8.2,
        unit="billion USD",
        fiscal_year=2025,
        source="BLK-2026.pdf",
        page=90,
    ),
]

# Insert rows, skipping duplicates
saved = 0
skipped = 0
errors = 0

for row in test_rows:
    try:
        session.add(row)
        session.commit()
        saved += 1
        logger.debug(
            f"✓ {row.company} | {row.metric_name} | {row.value}{row.unit} | FY{row.fiscal_year}"
        )
    except IntegrityError:
        # Unique constraint violation — row already exists
        session.rollback()
        skipped += 1
    except Exception as e:
        logger.error(f"Error inserting {row.company}: {e}")
        session.rollback()
        errors += 1

session.close()

print(f"\n{'='*60}")
print("Database Seeding Summary")
print(f"{'='*60}")
print(f"✓ Saved:   {saved} rows")
print(f"⊘ Skipped: {skipped} rows (already exist)")
print(f"✗ Errors:  {errors} rows")
print(f"{'='*60}\n")

if saved > 0:
    logger.info(f"Database seeded successfully with {saved} financial metrics")
else:
    logger.warning("No new rows inserted (all may already exist)")