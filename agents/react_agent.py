"""
agents/sql_agent.py

Natural language → SQL chain over the financial_metrics SQLite table.
Generates SQL via Groq LLM, validates it (SELECT-only), executes via
SQLAlchemy, and returns formatted results.

This is intentionally lightweight — no langchain-community dependency.
For complex multi-table schemas, swap in create_sql_agent from
langchain_community when that package becomes available.

Usage:
    from agents.sql_agent import SQLAgent

    agent = SQLAgent()
    result = agent.query("Show me all revenue metrics for Apple in 2024")
    print(result)
"""

import logging
import re
from typing import Optional

from groq import Groq
from sqlalchemy import text

import config
from database.db import engine, get_session, init_db
from database.models import FinancialMetric

logger = logging.getLogger(__name__)

# Schema description injected into every prompt so the LLM knows the table
SCHEMA = """
Table: financial_metrics
Columns:
  id            INTEGER  PRIMARY KEY
  company       TEXT     e.g. 'Apple', 'Goldman Sachs', 'JPMorgan', 'BlackRock', 'Microsoft'
  metric_name   TEXT     e.g. 'revenue', 'net_income', 'ficc_revenue', 'equity_revenue',
                              'investment_banking_revenue', 'total_assets', 'eps', 'var',
                              'tier1_capital_ratio'
  value         REAL     numeric value
  unit          TEXT     e.g. 'billion USD', 'million USD', 'percent'
  fiscal_year   INTEGER  4-digit year, may be NULL
  source        TEXT     PDF filename
  page          INTEGER  page number in source PDF
  created_at    DATETIME
"""

SQL_PROMPT = """You are a SQLite expert. Write a SQL SELECT query for the question below.

{schema}

Rules:
- Write ONLY a single SELECT statement
- No INSERT, UPDATE, DELETE, DROP, or any DDL
- Use LOWER() for case-insensitive company/metric comparisons when helpful
- Return ONLY the SQL query, no explanation, no markdown fences

Question: {question}

SQL:"""


class SQLAgent:
    """
    Converts natural language questions into SQL queries over
    the financial_metrics table and returns formatted results.

    Args:
        groq_client: Optional pre-built Groq client.
    """

    def __init__(self, groq_client: Optional[Groq] = None):
        if not config.GROQ_API_KEY:
            raise ValueError("GROQ_API_KEY not set in .env")

        self.client = groq_client or Groq(api_key=config.GROQ_API_KEY)
        init_db()
        logger.info("SQLAgent ready")

    def query(self, question: str) -> str:
        """
        Answer a natural language question by generating and running SQL.

        Args:
            question: Plain English question about financial metrics.

        Returns:
            Formatted string of results, or an error message.
        """
        sql = self._generate_sql(question)
        if not sql:
            return "Could not generate a valid SQL query for this question."

        logger.info(f"Generated SQL: {sql}")
        return self._execute_sql(sql, question)

    def _generate_sql(self, question: str) -> Optional[str]:
        """Ask Groq to write a SELECT query for this question."""
        prompt = SQL_PROMPT.format(schema=SCHEMA, question=question)

        try:
            response = self.client.chat.completions.create(
                model=config.GROQ_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
            )
            raw = response.choices[0].message.content.strip()

            # Strip markdown fences if model added them
            raw = re.sub(r"```sql|```", "", raw).strip()

            # Safety: only allow SELECT statements
            if not raw.upper().lstrip().startswith("SELECT"):
                logger.warning(f"Non-SELECT query rejected: {raw[:100]}")
                return None

            return raw

        except Exception as e:
            logger.error(f"SQL generation error: {e}")
            return None

    def _execute_sql(self, sql: str, question: str) -> str:
        """Execute the SQL and format the results."""
        try:
            with engine.connect() as conn:
                result = conn.execute(text(sql))
                rows = result.fetchall()
                columns = list(result.keys())

            if not rows:
                return (
                    f"No results found for: '{question}'\n"
                    f"SQL used: {sql}"
                )

            # Format as readable table
            lines = [f"Results for: '{question}'", f"SQL: {sql}", ""]
            header = " | ".join(columns)
            lines.append(header)
            lines.append("-" * len(header))

            for row in rows[:20]:  # cap at 20 rows for readability
                lines.append(" | ".join(str(v) for v in row))

            if len(rows) > 20:
                lines.append(f"... ({len(rows) - 20} more rows)")

            logger.info(f"SQL query returned {len(rows)} rows")
            return "\n".join(lines)

        except Exception as e:
            logger.error(f"SQL execution error: {e}")
            return f"SQL execution failed: {e}\nSQL: {sql}"


if __name__ == "__main__":
    import sys
    from utils.logging_config import setup_logging
    setup_logging()

    question = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else \
        "Show me all revenue metrics for Apple"

    agent = SQLAgent()
    print(agent.query(question))