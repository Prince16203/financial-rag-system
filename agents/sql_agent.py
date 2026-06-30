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

# ---------------------------------------------------------------------------
# SQL Schema Description
# ---------------------------------------------------------------------------

SCHEMA = """
Table: financial_metrics
  Columns:
    - id (INTEGER PRIMARY KEY): auto-incrementing row ID
    - company (STRING): company name, e.g. "Apple", "Goldman Sachs", "JPMorgan", "BlackRock", "Microsoft"
    - metric_name (STRING): metric type, one of:
        * revenue
        * net_income
        * ficc_revenue
        * equity_revenue
        * investment_banking_revenue
        * total_assets
        * eps (earnings per share)
        * var (value at risk)
        * tier1_capital_ratio
    - value (FLOAT): numeric value of the metric
    - unit (STRING): unit of measurement, e.g. "billion USD", "million USD", "USD", "percent"
    - fiscal_year (INTEGER): fiscal year (nullable), e.g. 2024, 2023, 2022
    - source (STRING): source PDF filename, e.g. "aapl-20240928.pdf"
    - page (INTEGER): page number within the PDF
    - created_at (DATETIME): timestamp when row was inserted

Example rows:
    1 | Apple | revenue | 391.04 | billion USD | 2024 | aapl-20240928.pdf | 32 | 2024-06-30 12:00:00
    2 | Goldman Sachs | ficc_revenue | 14.52 | billion USD | 2025 | gs-20251231.pdf | 193 | 2024-06-30 12:00:00
    3 | JPMorgan | net_income | 50.15 | billion USD | 2024 | jpm-20241231.pdf | 45 | 2024-06-30 12:00:00
"""

# ---------------------------------------------------------------------------
# SQL Generation Prompt
# ---------------------------------------------------------------------------

SQL_PROMPT = """You are a financial database expert writing SQL queries.

Schema:
{schema}

Your task: Generate a single SELECT query to answer this question.

Question: {question}

Rules:
  1. Return ONLY valid SQL — no explanation, no markdown, no backticks
  2. SELECT statements only — NO INSERT, UPDATE, DELETE, DROP
  3. Always filter by exact company names (use LIKE if unsure: WHERE company LIKE '%Apple%')
  4. If a specific year is mentioned, filter fiscal_year = <year>
  5. If no year mentioned, return all years (no year filter)
  6. Always include: company, metric_name, value, unit, fiscal_year, source, page
  7. Order results by fiscal_year DESC, then metric_name
  8. Limit to 100 rows max for readability

Example queries:
  - "Show Apple revenue for 2024"
    SELECT * FROM financial_metrics WHERE company = 'Apple' AND metric_name = 'revenue' AND fiscal_year = 2024
  
  - "Compare net income of Apple and Goldman Sachs"
    SELECT company, value, fiscal_year, unit FROM financial_metrics WHERE company IN ('Apple', 'Goldman Sachs') AND metric_name = 'net_income' ORDER BY company, fiscal_year DESC
  
  - "All metrics for Microsoft"
    SELECT * FROM financial_metrics WHERE company = 'Microsoft' ORDER BY fiscal_year DESC

Now generate the query:
"""

# ---------------------------------------------------------------------------
# System Message for Multi-Turn Conversation
# ---------------------------------------------------------------------------

SYSTEM_MESSAGE = """You are a financial document analyst with access to tools.

You have three tools available:
1. query_metrics_db — use for specific financial numbers (revenue, net income, FICC revenue etc.)
2. calculator — use for any arithmetic (percentage change, ratios, differences)
3. search_documents — use for qualitative questions (risk factors, business strategy, descriptions)

IMPORTANT RULES:
- query_metrics_db only accepts ONE company and ONE metric at a time. 
  For comparisons, call it separately for each company.
- calculator only accepts numeric expressions with actual numbers — 
  never use placeholder names like result1 or result2. 
  Read the actual numeric values from previous tool outputs and substitute them directly.
- If query_metrics_db returns no results for a specific year, try without 
  specifying the year, or try search_documents as a fallback.
- Always state the actual year used if it differs from what was asked.

Always follow this process:
- Think about what information you need
- Use query_metrics_db FIRST for any numeric financial question, one company per call
- Once you have two numbers, use calculator with the literal numbers, not variable names
- Use search_documents for context and narrative questions
- Always cite the source PDF and page in your final answer
"""


# ---------------------------------------------------------------------------
# SQLAgent Class
# ---------------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # Convenience: check if database is populated
    # ------------------------------------------------------------------

    def get_metrics_count(self) -> int:
        """Return total number of rows in the metrics table."""
        session = get_session()
        try:
            return session.query(FinancialMetric).count()
        finally:
            session.close()

    def list_companies(self) -> list[str]:
        """Return unique company names in the database."""
        session = get_session()
        try:
            companies = session.query(FinancialMetric.company).distinct().all()
            return [c[0] for c in companies]
        finally:
            session.close()

    def list_metrics(self) -> list[str]:
        """Return unique metric names in the database."""
        session = get_session()
        try:
            metrics = session.query(FinancialMetric.metric_name).distinct().all()
            return [m[0] for m in metrics]
        finally:
            session.close()


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    from utils.logging_config import setup_logging
    setup_logging()

    question = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else \
        "Show me all revenue metrics for Apple"

    agent = SQLAgent()
    
    print(f"\nDatabase stats:")
    print(f"  Total metrics: {agent.get_metrics_count()}")
    print(f"  Companies: {agent.list_companies()}")
    print(f"  Metric types: {agent.list_metrics()}")
    
    print(f"\nQuery: {question}")
    result = agent.query(question)
    print(f"\n{result}")