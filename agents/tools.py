"""
agents/tools.py

Defines the tools available to the ReAct agent.
Each tool is a function wrapped with @tool decorator.

Tools:
    1. query_metrics_db   — SQL query on financial_metrics table
    2. calculator         — safe arithmetic evaluation
    3. search_documents   — falls back to RAG pipeline for text questions

The agent reads each tool's docstring to decide which one to use.
Docstrings are the most important part — write them clearly.
"""

import logging
import math
from typing import Optional

from langchain_core.tools import tool

from database.db import get_session
from database.models import FinancialMetric

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Tool 1 — SQL Database Query
# ---------------------------------------------------------------------------

@tool
def query_metrics_db(query: str) -> str:
    """
    Query the financial metrics database to get structured numbers.

    Use this tool when the user asks for specific financial metrics
    such as revenue, net income, FICC revenue, total assets, EPS,
    VaR, or any numeric financial figure for a specific company and year.

    Input format: natural language description of what you want.
    Examples:
        "Apple revenue 2024"
        "Goldman Sachs FICC revenue 2025"
        "JPMorgan net income 2024"
        "all metrics for Microsoft"
        "compare revenue for Apple and Goldman"

    Returns a formatted string of matching metrics from the database.
    """
    session = get_session()
    try:
        query_lower = query.lower()

        # Build filters based on keywords in the query
        db_query = session.query(FinancialMetric)

        # Company filter
        company_keywords = {
            "apple":         "Apple",
            "goldman":       "Goldman Sachs",
            "goldman sachs": "Goldman Sachs",
            "jpmorgan":      "JPMorgan",
            "jp morgan":     "JPMorgan",
            "blackrock":     "BlackRock",
            "microsoft":     "Microsoft",
            "msft":          "Microsoft",
        }

        matched_company = None
        for keyword, company_name in company_keywords.items():
            if keyword in query_lower:
                matched_company = company_name
                break

        if matched_company:
            db_query = db_query.filter(
                FinancialMetric.company == matched_company
            )

        # Metric type filter
        metric_keywords = {
            "revenue":            "revenue",
            "net income":         "net_income",
            "ficc":               "ficc_revenue",
            "equity revenue":     "equity_revenue",
            "investment banking": "investment_banking_revenue",
            "total assets":       "total_assets",
            "assets":             "total_assets",
            "eps":                "eps",
            "earnings per share": "eps",
            "var":                "var",
            "value at risk":      "var",
            "tier1":              "tier1_capital_ratio",
            "tier 1":             "tier1_capital_ratio",
        }

        for keyword, metric_name in metric_keywords.items():
            if keyword in query_lower:
                db_query = db_query.filter(
                    FinancialMetric.metric_name == metric_name
                )
                break

        # Year filter
        for year in range(2020, 2027):
            if str(year) in query:
                db_query = db_query.filter(
                    FinancialMetric.fiscal_year == year
                )
                break

        results = db_query.limit(10).all()

        if not results:
            return (
                f"No metrics found in database for query: '{query}'. "
                "Try the search_documents tool instead for text-based answers."
            )

        # Format results as readable string
        lines = [f"Financial metrics matching '{query}':"]
        for r in results:
            lines.append(
                f"  {r.company} | {r.metric_name} | "
                f"{r.value} {r.unit} | "
                f"FY{r.fiscal_year} | "
                f"{r.source} p.{r.page}"
            )

        logger.info(f"query_metrics_db: found {len(results)} results for '{query}'")
        return "\n".join(lines)

    except Exception as e:
        logger.error(f"query_metrics_db error: {e}")
        return f"Database query failed: {e}"
    finally:
        session.close()


# ---------------------------------------------------------------------------
# Tool 2 — Calculator
# ---------------------------------------------------------------------------

@tool
def calculator(expression: str) -> str:
    """
    Perform safe arithmetic calculations.

    Use this tool when you need to calculate something — percentage
    differences, growth rates, ratios, sums, or any math operation.

    Input: a valid Python arithmetic expression as a string.
    Examples:
        "14.52 - 13.34"
        "(14.52 - 13.34) / 13.34 * 100"
        "391.04 / 4"
        "14.52 / 19.8 * 100"

    Returns the numeric result as a string.
    Do NOT include variable names — only numbers and operators.
    """
    try:
        # Only allow safe math operations — no exec, no imports
        allowed = set("0123456789+-*/()., ")
        if not all(c in allowed for c in expression):
            return "Error: only numeric expressions are allowed."

        # Safe evaluation using only math functions
        result = eval(expression, {"__builtins__": {}}, {"math": math})
        logger.info(f"calculator: {expression} = {result}")
        return str(round(float(result), 4))

    except ZeroDivisionError:
        return "Error: division by zero."
    except Exception as e:
        return f"Calculation error: {e}"


# ---------------------------------------------------------------------------
# Tool 3 — Document Search (RAG fallback)
# ---------------------------------------------------------------------------

@tool
def search_documents(question: str) -> str:
    """
    Search the financial documents using hybrid retrieval and reranking.

    Use this tool when:
    - The question requires qualitative information (risks, strategies,
      business descriptions, regulatory disclosures)
    - The metrics database does not have the answer
    - The question asks about context, explanations, or narratives

    Do NOT use this for exact numbers — use query_metrics_db instead.

    Input: a natural language question about the financial documents.
    Examples:
        "What are the main risk factors disclosed by Goldman Sachs?"
        "What is Apple's strategy for services revenue growth?"
        "What does JPMorgan say about credit risk exposure?"

    Returns relevant text passages from the indexed documents.
    """
    try:
        # Lazy imports to avoid circular dependencies
        from retrieval.hybrid_retriever import HybridRetriever
        from retrieval.reranker import Reranker

        retriever = HybridRetriever()
        reranker = Reranker()

        raw = retriever.retrieve(question, k=10)
        results = reranker.rerank(question, raw, top_n=3)

        if not results:
            return (
                "No relevant documents found for this question. "
                "The indexed documents may not contain this information."
            )

        parts = []
        for r in results:
            parts.append(
                f"[{r['source']} | Page {r['page']}]\n{r['text'][:500]}"
            )

        logger.info(
            f"search_documents: found {len(results)} chunks for '{question[:60]}'"
        )
        return "\n\n---\n\n".join(parts)

    except ValueError as e:
        # ChromaDB collection not found or empty
        if "not found" in str(e).lower() or "empty" in str(e).lower():
            return (
                "No indexed documents available. "
                "Please run the ingestion pipeline first: "
                "`python -m ingestion.indexer`"
            )
        logger.error(f"search_documents ValueError: {e}")
        return f"Document search error: {e}"
    
    except Exception as e:
        # Catch all other errors (timeouts, connection errors, etc.)
        logger.error(f"search_documents error: {e}")
        return (
            f"Document search failed: {e}. "
            "This might be a temporary issue — try again in a moment."
        )


# ---------------------------------------------------------------------------
# Tool registry — used by react_agent.py
# ---------------------------------------------------------------------------

TOOLS = [query_metrics_db, calculator, search_documents]


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    from utils.logging_config import setup_logging
    setup_logging()

    print("\n" + "="*60)
    print("Testing tools.py")
    print("="*60)

    print("\n--- Testing query_metrics_db ---")
    try:
        result = query_metrics_db.invoke({"query": "Apple revenue 2024"})
        print(result)
    except Exception as e:
        print(f"Error (expected if DB empty): {e}")

    print("\n--- Testing calculator ---")
    result = calculator.invoke({"expression": "(14.52 - 13.34) / 13.34 * 100"})
    print(f"Result: {result}%")

    print("\n--- Testing search_documents ---")
    try:
        result = search_documents.invoke({
            "question": "What are Goldman Sachs main risk factors?"
        })
        print(result[:400])
    except Exception as e:
        print(f"Error (expected if index empty): {e}")

    print("\n" + "="*60)
    print("All tools tested")
    print("="*60)