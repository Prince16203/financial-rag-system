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
import time
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
    # RATE LIMIT PROTECTOR: Pause to prevent hitting Groq's 429 TPM limit
    time.sleep(1.5)
    
    session = get_session()
    try:
        query_lower = query.lower()
        db_query = session.query(FinancialMetric)

        # Company filter - Ordered longest to shortest to prevent substring overlap
        company_keywords = {
            "goldman sachs": "Goldman Sachs",
            "blackrock":     "BlackRock",
            "microsoft":     "Microsoft",
            "jpmorgan":      "JPMorgan",
            "jp morgan":     "JPMorgan",
            "goldman":       "Goldman Sachs",
            "apple":         "Apple",
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

        # Metric type filter - Ordered longest to shortest to prevent substring overlap
        metric_keywords = {
            "investment banking revenue": "investment_banking_revenue",
            "investment banking":         "investment_banking_revenue",
            "earnings per share":         "eps",
            "equity revenue":             "equity_revenue",
            "value at risk":              "var",
            "total assets":               "total_assets",
            "net income":                 "net_income",
            "tier 1":                     "tier1_capital_ratio",
            "revenue":                    "revenue",
            "assets":                     "total_assets",
            "tier1":                      "tier1_capital_ratio",
            "ficc":                       "ficc_revenue",
            "eps":                        "eps",
            "var":                        "var",
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
    # RATE LIMIT PROTECTOR
    time.sleep(1.5)
    
    try:
        # FIX: Remove commas so numbers like 47,830 don't evaluate as Python tuples
        expression_clean = expression.replace(",", "")
        
        # Only allow safe math operations — no exec, no imports
        allowed = set("0123456789+-*/()., ")
        if not all(c in allowed for c in expression_clean):
            return "Error: only numeric expressions are allowed."

        # Safe evaluation using only math functions
        result = eval(expression_clean, {"__builtins__": {}}, {"math": math})
        logger.info(f"calculator: {expression_clean} = {result}")
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
    # RATE LIMIT PROTECTOR
    time.sleep(1.5)
    
    try:
        from retrieval.hybrid_retriever import HybridRetriever
        from retrieval.reranker import Reranker

        retriever = HybridRetriever()
        reranker = Reranker()

        raw = retriever.retrieve(question, k=10)
        results = reranker.rerank(question, raw, top_n=3)

        if not results:
            return "No relevant documents found for this question."

        parts = []
        for r in results:
            parts.append(
                f"[{r['source']} | Page {r['page']}]\n{r['text'][:500]}"
            )

        logger.info(f"search_documents: found {len(results)} chunks for '{question[:60]}'")
        return "\n\n---\n\n".join(parts)

    except Exception as e:
        logger.error(f"search_documents error: {e}")
        return f"Document search failed: {e}"


# ---------------------------------------------------------------------------
# Tool registry
# ---------------------------------------------------------------------------

TOOLS = [query_metrics_db, calculator, search_documents]


if __name__ == "__main__":
    print("\n--- Testing query_metrics_db ---")
    print(query_metrics_db.invoke("Apple revenue 2024"))

    print("\n--- Testing calculator ---")
    print(f"Result: {calculator.invoke('(14.52 - 13.34) / 13.34 * 100')}%")