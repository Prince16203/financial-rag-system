"""
chains/metric_extractor.py

Reads chunks from ChromaDB, uses Groq LLM to extract structured
financial metrics, and saves them to the SQLite database.

Metrics extracted:
    - revenue / net_revenue
    - net_income / net_loss
    - ficc_revenue
    - equity_revenue
    - investment_banking_revenue
    - total_assets
    - eps (earnings per share)
    - var (value at risk)
    - tier1_capital_ratio

Usage:
    from chains.metric_extractor import MetricExtractor

    extractor = MetricExtractor()
    extractor.run()               # processes all chunks
    extractor.run(limit=50)       # process only first 50 chunks (testing)
"""

import json
import logging
import re
from typing import Optional

from groq import Groq
from sqlalchemy.exc import IntegrityError

import config
from database.db import get_session, init_db
from database.models import FinancialMetric

import chromadb
from ingestion.indexer import DEFAULT_COLLECTION, DEFAULT_CHROMA_PATH

# logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Company name mapping from PDF filename to clean company name
COMPANY_MAP = {
    "aapl":  "Apple",
    "apple": "Apple",
    "gs":    "Goldman Sachs",
    "jpm":   "JPMorgan",
    "blk":   "BlackRock",
    "msft":  "Microsoft",
}

EXTRACTION_PROMPT = """You are a financial data extraction assistant.

Extract financial metrics from the text below. Return ONLY a JSON array.
Each element must have exactly these fields:
- company: company name (string)
- metric_name: one of [revenue, net_income, ficc_revenue, equity_revenue,
  investment_banking_revenue, total_assets, eps, var, tier1_capital_ratio]
- value: numeric value only, no symbols (float)
- unit: e.g. "billion USD", "million USD", "USD", "percent" (string)
- fiscal_year: 4-digit year as integer, or null if not mentioned

Rules:
- Only extract metrics that are explicitly stated with a numeric value
- Do NOT invent or estimate values
- If no metrics found, return an empty array: []
- Return ONLY the JSON array, no explanation, no markdown

Text:
{text}

Source PDF: {source}
Page: {page}
"""


class MetricExtractor:
    """
    Extracts structured financial metrics from ChromaDB chunks
    and stores them in SQLite via SQLAlchemy.

    Args:
        collection_name: ChromaDB collection to read from.
        chroma_path:     Path to ChromaDB persistence directory.
    """

    def __init__(
        self,
        collection_name: str = DEFAULT_COLLECTION,
        chroma_path: str = DEFAULT_CHROMA_PATH,
    ):
        if not config.GROQ_API_KEY:
            raise ValueError("GROQ_API_KEY not set in .env")

        self.client = Groq(api_key=config.GROQ_API_KEY)

        # Connect to ChromaDB
        chroma_client = chromadb.PersistentClient(path=chroma_path)
        self.collection = chroma_client.get_collection(name=collection_name)

        # Initialise SQLite database
        init_db()

        logger.info(
            f"MetricExtractor ready | "
            f"chunks={self.collection.count()}"
        )

    def run(self, limit: Optional[int] = None, batch_size: int = 50) -> dict:
        """
        Process all chunks and extract metrics into the database.

        Args:
            limit:      Max chunks to process. None = all chunks.
                        Use limit=100 for testing before full run.
            batch_size: How many chunks to fetch from ChromaDB at once.

        Returns:
            Stats dict with total processed, extracted, saved, skipped.
        """
        total = self.collection.count()
        process_count = min(limit, total) if limit else total

        logger.info(f"Starting extraction: {process_count} chunks to process")

        stats = {
            "processed": 0,
            "extracted": 0,
            "saved":     0,
            "skipped":   0,
            "errors":    0,
        }

        # Process in batches
        offset = 0
        while offset < process_count:
            fetch = min(batch_size, process_count - offset)

            results = self.collection.get(
                limit=fetch,
                offset=offset,
                include=["documents", "metadatas"],
            )

            for text, meta in zip(results["documents"], results["metadatas"]):
                stats["processed"] += 1

                # Skip very short chunks — unlikely to have metrics
                if len(text.strip()) < 100:
                    stats["skipped"] += 1
                    continue

                # Extract metrics from this chunk
                metrics = self._extract_metrics(
                    text=text,
                    source=meta.get("source", "unknown"),
                    page=meta.get("page", 0),
                )

                stats["extracted"] += len(metrics)

                # Save each metric to database
                for metric in metrics:
                    saved = self._save_metric(metric, meta)
                    if saved:
                        stats["saved"] += 1

            offset += fetch
            logger.info(
                f"Progress: {min(offset, process_count)}/{process_count} chunks | "
                f"saved={stats['saved']} metrics so far"
            )

        logger.info(
            f"Extraction complete | "
            f"processed={stats['processed']} | "
            f"extracted={stats['extracted']} | "
            f"saved={stats['saved']} | "
            f"skipped={stats['skipped']}"
        )
        return stats

    def _extract_metrics(self, text: str, source: str, page: int) -> list[dict]:
        """
        Send one chunk to Groq and parse the returned JSON array.

        Returns list of metric dicts. Empty list if nothing found or on error.
        """
        prompt = EXTRACTION_PROMPT.format(
            text=text[:2000],  # truncate to avoid token limits
            source=source,
            page=page,
        )

        try:
            response = self.client.chat.completions.create(
                model=config.GROQ_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,  # zero temp for deterministic extraction
            )
            raw = response.choices[0].message.content.strip()

            # Strip markdown fences if model added them
            raw = re.sub(r"```json|```", "", raw).strip()

            metrics = json.loads(raw)

            if not isinstance(metrics, list):
                return []

            return metrics

        except json.JSONDecodeError:
            logger.debug(f"JSON parse failed for {source} p{page}")
            return []
        except Exception as e:
            logger.error(f"Groq error on {source} p{page}: {e}")
            return []

    def _save_metric(self, metric: dict, meta: dict) -> bool:
        """
        Validate and save one extracted metric to SQLite.

        Returns True if saved, False if skipped or duplicate.
        """
        # Validate required fields
        required = ["company", "metric_name", "value"]
        if not all(k in metric for k in required):
            return False

        # Clean company name using mapping
        company_raw = str(metric["company"]).lower()
        company = metric["company"]
        for key, clean_name in COMPANY_MAP.items():
            if key in company_raw:
                company = clean_name
                break

        # Validate value is numeric
        try:
            value = float(metric["value"])
        except (TypeError, ValueError):
            return False

        session = get_session()
        try:
            row = FinancialMetric(
                company=company,
                metric_name=str(metric["metric_name"]).lower().strip(),
                value=value,
                unit=str(metric.get("unit", "")).strip(),
                fiscal_year=metric.get("fiscal_year"),
                source=meta.get("source", "unknown"),
                page=meta.get("page", 0),
            )
            session.add(row)
            session.commit()
            return True

        except IntegrityError:
            # Duplicate — already extracted this metric from this source
            session.rollback()
            return False
        except Exception as e:
            logger.error(f"DB save error: {e}")
            session.rollback()
            return False
        finally:
            session.close()

    def get_stats(self) -> dict:
        """Return current database stats."""
        session = get_session()
        try:
            from sqlalchemy import func
            total = session.query(FinancialMetric).count()
            companies = session.query(
                FinancialMetric.company
            ).distinct().all()
            metrics = session.query(
                FinancialMetric.metric_name
            ).distinct().all()

            return {
                "total_metrics": total,
                "companies": [c[0] for c in companies],
                "metric_types": [m[0] for m in metrics],
            }
        finally:
            session.close()


# ---------------------------------------------------------------------------
# Smoke test — run on first 100 chunks only for testing
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    from utils.logging_config import setup_logging
    setup_logging()

    limit = int(sys.argv[1]) if len(sys.argv) > 1 else 100

    extractor = MetricExtractor()

    print(f"\nRunning metric extraction on first {limit} chunks...")
    stats = extractor.run(limit=limit)

    print(f"\n{'='*60}")
    print("Extraction stats:")
    for k, v in stats.items():
        print(f"  {k}: {v}")

    print("\nDatabase stats:")
    db_stats = extractor.get_stats()
    for k, v in db_stats.items():
        print(f"  {k}: {v}")
    print(f"{'='*60}")