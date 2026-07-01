"""
Page-level financial metric extraction with filename-derived company/year
grounding. Validation uses regex number matching against evidence_text
rather than exact sentence matching, which fails on table data.
"""

import json
import logging
import re
import time
from pathlib import Path
from typing import Optional, Literal

from groq import Groq
from pydantic import BaseModel, Field
from sqlalchemy.exc import IntegrityError
from streamlit import metric

import config
from database.db import get_session, init_db
from database.models import FinancialMetric
from ingestion.pdf_parser import parse_pdf, table_to_text

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Filename -> (company, fiscal_year)
# ---------------------------------------------------------------------------

FILENAME_COMPANY_MAP = {
    "aapl":  "Apple",
    "gs":    "Goldman Sachs",
    "jpm":   "JPMorgan",
    "blk":   "BlackRock",
    "msft":  "Microsoft",
}


def parse_filename(filename: str) -> tuple[Optional[str], Optional[int]]:
    """Derive company and fiscal year from filename — no LLM guessing."""
    name_lower = filename.lower()

    company = None
    for prefix, clean_name in FILENAME_COMPANY_MAP.items():
        if name_lower.startswith(prefix):
            company = clean_name
            break

    year_match = re.search(r"(20\d{2})", filename)
    fiscal_year = int(year_match.group(1)) if year_match else None

    return company, fiscal_year


# ---------------------------------------------------------------------------
# Pydantic schema — uses evidence_text not source_sentence
# ---------------------------------------------------------------------------

class ExtractedMetric(BaseModel):
    """One extracted metric with short evidence snippet."""

    metric_name: Literal[
        "revenue", "net_income", "ficc_revenue", "equity_revenue",
        "investment_banking_revenue", "total_assets", "eps", "var",
        "tier1_capital_ratio",
    ] = Field(description="Must be exactly one of the allowed metric names")

    value: float = Field(description="Numeric value only, no symbols or commas")

    unit: Literal[
        "billion USD", "million USD", "thousand USD", "USD", "percent", "USD per share"
    ] = Field(description="Unit of the value")

    evidence_text: str = Field(
        description="Short text snippet (10-50 chars) from the page that "
        "contains the number. Can be a table row fragment like "
        "'Revenue | 47,830' or a sentence fragment. Keep it short."
    )


class PageExtraction(BaseModel):
    """All metrics found on a single page."""
    metrics: list[ExtractedMetric] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Prompt — asks for evidence_text not source_sentence
# ---------------------------------------------------------------------------

EXTRACTION_PROMPT = """You are a financial data extraction assistant working on a 10-K filing.

This page is from {company}'s fiscal year {fiscal_year} annual report.
The company is ALREADY KNOWN — do not try to determine it from the text.

Extract financial metrics from the page text below. Return ONLY valid JSON
matching this exact structure:
{{
  "metrics": [
    {{
      "metric_name": "one of: revenue, net_income, ficc_revenue, equity_revenue, investment_banking_revenue, total_assets, eps, var, tier1_capital_ratio",
      "value": <number as float, no commas or symbols, e.g. 47830.0 not 47,830>,
      "unit": "one of: billion USD, million USD, thousand USD, USD, percent, USD per share",
      "evidence_text": "<short snippet 10-50 chars from the page containing the number, e.g. 'Net revenues 47,830'>"
    }}
  ]
}}

CRITICAL RULES:
- value must be a plain float — no commas, no dollar signs, no billion/million words
- evidence_text should be a SHORT fragment (not a full sentence) from the page near the number
- Only extract metrics for {company} — ignore S&P 500, Dow Jones, money market funds, other companies
- Ignore subtotals and "Total" rows unless they are {company}'s consolidated figure
- If no metrics found, return {{"metrics": []}}
- Do NOT invent or estimate numbers

PAGE TEXT:
{text}
"""


# ---------------------------------------------------------------------------
# Unit normalization
# ---------------------------------------------------------------------------

def normalize_to_billions(value: float, unit: str) -> Optional[float]:
    """Convert to billions USD. Returns None for non-currency units."""
    unit_lower = unit.lower().strip()

    if "billion" in unit_lower:
        return round(value, 4)
    elif "million" in unit_lower:
        return round(value / 1000, 4)
    elif "thousand" in unit_lower:
        return round(value / 1_000_000, 4)
    elif unit_lower == "usd":
        return round(value / 1_000_000_000, 4)
    else:
        return None


# ---------------------------------------------------------------------------
# Grounding verification — regex number matching, not exact string match
# ---------------------------------------------------------------------------

def extract_numbers_from_text(text: str) -> list[float]:
    """
    Extract all numbers from a text string, handling commas and decimals.
    '47,830.5 million' -> [47830.5]
    '$14.52 billion'   -> [14.52]
    """
    pattern = r"[-+]?\d[\d,]*\.?\d*"
    matches = re.findall(pattern, text)
    numbers = []
    for m in matches:
        try:
            numbers.append(float(m.replace(",", "")))
        except ValueError:
            pass
    return numbers


def verify_grounding(metric: ExtractedMetric, page_text: str) -> bool:
    """
    Verify that:
    1. evidence_text appears (approximately) in the page
    2. A number close to metric.value appears in evidence_text

    Uses regex number extraction instead of exact string matching
    to handle table formatting, comma differences, and rounding.
    """
    # FIX: Added .lower() for case-insensitive matching
    page_normalized = " ".join(page_text.split()).lower()
    evidence_normalized = " ".join(metric.evidence_text.split()).lower()

    # Check evidence_text appears in page (relaxed — first 6 words)
    evidence_words = evidence_normalized.split()
    if len(evidence_words) >= 3:
        check_fragment = " ".join(evidence_words[:min(6, len(evidence_words))])
        if check_fragment not in page_normalized:
            logger.debug(
                f"REJECT: evidence_text not found in page | "
                f"fragment='{check_fragment}' | metric={metric.metric_name}"
            )
            return False

    # Extract numbers from evidence_text and check one is close to metric.value
    numbers_in_evidence = extract_numbers_from_text(metric.evidence_text)

    if not numbers_in_evidence:
        logger.debug(
            f"REJECT: no numbers found in evidence_text | "
            f"evidence='{metric.evidence_text}' | metric={metric.metric_name}"
        )
        return False

    # Allow 1% tolerance for rounding differences
    tolerance = abs(metric.value) * 0.01 + 0.01
    match_found = any(
        abs(n - metric.value) <= tolerance
        for n in numbers_in_evidence
    )

    if not match_found:
        logger.debug(
            f"REJECT: value {metric.value} not found in evidence numbers "
            f"{numbers_in_evidence} | evidence='{metric.evidence_text}'"
        )
        return False

    return True


# ---------------------------------------------------------------------------
# MetricExtractor
# ---------------------------------------------------------------------------

class MetricExtractor:
    """
    Page-level financial metric extractor with company/year grounding
    and regex-based evidence verification.
    """

    def __init__(self):
        if not config.GROQ_API_KEY:
            raise ValueError("GROQ_API_KEY not set in .env")

        self.client = Groq(api_key=config.GROQ_API_KEY)
        init_db()
        logger.info("MetricExtractor ready (page-level, evidence-verified)")

    def run_on_pdf(self, pdf_path: str, max_pages: Optional[int] = None) -> dict:
        """Extract metrics from one PDF, page by page."""
        pdf_path = Path(pdf_path)
        company, fiscal_year = parse_filename(pdf_path.name)

        if company is None:
            logger.warning(f"Unknown company in filename: {pdf_path.name} — skipping")
            return {"processed": 0, "extracted": 0, "saved": 0,
                    "rejected_grounding": 0, "rejected_plausibility": 0, "duplicates": 0}

        logger.info(f"Processing {pdf_path.name} | company={company} fiscal_year={fiscal_year}")

        pages = parse_pdf(str(pdf_path))
        if max_pages:
            pages = pages[:max_pages]

        stats = {
            "processed": 0, "extracted": 0, "saved": 0,
            "rejected_grounding": 0, "rejected_plausibility": 0, "duplicates": 0
        }

        for page in pages:
            page_text = self._build_page_text(page)

            if len(page_text.strip()) < 150:
                continue

            stats["processed"] += 1

            extraction = self._extract_page(
                text=page_text,
                company=company,
                fiscal_year=fiscal_year,
            )

            if not extraction:
                continue

            stats["extracted"] += len(extraction.metrics)

            for metric in extraction.metrics:
                if not verify_grounding(metric, page_text):
                    stats["rejected_grounding"] += 1
                    continue

                result = self._save_metric(
                    metric=metric,
                    company=company,
                    fiscal_year=fiscal_year,
                    source=pdf_path.name,
                    page=page["page"],
                )

                if result == "saved":
                    stats["saved"] += 1
                elif result == "duplicate":
                    stats["duplicates"] += 1
                elif result == "implausible":
                    stats["rejected_plausibility"] += 1
                else:
                    stats["rejected_grounding"] += 1
            
            # FIX: Add a small delay to respect Groq's rate limits
            time.sleep(2.5) 

        logger.info(
            f"{pdf_path.name} done | "
            f"pages={stats['processed']} extracted={stats['extracted']} "
            f"saved={stats['saved']} rej_grounding={stats['rejected_grounding']} "
            f"rej_plausibility={stats['rejected_plausibility']} "
            f"duplicates={stats['duplicates']}"
        )

        return stats

    def run_on_directory(
        self, directory: str, max_pages_per_pdf: Optional[int] = None
    ) -> dict:
        """Run extraction across every PDF in a directory."""
        directory = Path(directory)
        pdf_files = sorted(directory.glob("*.pdf"))

        if not pdf_files:
            logger.warning(f"No PDFs found in {directory}")
            return {}

        totals = {
            "processed": 0, "extracted": 0, "saved": 0,
            "rejected_grounding": 0, "rejected_plausibility": 0, "duplicates": 0
        }

        for pdf_file in pdf_files:
            stats = self.run_on_pdf(str(pdf_file), max_pages=max_pages_per_pdf)
            for k in totals:
                totals[k] += stats.get(k, 0)

        logger.info(f"Directory extraction complete | totals={totals}")
        return totals

    def get_stats(self) -> dict:
        """Return current database stats."""
        session = get_session()
        try:
            total = session.query(FinancialMetric).count()
            companies = session.query(FinancialMetric.company).distinct().all()
            metrics = session.query(FinancialMetric.metric_name).distinct().all()
            return {
                "total_metrics": total,
                "companies": [c[0] for c in companies],
                "metric_types": [m[0] for m in metrics],
            }
        finally:
            session.close()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _build_page_text(self, page: dict) -> str:
        """Combine narrative text and tables."""
        parts = []
        if page.get("text"):
            parts.append(page["text"])
        if page.get("tables"):
            for table in page["tables"]:
                table_text = table_to_text(table)
                if table_text.strip():
                    parts.append(f"\n[TABLE]\n{table_text}\n[/TABLE]")
        return "\n\n".join(parts)

    def _extract_page(
        self,
        text: str,
        company: str,
        fiscal_year: Optional[int],
    ) -> Optional[PageExtraction]:
        """Call Groq with JSON mode and validate Pydantic schema."""
        prompt = EXTRACTION_PROMPT.format(
            company=company,
            fiscal_year=fiscal_year or "unknown",
            text=text[:8000],  # increased from 6000
        )

        try:
            response = self.client.chat.completions.create(
                model=config.GROQ_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                response_format={"type": "json_object"},
            )
            raw = response.choices[0].message.content.strip()
            data = json.loads(raw)
            return PageExtraction.model_validate(data)

        except json.JSONDecodeError:
            logger.debug("JSON parse failed for a page")
            return None
        except Exception as e:
            logger.error(f"Extraction error: {e}")
            return None

    def _save_metric(
        self,
        metric: ExtractedMetric,
        company: str,
        fiscal_year: Optional[int],
        source: str,
        page: int,
    ) -> str:
        """
        Normalize, plausibility-check, and save metric.
        Returns: 'saved', 'duplicate', 'implausible', or 'error'
        """
        normalized = normalize_to_billions(metric.value, metric.unit)

        if normalized is not None and abs(normalized) > 5000:
            logger.debug(
                f"Implausible: {metric.metric_name}={normalized:.1f}B "
                f"from {source} p{page}"
            )
            return "implausible"

        session = get_session()
        try:
            row = FinancialMetric(
                company=company,
                metric_name=metric.metric_name,
                value=metric.value,
                unit=metric.unit,
                fiscal_year=fiscal_year,
                source=source,
                page=page,
            )
            if hasattr(row, "value_normalized"):
                row.value_normalized = normalized

            session.add(row)
            session.commit()
            logger.info(
                f"SAVED: {company} | {metric.metric_name} | "
                f"{metric.value} {metric.unit} | FY{fiscal_year} | p{page}"
            )
            return "saved"

        except IntegrityError:
            session.rollback()
            return "duplicate"
        except Exception as e:
            logger.error(f"DB save error: {e}")
            session.rollback()
            return "error"
        finally:
            session.close()


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    pdf_arg = sys.argv[1] if len(sys.argv) > 1 else "data/raw_pdfs"
    max_pages = int(sys.argv[2]) if len(sys.argv) > 2 else 20

    extractor = MetricExtractor()

    path = Path(pdf_arg)
    if path.is_dir():
        print(f"\nRunning on all PDFs in {path}, max {max_pages} pages each...")
        stats = extractor.run_on_directory(str(path), max_pages_per_pdf=max_pages)
    else:
        print(f"\nRunning on {path}, max {max_pages} pages...")
        stats = extractor.run_on_pdf(str(path), max_pages=max_pages)

    print(f"\n{'='*60}")
    print("Extraction stats:")
    for k, v in stats.items():
        print(f"  {k}: {v}")

    print("\nDatabase stats:")
    db_stats = extractor.get_stats()
    for k, v in db_stats.items():
        print(f"  {k}: {v}")
    print(f"{'='*60}")