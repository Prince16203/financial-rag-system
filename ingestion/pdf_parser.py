"""
ingestion/pdf_parser.py

Extracts text (and tables) from financial PDFs using pdfplumber.
Returns a list of page-level dicts with text content + metadata.

Each dict:
{
    "source":    "apple_10k_2023.pdf",
    "page":      3,               # 1-indexed
    "text":      "...",           # narrative text on the page
    "tables":    [                # list of tables found on the page (may be empty)
        [["Header1", "Header2"], ["val1", "val2"], ...]
    ]
}

Usage:
    from ingestion.pdf_parser import parse_pdf, parse_directory

    pages = parse_pdf("data/raw/apple_10k_2023.pdf")
    all_pages = parse_directory("data/raw/")
"""

import pdfplumber
import os
import logging
from pathlib import Path
from typing import Optional

import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Core: parse a single PDF
# ---------------------------------------------------------------------------

def parse_pdf(pdf_path: str) -> list[dict]:
    """
    Parse a single PDF and return a list of page dicts.

    Args:
        pdf_path: Path to the PDF file.

    Returns:
        List of dicts, one per page, with keys:
        source, page, text, tables.
    """
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    source_name = pdf_path.name
    pages_data = []

    logger.info(f"Parsing: {source_name}")

    with pdfplumber.open(pdf_path) as pdf:
        total_pages = len(pdf.pages)
        logger.info(f"  Total pages: {total_pages}")

        for i, page in enumerate(pdf.pages):
            page_number = i + 1  # 1-indexed for human readability

            # --- Text extraction ---
            # keep_blank_chars=False strips noise; use_text_flow=True
            # respects reading order in multi-column layouts (common in 10-Ks)
            raw_text = page.extract_text(keep_blank_chars=False, use_text_flow=True)
            text = raw_text.strip() if raw_text else ""

            # --- Table extraction ---
            # pdfplumber detects ruled/borderless tables via line detection.
            # Financial filings often have bordered tables (income statements etc.)
            # extract_tables() returns: list[list[list[str | None]]]
            raw_tables = page.extract_tables()
            tables = _clean_tables(raw_tables)

            # Skip completely blank pages (common in 10-K cover sections)
            if not text and not tables:
                logger.debug(f"  Page {page_number}: blank, skipping")
                continue

            pages_data.append({
                "source": source_name,
                "page":   page_number,
                "text":   text,
                "tables": tables,
            })

        logger.info(f"  Extracted {len(pages_data)} non-blank pages from {source_name}")

    return pages_data


# ---------------------------------------------------------------------------
# Batch: parse every PDF in a directory
# ---------------------------------------------------------------------------

def parse_directory(directory: str, skip_on_error: bool = True) -> list[dict]:
    """
    Parse all .pdf files in a directory.

    Args:
        directory:     Path to folder containing PDFs.
        skip_on_error: If True, log errors and continue; otherwise re-raise.

    Returns:
        Flat list of page dicts from all PDFs combined.
    """
    directory = Path(directory)
    pdf_files = sorted(directory.glob("*.pdf"))

    if not pdf_files:
        logger.warning(f"No PDF files found in: {directory}")
        return []

    logger.info(f"Found {len(pdf_files)} PDF(s) in {directory}")

    all_pages = []
    for pdf_file in pdf_files:
        try:
            pages = parse_pdf(str(pdf_file))
            all_pages.extend(pages)
        except Exception as e:
            if skip_on_error:
                logger.error(f"Failed to parse {pdf_file.name}: {e}")
            else:
                raise

    logger.info(f"Total pages extracted: {len(all_pages)}")
    return all_pages


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _clean_tables(raw_tables: list) -> list[list[list[str]]]:
    """
    Normalise pdfplumber table output.
    - Replace None cells with empty string.
    - Skip entirely empty rows.
    """
    if not raw_tables:
        return []

    cleaned = []
    for table in raw_tables:
        clean_table = []
        for row in table:
            clean_row = [cell if cell is not None else "" for cell in row]
            # skip rows that are entirely empty
            if any(cell.strip() for cell in clean_row):
                clean_table.append(clean_row)
        if clean_table:
            cleaned.append(clean_table)
    return cleaned


def table_to_text(table: list[list[str]]) -> str:
    """
    Convert a parsed table (list of rows) to a plain-text representation.
    Used by chunker.py to include table content in the text chunk.

    Example output:
        | Metric | 2023 | 2022 |
        | Revenue | 394B | 365B |
    """
    if not table:
        return ""
    rows = []
    for row in table:
        rows.append("| " + " | ".join(cell.replace("\n", " ") for cell in row) + " |")
    return "\n".join(rows)


# ---------------------------------------------------------------------------
# Quick smoke-test — run this file directly to verify a PDF
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    path = sys.argv[1] if len(sys.argv) > 1 else str(config.RAW_PDF_DIR)

    if os.path.isdir(path):
        pages = parse_directory(path)
    else:
        pages = parse_pdf(path)

    print(f"\n{'='*60}")
    print(f"Extracted {len(pages)} pages total")
    print(f"{'='*60}")

    # Preview first 3 pages
    for p in pages[:3]:
        print(f"\n--- {p['source']}  |  Page {p['page']} ---")
        print(p["text"][:500])
        if p["tables"]:
            print(f"  [Tables on this page: {len(p['tables'])}]")
            print(table_to_text(p["tables"][0])[:300])