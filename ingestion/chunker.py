"""
ingestion/chunker.py

Splits parsed page dicts (from pdf_parser.py) into smaller, overlapping
text chunks suitable for embedding and retrieval.

Strategy used: RecursiveCharacterTextSplitter
- Tries to split on paragraphs → sentences → words → characters (in order)
- Keeps financial sentences intact as long as possible
- Tables are serialised to text and appended to their page's text before splitting

Each output chunk dict:
{
    "source":       "apple_10k_2023.pdf",
    "page":         3,
    "chunk_index":  0,          # position of this chunk within the page
    "text":         "...",      # the actual chunk text (sent to embedder)
    "char_count":   412,
}

Defaults (chunk_size / chunk_overlap) are pulled from config.py.

Usage:
    from ingestion.chunker import chunk_pages
    from ingestion.pdf_parser import parse_directory

    pages  = parse_directory("data/raw/")
    chunks = chunk_pages(pages)
"""

import logging
from typing import Optional

from langchain_text_splitters import RecursiveCharacterTextSplitter

import config
from ingestion.pdf_parser import table_to_text

# logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def chunk_pages(
    pages: list[dict],
    chunk_size: int = config.CHUNK_SIZE,
    chunk_overlap: int = config.CHUNK_OVERLAP,
    include_tables: bool = True,
) -> list[dict]:
    """
    Convert a list of page dicts (from pdf_parser) into chunk dicts.

    Args:
        pages:          Output of parse_pdf() or parse_directory().
        chunk_size:     Target character length per chunk.
        chunk_overlap:  Character overlap between consecutive chunks.
        include_tables: If True, serialise tables to text and append to page text.

    Returns:
        List of chunk dicts, each with: source, page, chunk_index, text, char_count.
    """
    splitter = _build_splitter(chunk_size, chunk_overlap)

    all_chunks = []
    skipped = 0

    for page in pages:
        full_text = _build_page_text(page, include_tables)

        if not full_text.strip():
            skipped += 1
            continue

        raw_chunks = splitter.split_text(full_text)

        for idx, chunk_text in enumerate(raw_chunks):
            chunk_text = chunk_text.strip()
            if not chunk_text:
                continue

            all_chunks.append({
                "source":      page["source"],
                "page":        page["page"],
                "chunk_index": idx,
                "text":        chunk_text,
                "char_count":  len(chunk_text),
            })

    logger.info(
        f"Chunking complete: {len(pages)} pages → {len(all_chunks)} chunks "
        f"(skipped {skipped} empty pages)"
    )
    return all_chunks


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_splitter(chunk_size: int, chunk_overlap: int) -> RecursiveCharacterTextSplitter:
    """
    Build the text splitter with financial-document-aware separators.

    Priority order (tried top-down; falls through on no match):
      1. Double newline          → paragraph break
      2. Single newline          → line break
      3. ". " / "! " / "? "     → sentence boundary
      4. "; " / ", "            → clause boundary
      5. " "                     → word boundary
      6. ""                      → hard character split (last resort)
    """
    separators = [
        "\n\n",   # paragraph
        "\n",     # line
        ". ",     # sentence end
        "! ",
        "? ",
        "; ",
        ", ",
        " ",
        "",
    ]
    return RecursiveCharacterTextSplitter(
        separators=separators,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        is_separator_regex=False,
    )


def _build_page_text(page: dict, include_tables: bool) -> str:
    """
    Combine narrative text + serialised tables into a single string for splitting.

    Table text is appended after the narrative with a clear separator so the
    splitter can break between them if needed.
    """
    parts = []

    # Narrative text
    if page.get("text"):
        parts.append(page["text"])

    # Tables → plain text rows
    if include_tables and page.get("tables"):
        for table in page["tables"]:
            table_text = table_to_text(table)
            if table_text.strip():
                parts.append(f"\n[TABLE]\n{table_text}\n[/TABLE]")

    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Convenience: inspect chunk distribution
# ---------------------------------------------------------------------------

def chunk_stats(chunks: list[dict]) -> dict:
    """
    Return basic stats about the chunk collection.
    Useful for tuning chunk_size before embedding.
    """
    if not chunks:
        return {}

    sizes = [c["char_count"] for c in chunks]
    sources = {c["source"] for c in chunks}

    return {
        "total_chunks":  len(chunks),
        "unique_sources": len(sources),
        "avg_chars":     round(sum(sizes) / len(sizes)),
        "min_chars":     min(sizes),
        "max_chars":     max(sizes),
        "sources":       sorted(sources),
    }


# ---------------------------------------------------------------------------
# Smoke-test
# ---------------------------------------------------------------------------
from utils.logging_config import setup_logging
setup_logging()


if __name__ == "__main__":
    import sys
    from ingestion.pdf_parser import parse_pdf, parse_directory
    import os

    path = sys.argv[1] if len(sys.argv) > 1 else str(config.RAW_PDF_DIR)

    if os.path.isdir(path):
        pages = parse_directory(path)
    else:
        pages = parse_pdf(path)

    chunks = chunk_pages(pages)
    stats  = chunk_stats(chunks)

    print(f"\n{'='*60}")
    print("Chunk stats:")
    for k, v in stats.items():
        print(f"  {k}: {v}")
    print(f"{'='*60}")

    # Preview first 2 chunks
    for c in chunks[:2]:
        print(f"\n--- {c['source']}  |  Page {c['page']}  |  Chunk {c['chunk_index']} ---")
        print(c["text"][:400])