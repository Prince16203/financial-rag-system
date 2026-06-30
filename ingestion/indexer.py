"""
ingestion/indexer.py

Takes embedded chunks (from embedder.py) and writes them into ChromaDB
with full metadata for later retrieval.

ChromaDB collection schema per document:
    id:        "source__page__chunkidx"   (unique string ID)
    embedding: [0.021, -0.134, ...]       (768-dim float list)
    document:  chunk["text"]              (raw text for LLM context)
    metadata:  {source, page, chunk_index, char_count}

Collection name / persist path are pulled from config.py.

Usage:
    from ingestion.indexer import Indexer

    indexer = Indexer()                   # connects to ChromaDB
    indexer.index_chunks(chunks)          # upserts all chunks
    indexer.get_collection_stats()        # prints count + sources
"""

import logging
import os
from pathlib import Path
from typing import Optional

import chromadb
from chromadb.config import Settings

import config

# logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_COLLECTION  = config.COLLECTION_NAME
DEFAULT_CHROMA_PATH = config.CHROMA_DB_PATH
UPSERT_BATCH_SIZE   = 100   # how many chunks to upsert per ChromaDB call


# ---------------------------------------------------------------------------
# Indexer class
# ---------------------------------------------------------------------------

class Indexer:
    """
    Manages a persistent ChromaDB collection for financial document chunks.

    Args:
        collection_name: ChromaDB collection to write into.
        chroma_path:     Directory where ChromaDB persists its data.
    """

    def __init__(
        self,
        collection_name: str = DEFAULT_COLLECTION,
        chroma_path: str = DEFAULT_CHROMA_PATH,
    ):
        self.collection_name = collection_name
        self.chroma_path     = chroma_path

        Path(chroma_path).mkdir(parents=True, exist_ok=True)

        logger.info(f"Connecting to ChromaDB at: {chroma_path}")
        self.client = chromadb.PersistentClient(path=chroma_path)

        # get_or_create so re-running indexer is always safe (upsert below)
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},   # cosine distance for BGE
        )
        logger.info(
            f"Collection '{collection_name}' ready "
            f"(existing docs: {self.collection.count()})"
        )

    # ------------------------------------------------------------------
    # Core: write chunks into ChromaDB
    # ------------------------------------------------------------------

    def index_chunks(self, chunks: list[dict]) -> int:
        """
        Upsert a list of embedded chunks into ChromaDB.

        Upsert (not insert) means re-running on the same PDF is safe —
        existing chunks are overwritten, not duplicated.

        Args:
            chunks: Output of embedder.embed_chunks() — must have "embedding" key.

        Returns:
            Number of chunks successfully indexed.
        """
        if not chunks:
            logger.warning("index_chunks called with empty list")
            return 0

        # Verify embeddings exist
        missing = [i for i, c in enumerate(chunks) if "embedding" not in c]
        if missing:
            raise ValueError(
                f"{len(missing)} chunks are missing 'embedding' key. "
                "Run embedder.embed_chunks() before indexing."
            )

        total = len(chunks)
        indexed = 0

        # Process in batches to avoid memory spikes on large corpora
        for batch_start in range(0, total, UPSERT_BATCH_SIZE):
            batch = chunks[batch_start : batch_start + UPSERT_BATCH_SIZE]

            ids        = []
            embeddings = []
            documents  = []
            metadatas  = []

            for chunk in batch:
                # Unique, human-readable ID
                chunk_id = _make_id(chunk)

                ids.append(chunk_id)
                embeddings.append(chunk["embedding"])
                documents.append(chunk["text"])
                metadatas.append({
                    "source":      chunk["source"],
                    "page":        chunk["page"],
                    "chunk_index": chunk["chunk_index"],
                    "char_count":  chunk["char_count"],
                })

            self.collection.upsert(
                ids=ids,
                embeddings=embeddings,
                documents=documents,
                metadatas=metadatas,
            )

            indexed += len(batch)
            logger.info(f"  Indexed {indexed}/{total} chunks...")

        logger.info(
            f"Indexing complete: {indexed} chunks in collection "
            f"'{self.collection_name}' (total: {self.collection.count()})"
        )
        return indexed

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def get_collection_stats(self) -> dict:
        """Return count and unique sources currently in the collection."""
        count = self.collection.count()
        if count == 0:
            return {"total_chunks": 0, "sources": []}

        # Fetch all metadatas (no embeddings needed)
        results = self.collection.get(include=["metadatas"])
        sources = sorted({m["source"] for m in results["metadatas"]})

        stats = {
            "total_chunks":   count,
            "unique_sources": len(sources),
            "sources":        sources,
        }
        logger.info(f"Collection stats: {stats}")
        return stats

    def delete_source(self, source_name: str) -> None:
        """
        Remove all chunks belonging to a specific PDF from the collection.
        Useful when re-ingesting an updated filing.
        """
        self.collection.delete(where={"source": source_name})
        logger.info(f"Deleted all chunks for source: {source_name}")

    def reset_collection(self) -> None:
        """Drop and recreate the collection. Wipes all data."""
        self.client.delete_collection(self.collection_name)
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        logger.warning(f"Collection '{self.collection_name}' reset (all data deleted)")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_id(chunk: dict) -> str:
    """
    Build a unique, deterministic ID for a chunk.
    Format: "filename__page3__chunk0"
    Double underscores avoid collisions with filenames containing underscores.
    """
    source = chunk["source"].replace(" ", "_")
    return f"{source}__page{chunk['page']}__chunk{chunk['chunk_index']}"


# ---------------------------------------------------------------------------
# Full pipeline convenience function
# ---------------------------------------------------------------------------

def run_ingestion_pipeline(
    pdf_dir: str = str(config.RAW_PDF_DIR),
    chroma_path: str = DEFAULT_CHROMA_PATH,
    collection_name: str = DEFAULT_COLLECTION,
    chunk_size: int = config.CHUNK_SIZE,
    chunk_overlap: int = config.CHUNK_OVERLAP,
) -> dict:
    """
    End-to-end pipeline: PDF dir -> ChromaDB.
    Combines parser + chunker + embedder + indexer in one call.

    Returns collection stats dict.
    """
    from ingestion.pdf_parser import parse_directory
    from ingestion.chunker   import chunk_pages
    from ingestion.embedder  import Embedder

    logger.info("=" * 60)
    logger.info("Starting full ingestion pipeline")
    logger.info(f"  PDF dir:    {pdf_dir}")
    logger.info(f"  ChromaDB:   {chroma_path}")
    logger.info(f"  Collection: {collection_name}")
    logger.info("=" * 60)

    pages  = parse_directory(pdf_dir)
    chunks = chunk_pages(pages, chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    embedder = Embedder()
    chunks   = embedder.embed_chunks(chunks)

    indexer = Indexer(collection_name=collection_name, chroma_path=chroma_path)
    indexer.index_chunks(chunks)

    return indexer.get_collection_stats()


# ---------------------------------------------------------------------------
# Smoke-test
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    import sys

    from utils.logging_config import setup_logging
    setup_logging()
    
    pdf_dir = sys.argv[1] if len(sys.argv) > 1 else str(config.RAW_PDF_DIR)
    stats   = run_ingestion_pipeline(pdf_dir=pdf_dir)

    print(f"\n{'='*60}")
    print("Final collection stats:")
    for k, v in stats.items():
        print(f"  {k}: {v}")
    print(f"{'='*60}")