"""
ingestion/embedder.py

Loads the BAAI/bge-base-en-v1.5 embedding model (via sentence-transformers)
and converts text chunks into 768-dimensional float vectors.

Why BAAI/bge-base-en-v1.5?
  - Top performer on MTEB financial/retrieval benchmarks at "base" size
  - 768-dim vectors; good balance of quality vs. storage
  - Runs entirely locally — no API key, no cost per embed
  - Has a specific instruction prefix for queries (improves retrieval accuracy)

BGE query prefix convention:
  When embedding a SEARCH QUERY, prepend:
      "Represent this sentence for searching relevant passages: "
  When embedding a DOCUMENT CHUNK, no prefix needed.
  The embedder handles this automatically via embed_query() vs embed_chunks().

Model name and batch size are pulled from config.py.

Output shape:
  embed_chunks(chunks) → same list of chunk dicts, each gaining an "embedding" key:
  {
      "source":      "apple_10k_2023.pdf",
      "page":        3,
      "chunk_index": 0,
      "text":        "...",
      "char_count":  412,
      "embedding":   [0.021, -0.134, ...]   # list of 768 floats
  }

Usage:
    from ingestion.embedder import Embedder

    embedder = Embedder()                          # loads model once
    chunks   = embedder.embed_chunks(chunks)       # adds "embedding" to each chunk
    vec      = embedder.embed_query("What is VaR?")  # single query vector
"""

import logging
import os
from typing import Optional

from sentence_transformers import SentenceTransformer

import config

# logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_MODEL   = config.EMBEDDING_MODEL
QUERY_PREFIX    = "Represent this sentence for searching relevant passages: "
BATCH_SIZE      = config.EMBEDDING_BATCH_SIZE   # reduce to 32 if you hit OOM on CPU


# ---------------------------------------------------------------------------
# Embedder class
# ---------------------------------------------------------------------------

class Embedder:
    """
    Wraps sentence-transformers to embed document chunks and queries.

    Args:
        model_name:  HuggingFace model ID. Defaults to BAAI/bge-base-en-v1.5.
        device:      "cpu", "cuda", or "mps". Auto-detected if None.
        batch_size:  Chunks per forward pass during embed_chunks().
        cache_dir:   Optional path to cache the downloaded model weights.
                     Defaults to HuggingFace's standard ~/.cache/huggingface.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        device: Optional[str] = None,
        batch_size: int = BATCH_SIZE,
        cache_dir: Optional[str] = None,
    ):
        self.model_name = model_name
        self.batch_size = batch_size
        self._device    = device  # None → auto

        logger.info(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(
            model_name,
            device=device,
            cache_folder=cache_dir,
        )

        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        logger.info(
            f"Model loaded | dim={self.embedding_dim} | "
            f"device={self.model.device}"
        )

    # ------------------------------------------------------------------
    # Document chunks — no prefix
    # ------------------------------------------------------------------

    def embed_chunks(self, chunks: list[dict]) -> list[dict]:
        """
        Add an "embedding" key to each chunk dict in-place.

        Args:
            chunks: Output of chunker.chunk_pages().

        Returns:
            Same list with "embedding" (list of floats) added to each dict.
        """
        if not chunks:
            logger.warning("embed_chunks called with empty list")
            return chunks

        texts = [c["text"] for c in chunks]
        logger.info(f"Embedding {len(texts)} chunks (batch_size={self.batch_size})...")

        # normalize_embeddings=True → cosine similarity works as dot product
        # show_progress_bar=True    → gives ETA for large corpora
        vectors = self.model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=True,
            show_progress_bar=True,
            convert_to_numpy=True,
        )

        for chunk, vec in zip(chunks, vectors):
            chunk["embedding"] = vec.tolist()

        logger.info(f"Embedding complete: {len(chunks)} chunks × {self.embedding_dim} dims")
        return chunks

    # ------------------------------------------------------------------
    # Query — with BGE query prefix
    # ------------------------------------------------------------------

    def embed_query(self, query: str) -> list[float]:
        """
        Embed a single search query using the BGE query prefix.

        Args:
            query: Natural language question.

        Returns:
            List of floats (length = embedding_dim).
        """
        prefixed = QUERY_PREFIX + query
        vec = self.model.encode(
            prefixed,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        return vec.tolist()

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def embed_texts(self, texts: list[str], use_query_prefix: bool = False) -> list[list[float]]:
        """
        Embed a plain list of strings.

        Args:
            texts:            List of strings.
            use_query_prefix: If True, prepend BGE query prefix to each string.

        Returns:
            List of embedding vectors (list of list of floats).
        """
        if use_query_prefix:
            texts = [QUERY_PREFIX + t for t in texts]

        vectors = self.model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=True,
            show_progress_bar=len(texts) > 10,
            convert_to_numpy=True,
        )
        return [v.tolist() for v in vectors]

    @property
    def dim(self) -> int:
        """Embedding dimensionality."""
        return self.embedding_dim


# ---------------------------------------------------------------------------
# Smoke-test
# ---------------------------------------------------------------------------

from utils.logging_config import setup_logging
setup_logging()
if __name__ == "__main__":
    import sys
    from ingestion.pdf_parser import parse_pdf, parse_directory
    from ingestion.chunker import chunk_pages
    import os

    path = sys.argv[1] if len(sys.argv) > 1 else str(config.RAW_PDF_DIR)

    if os.path.isdir(path):
        pages = parse_directory(path)
    else:
        pages = parse_pdf(path)

    chunks = chunk_pages(pages)

    # Only embed first 20 chunks during smoke-test to keep it fast
    sample = chunks[:20]
    print(f"\nSmoke-test: embedding first {len(sample)} chunks...")

    embedder = Embedder()
    sample   = embedder.embed_chunks(sample)

    print(f"\n{'='*60}")
    print(f"Embedding dim : {embedder.dim}")
    print(f"Sample chunks : {len(sample)}")
    print(f"First vector  : {sample[0]['embedding'][:6]} ...")

    # Test query embedding
    q_vec = embedder.embed_query("What is the Value at Risk for this fund?")
    print(f"Query vector  : {q_vec[:6]} ...")
    print(f"{'='*60}")