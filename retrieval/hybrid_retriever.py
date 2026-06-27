"""
retrieval/hybrid_retriever.py

Combines dense (ChromaDB semantic) and sparse (BM25 keyword) retrieval
into a single hybrid retriever using Reciprocal Rank Fusion (RRF).

Why hybrid?
- Dense catches meaning: "what were earnings?" finds revenue chunks
- Sparse catches exact terms: "FICC net revenues" finds exact matches
- RRF combines both ranked lists without needing score normalisation

Flow:
    query
      ├── DenseRetriever  → top-k ranked list
      ├── SparseRetriever → top-k ranked list
      └── RRF fusion      → single merged ranked list

Usage:
    from retrieval.hybrid_retriever import HybridRetriever

    retriever = HybridRetriever()
    results = retriever.retrieve("What is FICC net revenue?", k=5)
"""

import logging
from typing import Optional
from collections import defaultdict

from retrieval.dense_retriever import DenseRetriever
from retrieval.sparse_retriever import SparseRetriever

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# RRF constant — 60 is standard, higher = less penalty for lower ranks
RRF_K = 60


class HybridRetriever:
    """
    Hybrid retriever combining dense and sparse retrieval via RRF.

    Args:
        dense_weight:    Weight for dense retriever scores (0-1).
        sparse_weight:   Weight for sparse retriever scores (0-1).
        default_k:       Number of results to return after fusion.
        fetch_k:         How many results to fetch from each retriever
                         before fusion. Should be > default_k.
        collection_name: ChromaDB collection name.
        chroma_path:     Path to ChromaDB persistence directory.
    """

    def __init__(
        self,
        dense_weight: float = 0.5,
        sparse_weight: float = 0.5,
        default_k: int = 5,
        fetch_k: int = 20,
        collection_name: str = None,
        chroma_path: str = None,
    ):
        self.dense_weight = dense_weight
        self.sparse_weight = sparse_weight
        self.default_k = default_k
        self.fetch_k = fetch_k

        logger.info("Initialising HybridRetriever...")

        # Initialise both retrievers
        # Pass collection/path only if explicitly provided
        dense_kwargs = {}
        sparse_kwargs = {}
        if collection_name:
            dense_kwargs["collection_name"] = collection_name
            sparse_kwargs["collection_name"] = collection_name
        if chroma_path:
            dense_kwargs["chroma_path"] = chroma_path
            sparse_kwargs["chroma_path"] = chroma_path

        self.dense = DenseRetriever(**dense_kwargs)
        self.sparse = SparseRetriever(**sparse_kwargs)

        logger.info(
            f"HybridRetriever ready | "
            f"dense_weight={dense_weight} | "
            f"sparse_weight={sparse_weight}"
        )

    def retrieve(
        self,
        query: str,
        k: Optional[int] = None,
        source_filter: Optional[str] = None,
    ) -> list[dict]:
        """
        Retrieve top-k chunks using hybrid dense + sparse retrieval.

        Uses Reciprocal Rank Fusion to combine ranked lists from
        both retrievers without needing score normalisation.

        Args:
            query:         Natural language question.
            k:             Number of results to return.
            source_filter: If set, only return chunks from this PDF.

        Returns:
            List of result dicts sorted by fused score descending.
        """
        k = k or self.default_k

        # Step 1 — Fetch from both retrievers independently
        logger.info(f"Fetching {self.fetch_k} results from dense retriever...")
        dense_results = self.dense.retrieve(
            query,
            k=self.fetch_k,
            source_filter=source_filter,
        )

        logger.info(f"Fetching {self.fetch_k} results from sparse retriever...")
        sparse_results = self.sparse.retrieve(
            query,
            k=self.fetch_k,
        )

        # Step 2 — Apply source filter to sparse results manually
        # (sparse retriever does not support source_filter natively)
        if source_filter:
            sparse_results = [
                r for r in sparse_results
                if r["source"] == source_filter
            ]

        # Step 3 — Fuse using RRF
        fused = self._reciprocal_rank_fusion(dense_results, sparse_results)

        # Step 4 — Return top-k
        results = fused[:k]

        logger.info(
            f"HybridRetriever query: '{query[:60]}' | "
            f"dense={len(dense_results)} sparse={len(sparse_results)} "
            f"fused={len(results)}"
        )
        return results

    def _reciprocal_rank_fusion(
        self,
        dense_results: list[dict],
        sparse_results: list[dict],
    ) -> list[dict]:
        """
        Combine two ranked lists using Reciprocal Rank Fusion.

        RRF score for a document = sum of 1 / (RRF_K + rank)
        across all lists it appears in, weighted by retriever weight.

        Documents are matched by (source, page, chunk_index) key.
        """
        # Map from chunk identity to its data and fused score
        scores = defaultdict(float)
        chunk_data = {}

        # Dense results
        for rank, result in enumerate(dense_results, start=1):
            key = _chunk_key(result)
            scores[key] += self.dense_weight * (1.0 / (RRF_K + rank))
            chunk_data[key] = result

        # Sparse results
        for rank, result in enumerate(sparse_results, start=1):
            key = _chunk_key(result)
            scores[key] += self.sparse_weight * (1.0 / (RRF_K + rank))
            if key not in chunk_data:
                chunk_data[key] = result

        # Sort by fused score descending
        sorted_keys = sorted(scores.keys(), key=lambda k: scores[k], reverse=True)

        fused = []
        for key in sorted_keys:
            result = dict(chunk_data[key])
            result["score"] = round(scores[key], 6)
            result["retrieval_type"] = "hybrid"
            fused.append(result)

        return fused

    def format_context(self, results: list[dict], max_chars: int = 4000) -> str:
        """
        Format retrieved chunks as LLM context string.
        Same format as dense_retriever for drop-in compatibility.
        """
        parts = []
        total = 0

        for r in results:
            header = f"[Source: {r['source']} | Page {r['page']}]"
            block = f"{header}\n{r['text']}"

            if total + len(block) > max_chars:
                remaining = max_chars - total
                if remaining > len(header) + 50:
                    parts.append(block[:remaining])
                break

            parts.append(block)
            total += len(block)

        return "\n\n---\n\n".join(parts)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _chunk_key(result: dict) -> tuple:
    """Unique identity for a chunk across both retrievers."""
    return (
        result["source"],
        result["page"],
        result["chunk_index"],
    )


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    query = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "What is the total revenue?"

    retriever = HybridRetriever()
    results = retriever.retrieve(query, k=5)

    print(f"\n{'='*60}")
    print(f"Hybrid Query: {query}")
    print(f"Results: {len(results)}")
    print(f"{'='*60}")

    for i, r in enumerate(results, 1):
        print(f"\n[{i}] Score: {r['score']} | {r['source']} | Page {r['page']}")
        print(r["text"][:300])

    print(f"\n{'='*60}")
    print("Formatted context (first 800 chars):")
    print(retriever.format_context(results)[:800])