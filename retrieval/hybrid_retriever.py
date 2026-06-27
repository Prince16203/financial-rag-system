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
Combines dense (ChromaDB semantic) and sparse (BM25 keyword) retrieval
using Reciprocal Rank Fusion (RRF).
"""

import logging
from typing import Optional
from collections import defaultdict

import config
from retrieval.dense_retriever import DenseRetriever
from retrieval.sparse_retriever import SparseRetriever

logger = logging.getLogger(__name__)


class HybridRetriever:
    def __init__(
        self,
        dense_weight: float = config.DENSE_WEIGHT,
        sparse_weight: float = config.SPARSE_WEIGHT,
        default_k: int = config.DEFAULT_TOP_K,
        fetch_k: int = config.HYBRID_FETCH_K,
        collection_name: str = None,
        chroma_path: str = None,
    ):
        self.dense_weight = dense_weight
        self.sparse_weight = sparse_weight
        self.default_k = default_k
        self.fetch_k = fetch_k

        logger.info("Initialising HybridRetriever...")

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

        logger.info(f"HybridRetriever ready | dense_weight={dense_weight} | sparse_weight={sparse_weight}")

    def retrieve(self, query: str, k: Optional[int] = None, source_filter: Optional[str] = None) -> list[dict]:
        k = k or self.default_k

        logger.info(f"Fetching {self.fetch_k} results from dense retriever...")
        dense_results = self.dense.retrieve(query, k=self.fetch_k, source_filter=source_filter)

        logger.info(f"Fetching {self.fetch_k} results from sparse retriever...")
        sparse_results = self.sparse.retrieve(query, k=self.fetch_k)

        if source_filter:
            sparse_results = [r for r in sparse_results if r["source"] == source_filter]

        fused = self._reciprocal_rank_fusion(dense_results, sparse_results)
        results = fused[:k]

        logger.info(
            f"HybridRetriever query: '{query[:60]}' | dense={len(dense_results)} "
            f"sparse={len(sparse_results)} fused={len(results)}"
        )
        return results

    def _reciprocal_rank_fusion(self, dense_results: list[dict], sparse_results: list[dict]) -> list[dict]:
        scores = defaultdict(float)
        chunk_data = {}

        for rank, result in enumerate(dense_results, start=1):
            key = _chunk_key(result)
            scores[key] += self.dense_weight * (1.0 / (config.RRF_K + rank))
            chunk_data[key] = result

        for rank, result in enumerate(sparse_results, start=1):
            key = _chunk_key(result)
            scores[key] += self.sparse_weight * (1.0 / (config.RRF_K + rank))
            if key not in chunk_data:
                chunk_data[key] = result

        sorted_keys = sorted(scores.keys(), key=lambda k: scores[k], reverse=True)

        fused = []
        for key in sorted_keys:
            result = dict(chunk_data[key])
            result["score"] = round(scores[key], 6)
            result["retrieval_type"] = "hybrid"
            fused.append(result)

        return fused

    def format_context(self, results: list[dict], max_chars: int = 4000) -> str:
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


def _chunk_key(result: dict) -> tuple:
    return (result["source"], result["page"], result["chunk_index"])


if __name__ == "__main__":
    import sys
    from utils.logging_config import setup_logging
    setup_logging()

    query = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "What is the total revenue?"
    retriever = HybridRetriever()
    results = retriever.retrieve(query, k=5)

    print(f"\n{'='*60}\nHybrid Query: {query}\nResults: {len(results)}\n{'='*60}")
    for i, r in enumerate(results, 1):
        print(f"\n[{i}] Score: {r['score']} | {r['source']} | Page {r['page']}")
        print(r["text"][:300])
    print(f"\n{'='*60}\nFormatted context (first 800 chars):")
    print(retriever.format_context(results)[:800])