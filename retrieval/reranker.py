"""
retrieval/reranker.py

Re-scores hybrid retrieval results using Cohere's rerank API.

Why reranking?
- RRF fusion gives a good combined ranking but uses position-based scores
- Cohere reranker uses a cross-encoder model that reads BOTH the query
  and each chunk together — much more accurate relevance scoring
- Reranking is the final quality gate before chunks go to the LLM

Flow:
    query + hybrid results
        → Cohere rerank API
        → same chunks re-ordered by true relevance
        → top-k sent to LLM

Usage:
    from retrieval.reranker import Reranker

    reranker = Reranker()
    results  = reranker.rerank(query, hybrid_results, top_n=5)
"""

import logging
import os
from typing import Optional

import cohere
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Cohere's best reranking model
RERANK_MODEL = "rerank-english-v3.0"


class Reranker:
    """
    Cohere-based reranker for financial document chunks.

    Takes the output of HybridRetriever and re-scores each chunk
    using Cohere's cross-encoder rerank model.

    Args:
        model:   Cohere rerank model to use.
        api_key: Cohere API key. Reads from COHERE_API_KEY env var if None.
    """

    def __init__(
        self,
        model: str = RERANK_MODEL,
        api_key: Optional[str] = None,
    ):
        self.model = model

        api_key = api_key or os.getenv("COHERE_API_KEY")
        if not api_key:
            raise ValueError(
                "COHERE_API_KEY not found. "
                "Add it to your .env file."
            )

        self.client = cohere.ClientV2(api_key=api_key)
        logger.info(f"Reranker ready | model={model}")

    def rerank(
        self,
        query: str,
        results: list[dict],
        top_n: Optional[int] = None,
    ) -> list[dict]:
        """
        Re-score and re-order retrieval results using Cohere rerank.

        Args:
            query:   The original search query.
            results: Output of HybridRetriever.retrieve() or
                     DenseRetriever.retrieve().
            top_n:   Number of results to return after reranking.
                     Defaults to all results.

        Returns:
            Re-ordered list of result dicts with updated
            'score' (Cohere relevance score) and
            'rerank_index' (original position before reranking).
        """
        if not results:
            logger.warning("Reranker called with empty results list")
            return []

        top_n = top_n or len(results)

        # Extract texts for Cohere API
        documents = [r["text"] for r in results]

        logger.info(
            f"Reranking {len(documents)} chunks for query: "
            f"'{query[:60]}'"
        )

        # Call Cohere rerank API
        response = self.client.rerank(
            model=self.model,
            query=query,
            documents=documents,
            top_n=top_n,
        )

        # Build reranked results list
        reranked = []
        for hit in response.results:
            original = results[hit.index]
            reranked_result = dict(original)
            reranked_result["score"] = round(hit.relevance_score, 6)
            reranked_result["rerank_index"] = hit.index
            reranked_result["retrieval_type"] = "hybrid+reranked"
            reranked.append(reranked_result)

        logger.info(
            f"Reranking complete | "
            f"input={len(results)} → output={len(reranked)} chunks"
        )
        return reranked

    def retrieve_and_rerank(
        self,
        query: str,
        retriever,
        fetch_k: int = 20,
        top_n: int = 5,
    ) -> list[dict]:
        """
        Convenience method: retrieve then immediately rerank.

        Args:
            query:     Natural language question.
            retriever: Any retriever with a .retrieve() method
                       (DenseRetriever or HybridRetriever).
            fetch_k:   How many chunks to fetch before reranking.
            top_n:     How many to return after reranking.

        Returns:
            Top-n reranked result dicts.
        """
        # Fetch more than needed so reranker has room to re-order
        raw_results = retriever.retrieve(query, k=fetch_k)

        # Rerank and return top_n
        return self.rerank(query, raw_results, top_n=top_n)


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    from retrieval.hybrid_retriever import HybridRetriever

    query = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "What is FICC net revenue?"

    print(f"\nQuery: {query}")
    print("Step 1 — Running hybrid retrieval...")

    hybrid = HybridRetriever()
    raw_results = hybrid.retrieve(query, k=20)

    print(f"Step 2 — Reranking {len(raw_results)} results with Cohere...")

    reranker = Reranker()
    reranked = reranker.rerank(query, raw_results, top_n=5)

    print(f"\n{'='*60}")
    print(f"Reranked Results: {len(reranked)}")
    print(f"{'='*60}")

    for i, r in enumerate(reranked, 1):
        print(f"\n[{i}] Cohere Score: {r['score']} | {r['source']} | Page {r['page']}")
        print(r["text"][:300])