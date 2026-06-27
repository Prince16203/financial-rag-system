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

Re-scores hybrid retrieval results using Cohere's rerank API.
Degrades to pass-through (no crash) if the API key is missing or the call fails.
"""

import logging
from typing import Optional

import config

logger = logging.getLogger(__name__)


class Reranker:
    def __init__(self, model: str = None, api_key: Optional[str] = None):
        self.model = model or config.COHERE_RERANK_MODEL
        self.api_key = api_key or config.COHERE_API_KEY
        self.client = None

        if not self.api_key:
            logger.warning(
                "COHERE_API_KEY not set — Reranker will pass candidates through "
                "unchanged. Add COHERE_API_KEY to .env to enable reranking."
            )
            return

        import cohere
        self.client = cohere.ClientV2(api_key=self.api_key)
        logger.info(f"Reranker ready | model={self.model}")

    def rerank(self, query: str, results: list[dict], top_n: Optional[int] = None) -> list[dict]:
        if not results:
            logger.warning("Reranker called with empty results list")
            return []

        top_n = top_n or config.RERANK_TOP_N

        if self.client is None:
            return results[:top_n]

        documents = [r["text"] for r in results]
        logger.info(f"Reranking {len(documents)} chunks for query: '{query[:60]}'")

        try:
            response = self.client.rerank(
                model=self.model,
                query=query,
                documents=documents,
                top_n=top_n,
            )
        except Exception as e:
            # Covers Cohere timeouts, connection errors, 429s, 5xx — anything.
            # Reranking is a quality boost, not core functionality: fail soft.
            logger.error(f"Cohere rerank API call failed: {e}. Falling back to unreranked top-{top_n}.")
            return results[:top_n]

        reranked = []
        for hit in response.results:
            original = results[hit.index]
            reranked_result = dict(original)
            reranked_result["score"] = round(hit.relevance_score, 6)
            reranked_result["rerank_index"] = hit.index
            reranked_result["retrieval_type"] = "hybrid+reranked"
            reranked.append(reranked_result)

        logger.info(f"Reranking complete | input={len(results)} -> output={len(reranked)} chunks")
        return reranked

    def retrieve_and_rerank(self, query: str, retriever, fetch_k: int = None, top_n: int = None) -> list[dict]:
        fetch_k = fetch_k or config.HYBRID_FETCH_K
        top_n = top_n or config.RERANK_TOP_N
        raw_results = retriever.retrieve(query, k=fetch_k)
        return self.rerank(query, raw_results, top_n=top_n)


if __name__ == "__main__":
    import sys
    from utils.logging_config import setup_logging
    setup_logging()

    from retrieval.hybrid_retriever import HybridRetriever

    query = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "What is FICC net revenue?"
    print(f"\nQuery: {query}\nStep 1 — Running hybrid retrieval...")

    hybrid = HybridRetriever()
    raw_results = hybrid.retrieve(query, k=20)

    print(f"Step 2 — Reranking {len(raw_results)} results with Cohere...")
    reranker = Reranker()
    reranked = reranker.rerank(query, raw_results, top_n=5)

    print(f"\n{'='*60}\nReranked Results: {len(reranked)}\n{'='*60}")
    for i, r in enumerate(reranked, 1):
        print(f"\n[{i}] Score: {r['score']} | {r['source']} | Page {r['page']}")
        print(r["text"][:300])