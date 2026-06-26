"""
retrieval/dense_retriever.py

Given a natural language query, embeds it and retrieves the top-k
most semantically similar chunks from ChromaDB.

Each result dict:
{
    "text":        "...",           # chunk text (sent to LLM as context)
    "source":      "msft_10k.pdf",
    "page":        12,
    "chunk_index": 2,
    "score":       0.87,            # cosine similarity (higher = more relevant)
}

Usage:
    from retrieval.dense_retriever import DenseRetriever

    retriever = DenseRetriever()
    results   = retriever.retrieve("What is Microsoft's revenue?", k=5)
    for r in results:
        print(r["score"], r["source"], r["page"])
        print(r["text"][:200])
"""

import logging
from typing import Optional

import chromadb

from ingestion.embedder import Embedder
from ingestion.indexer  import DEFAULT_COLLECTION, DEFAULT_CHROMA_PATH

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# DenseRetriever
# ---------------------------------------------------------------------------

class DenseRetriever:
    """
    Semantic retriever using ChromaDB + BAAI/bge embeddings.

    Args:
        collection_name: ChromaDB collection to query (must already be indexed).
        chroma_path:     Path to the ChromaDB persistence directory.
        embedder:        Optional pre-loaded Embedder instance (avoids reloading).
        default_k:       Default number of chunks to return per query.
        score_threshold: Minimum cosine similarity to include a result (0-1).
                         Set to 0.0 to return all top-k regardless of score.
    """

    def __init__(
        self,
        collection_name: str = DEFAULT_COLLECTION,
        chroma_path: str = DEFAULT_CHROMA_PATH,
        embedder: Optional[Embedder] = None,
        # default_k: int = 5,
        default_k: int = 10,
        score_threshold: float = 0.0,
    ):
        self.default_k       = default_k
        self.score_threshold = score_threshold

        # Reuse embedder if passed in (saves ~3s model load time)
        self.embedder = embedder if embedder else Embedder()

        logger.info(f"Connecting to ChromaDB collection: '{collection_name}'")
        client = chromadb.PersistentClient(path=chroma_path)

        try:
            self.collection = client.get_collection(name=collection_name)
        except Exception:
            raise ValueError(
                f"Collection '{collection_name}' not found in {chroma_path}. "
                "Run indexer.py first to populate ChromaDB."
            )

        logger.info(
            f"DenseRetriever ready | "
            f"collection='{collection_name}' | "
            f"docs={self.collection.count()}"
        )

    # ------------------------------------------------------------------
    # Core: retrieve top-k chunks for a query
    # ------------------------------------------------------------------

    def retrieve(
        self,
        query: str,
        k: Optional[int] = None,
        source_filter: Optional[str] = None,
    ) -> list[dict]:
        """
        Retrieve the top-k chunks most relevant to the query.

        Args:
            query:         Natural language question.
            k:             Number of results to return. Defaults to self.default_k.
            source_filter: If set, only return chunks from this PDF filename.

        Returns:
            List of result dicts sorted by score descending:
            [{"text", "source", "page", "chunk_index", "score"}, ...]
        """
        k = k or self.default_k

        # Embed the query with BGE query prefix
        query_vec = self.embedder.embed_query(query)

        # Build optional where filter
        where = {"source": source_filter} if source_filter else None

        # Query ChromaDB
        raw = self.collection.query(
            query_embeddings=[query_vec],
            n_results=min(k, self.collection.count()),
            where=where,
            include=["documents", "metadatas", "distances"],
        )

        results = _parse_chroma_results(raw, self.score_threshold)

        logger.info(
            f"Query: '{query[:60]}...' | "
            f"returned {len(results)}/{k} results"
        )
        return results

    def retrieve_batch(self, queries: list[str], k: Optional[int] = None) -> list[list[dict]]:
        """
        Retrieve top-k chunks for multiple queries at once.

        Args:
            queries: List of question strings.
            k:       Results per query.

        Returns:
            List of result lists, one per query.
        """
        k = k or self.default_k
        query_vecs = [self.embedder.embed_query(q) for q in queries]

        raw = self.collection.query(
            query_embeddings=query_vecs,
            n_results=min(k, self.collection.count()),
            include=["documents", "metadatas", "distances"],
        )

        all_results = []
        for i in range(len(queries)):
            single_raw = {
                "documents": [raw["documents"][i]],
                "metadatas": [raw["metadatas"][i]],
                "distances": [raw["distances"][i]],
            }
            all_results.append(_parse_chroma_results(single_raw, self.score_threshold))

        return all_results

    # ------------------------------------------------------------------
    # Convenience: format results as LLM context string
    # ------------------------------------------------------------------

    def format_context(self, results: list[dict], max_chars: int = 4000) -> str:
        """
        Convert retrieved chunks into a formatted context string for the LLM.

        Format per chunk:
            [Source: msft_10k.pdf | Page 12]
            <chunk text>

        Args:
            results:   Output of retrieve().
            max_chars: Truncate total context at this many characters.

        Returns:
            Formatted string ready to inject into a prompt.
        """
        parts = []
        total = 0

        for r in results:
            header = f"[Source: {r['source']} | Page {r['page']}]"
            block  = f"{header}\n{r['text']}"

            if total + len(block) > max_chars:
                # Include as much of this chunk as fits
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

def _parse_chroma_results(raw: dict, score_threshold: float) -> list[dict]:
    """
    Convert ChromaDB query output into clean result dicts.

    ChromaDB returns cosine DISTANCE (0 = identical, 2 = opposite).
    We convert to similarity: score = 1 - distance
    """
    results = []

    docs      = raw["documents"][0]
    metas     = raw["metadatas"][0]
    distances = raw["distances"][0]

    for doc, meta, dist in zip(docs, metas, distances):
        score = round(1 - dist, 4)   # convert distance -> similarity

        if score < score_threshold:
            continue

        results.append({
            "text":        doc,
            "source":      meta["source"],
            "page":        meta["page"],
            "chunk_index": meta["chunk_index"],
            "score":       score,
        })

    # Already sorted by ChromaDB (nearest first), but re-sort to be safe
    results.sort(key=lambda x: x["score"], reverse=True)
    return results


# ---------------------------------------------------------------------------
# Smoke-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    query = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "What is the total revenue?"

    retriever = DenseRetriever()
    results   = retriever.retrieve(query, k=5)

    print(f"\n{'='*60}")
    print(f"Query: {query}")
    print(f"Results: {len(results)}")
    print(f"{'='*60}")

    for i, r in enumerate(results, 1):
        print(f"\n[{i}] Score: {r['score']} | {r['source']} | Page {r['page']}")
        print(r["text"][:300])

    print(f"\n{'='*60}")
    print("Formatted context (first 800 chars):")
    print(retriever.format_context(results)[:800])