"""
retrieval/sparse_retriever.py

BM25-based sparse retrieval over financial document chunks.
Uses term frequency and inverse document frequency (TF-IDF) scoring
to find chunks containing exact financial terms and keywords.

Why sparse retrieval?
- Dense retrieval finds semantic meaning but can miss exact terms
- BM25 catches exact financial terms like "FICC", "VaR", "Tier 1 Capital"
- Hybrid (dense + sparse) outperforms either alone on financial text

Usage:
    from retrieval.sparse_retriever import SparseRetriever

    retriever = SparseRetriever()
    results = retriever.retrieve("What is FICC net revenue?", k=5)
"""

import logging
from typing import Optional

import numpy as np
from rank_bm25 import BM25Okapi
from langchain_core.documents import Document

import chromadb
from ingestion.indexer import DEFAULT_COLLECTION, DEFAULT_CHROMA_PATH

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


class SparseRetriever:
    """
    BM25 keyword retriever over chunks stored in ChromaDB.

    BM25 does not use a vector database — it loads all chunk texts
    into memory and scores them using TF-IDF at query time.

    Args:
        collection_name: ChromaDB collection to load chunks from.
        chroma_path:     Path to ChromaDB persistence directory.
        default_k:       Default number of results to return.
    """

    def __init__(
        self,
        collection_name: str = DEFAULT_COLLECTION,
        chroma_path: str = DEFAULT_CHROMA_PATH,
        default_k: int = 5,
    ):
        self.collection_name = collection_name
        self.chroma_path = chroma_path
        self.default_k = default_k
        self.bm25 = None
        self._documents = []

        self.load_from_chroma()

    def load_from_chroma(self) -> None:
        """
        Load all chunks from ChromaDB into memory for BM25 indexing.

        BM25 needs all documents in memory to calculate IDF scores
        across the entire corpus. This is done once at startup.
        """
        logger.info("Loading chunks from ChromaDB for BM25 indexing...")

        client = chromadb.PersistentClient(path=self.chroma_path)

        try:
            collection = client.get_collection(name=self.collection_name)
        except Exception:
            raise ValueError(
                f"Collection '{self.collection_name}' not found. "
                "Run indexer.py first."
            )

        count = collection.count()
        if count == 0:
            raise ValueError("ChromaDB collection is empty. Run indexer.py first.")

        logger.info(f"Loading {count} chunks into BM25 index...")

        # Fetch all documents and metadata from ChromaDB
        # No embeddings needed for BM25 — only text and metadata
        results = collection.get(include=["documents", "metadatas"])

        # Convert to LangChain Document objects for consistency
        self._documents = []
        for text, meta in zip(results["documents"], results["metadatas"]):
            self._documents.append(
                Document(
                    page_content=text,
                    metadata=meta,
                )
            )

        # Tokenise all documents for BM25
        # Simple whitespace tokenisation works well for financial text
        tokenized_corpus = [
            doc.page_content.lower().split()
            for doc in self._documents
        ]

        # Build BM25 index
        self.bm25 = BM25Okapi(tokenized_corpus)

        logger.info(f"BM25 index built: {len(self._documents)} documents indexed")

    def retrieve(
        self,
        query: str,
        k: Optional[int] = None,
    ) -> list[dict]:
        """
        Retrieve top-k chunks using BM25 keyword scoring.

        Args:
            query: Natural language question or keyword query.
            k:     Number of results. Defaults to self.default_k.

        Returns:
            List of result dicts with keys: text, source, page,
            chunk_index, score. Same format as dense_retriever
            for consistency.
        """
        if self.bm25 is None:
            raise RuntimeError("BM25 not loaded. Call load_from_chroma() first.")

        k = k or self.default_k

        # Tokenise query the same way as corpus
        tokenized_query = query.lower().split()

        # Get BM25 scores for all documents
        scores = self.bm25.get_scores(tokenized_query)

        # Get top-k indices sorted by score descending
        top_indices = np.argsort(scores)[::-1][:k]

        results = []
        for idx in top_indices:
            doc = self._documents[idx]
            results.append({
                "text":        doc.page_content,
                "source":      doc.metadata.get("source", "unknown"),
                "page":        doc.metadata.get("page", 0),
                "chunk_index": doc.metadata.get("chunk_index", 0),
                "score":       round(float(scores[idx]), 4),
            })

        logger.info(
            f"BM25 query: '{query[:60]}' | returned {len(results)} results"
        )
        return results

    def get_documents(self) -> list[Document]:
        """
        Return all loaded documents.
        Used by hybrid_retriever.py to access the corpus directly.
        """
        return self._documents

    def get_bm25(self) -> BM25Okapi:
        """
        Return the raw BM25Okapi object.
        Used by hybrid_retriever.py for score combination.
        """
        return self.bm25


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    query = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "What is the total revenue?"

    retriever = SparseRetriever()
    results = retriever.retrieve(query, k=5)

    print(f"\n{'='*60}")
    print(f"BM25 Query: {query}")
    print(f"Results: {len(results)}")
    print(f"{'='*60}")

    for i, r in enumerate(results, 1):
        print(f"\n[{i}] Score: {r['score']} | {r['source']} | Page {r['page']}")
        print(r["text"][:300])