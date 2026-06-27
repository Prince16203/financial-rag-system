"""
retrieval/sparse_retriever.py

BM25-based sparse retrieval over financial document chunks.
"""

import logging
from typing import Optional

import numpy as np
from rank_bm25 import BM25Okapi
from langchain_core.documents import Document

import chromadb
import config
from ingestion.indexer import DEFAULT_COLLECTION, DEFAULT_CHROMA_PATH

logger = logging.getLogger(__name__)


class SparseRetriever:
    def __init__(
        self,
        collection_name: str = DEFAULT_COLLECTION,
        chroma_path: str = DEFAULT_CHROMA_PATH,
        default_k: int = config.DEFAULT_TOP_K,
    ):
        self.collection_name = collection_name
        self.chroma_path = chroma_path
        self.default_k = default_k
        self.bm25 = None
        self._documents = []

        self.load_from_chroma()

    def load_from_chroma(self) -> None:
        logger.info("Loading chunks from ChromaDB for BM25 indexing...")
        client = chromadb.PersistentClient(path=self.chroma_path)

        try:
            collection = client.get_collection(name=self.collection_name)
        except Exception:
            raise ValueError(f"Collection '{self.collection_name}' not found. Run indexer.py first.")

        count = collection.count()
        if count == 0:
            raise ValueError("ChromaDB collection is empty. Run indexer.py first.")

        logger.info(f"Loading {count} chunks into BM25 index...")
        results = collection.get(include=["documents", "metadatas"])

        self._documents = [
            Document(page_content=text, metadata=meta)
            for text, meta in zip(results["documents"], results["metadatas"])
        ]
        tokenized_corpus = [doc.page_content.lower().split() for doc in self._documents]
        self.bm25 = BM25Okapi(tokenized_corpus)
        logger.info(f"BM25 index built: {len(self._documents)} documents indexed")

    def retrieve(self, query: str, k: Optional[int] = None) -> list[dict]:
        if self.bm25 is None:
            raise RuntimeError("BM25 not loaded. Call load_from_chroma() first.")

        k = k or self.default_k
        tokenized_query = query.lower().split()
        scores = self.bm25.get_scores(tokenized_query)
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

        logger.info(f"BM25 query: '{query[:60]}' | returned {len(results)} results")
        return results

    def get_documents(self) -> list[Document]:
        return self._documents

    def get_bm25(self) -> BM25Okapi:
        return self.bm25


if __name__ == "__main__":
    import sys
    from utils.logging_config import setup_logging
    setup_logging()

    query = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "What is the total revenue?"
    retriever = SparseRetriever()
    results = retriever.retrieve(query, k=5)

    print(f"\n{'='*60}\nBM25 Query: {query}\nResults: {len(results)}\n{'='*60}")
    for i, r in enumerate(results, 1):
        print(f"\n[{i}] Score: {r['score']} | {r['source']} | Page {r['page']}")
        print(r["text"][:300])