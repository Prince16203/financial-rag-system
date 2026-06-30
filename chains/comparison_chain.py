"""
chains/comparison_chain.py

Multi-document comparison chain: given a question comparing two or more
companies/documents, retrieves relevant chunks for each entity separately
and asks Groq to compare them side-by-side.

Usage:
    from chains.comparison_chain import ComparisonChain

    chain = ComparisonChain()
    result = chain.compare(
        question="Compare Apple and Goldman Sachs revenue growth",
        entities=["Apple", "Goldman Sachs"],
    )
    print(result["answer"])
"""

import logging
from typing import Optional

import groq
from groq import Groq

import config
from retrieval.hybrid_retriever import HybridRetriever
from retrieval.reranker import Reranker

logger = logging.getLogger(__name__)

COMPARISON_PROMPT = """You are a financial analyst comparing multiple companies.

Below are relevant excerpts from financial documents for each entity.
Answer the question by comparing the entities directly.
Be specific — cite numbers, years, and sources where available.
If the documents don't contain enough information for a full comparison, say so.

{context_blocks}

Question: {question}

Provide a structured comparison:"""


class ComparisonChain:
    """
    Retrieves context for each entity separately, then asks the LLM
    to compare them in a single structured response.

    Args:
        hybrid_retriever: Optional pre-built HybridRetriever.
        reranker:         Optional pre-built Reranker.
        chunks_per_entity: How many reranked chunks to include per entity.
        groq_client:      Optional pre-built Groq client.
    """

    def __init__(
        self,
        hybrid_retriever: Optional[HybridRetriever] = None,
        reranker: Optional[Reranker] = None,
        chunks_per_entity: int = 3,
        groq_client: Optional[Groq] = None,
    ):
        self.retriever = hybrid_retriever or HybridRetriever()
        self.reranker  = reranker or Reranker()
        self.chunks_per_entity = chunks_per_entity

        if not config.GROQ_API_KEY:
            raise ValueError("GROQ_API_KEY not set in .env")
        self.client = groq_client or Groq(api_key=config.GROQ_API_KEY)

        logger.info(f"ComparisonChain ready | chunks_per_entity={chunks_per_entity}")

    def compare(self, question: str, entities: Optional[list[str]] = None) -> dict:
        """
        Compare entities by retrieving separate context for each.

        Args:
            question: Comparison question (e.g. "Compare revenue of Apple vs Goldman")
            entities: Optional list of entity names to filter by source PDF.
                      If None, retrieves from all indexed documents.

        Returns:
            {"answer": str, "sources": list[dict], "entity_chunks": dict}
        """
        if entities:
            context_blocks, all_sources, entity_chunks = \
                self._retrieve_by_entity(question, entities)
        else:
            context_blocks, all_sources, entity_chunks = \
                self._retrieve_general(question)

        if not context_blocks:
            return {
                "answer":        "No relevant documents found to make this comparison.",
                "sources":       [],
                "entity_chunks": {},
            }

        prompt = COMPARISON_PROMPT.format(
            context_blocks="\n\n".join(context_blocks),
            question=question,
        )

        answer = self._call_groq(prompt)
        logger.info(f"ComparisonChain answered: '{question[:60]}...'")

        return {
            "answer":        answer,
            "sources":       all_sources,
            "entity_chunks": entity_chunks,
        }

    def _retrieve_by_entity(
        self, question: str, entities: list[str]
    ) -> tuple[list[str], list[dict], dict]:
        """Retrieve and rerank chunks for each named entity separately."""
        context_blocks = []
        all_sources = []
        entity_chunks = {}

        for entity in entities:
            # Try source_filter first (matches PDF filename containing entity name)
            source_filter = None
            for src in self._get_indexed_sources():
                if entity.lower() in src.lower():
                    source_filter = src
                    break

            raw = self.retriever.retrieve(
                f"{entity} {question}",
                k=config.HYBRID_FETCH_K,
                source_filter=source_filter,
            )
            chunks = self.reranker.rerank(
                question, raw, top_n=self.chunks_per_entity
            )

            if chunks:
                block_lines = [f"### {entity}"]
                for c in chunks:
                    block_lines.append(
                        f"[{c['source']} | Page {c['page']}]\n{c['text']}"
                    )
                context_blocks.append("\n\n".join(block_lines))
                all_sources.extend(
                    {"source": c["source"], "page": c["page"]} for c in chunks
                )
                entity_chunks[entity] = chunks

        return context_blocks, all_sources, entity_chunks

    def _retrieve_general(
        self, question: str
    ) -> tuple[list[str], list[dict], dict]:
        """Retrieve without entity filter — for open-ended comparisons."""
        raw = self.retriever.retrieve(question, k=config.HYBRID_FETCH_K)
        chunks = self.reranker.rerank(question, raw, top_n=self.chunks_per_entity * 2)

        if not chunks:
            return [], [], {}

        block = "\n\n".join(
            f"[{c['source']} | Page {c['page']}]\n{c['text']}" for c in chunks
        )
        sources = [{"source": c["source"], "page": c["page"]} for c in chunks]
        return [block], sources, {"general": chunks}

    def _get_indexed_sources(self) -> list[str]:
        """Return list of source PDF filenames currently in ChromaDB."""
        try:
            import chromadb
            client = chromadb.PersistentClient(path=config.CHROMA_DB_PATH)
            collection = client.get_collection(config.COLLECTION_NAME)
            results = collection.get(include=["metadatas"])
            return list({m["source"] for m in results["metadatas"]})
        except Exception:
            return []

    def _call_groq(self, prompt: str) -> str:
        try:
            response = self.client.chat.completions.create(
                model=config.GROQ_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=config.LLM_TEMPERATURE,
            )
            return response.choices[0].message.content

        except groq.RateLimitError as e:
            logger.error(f"Groq rate limit: {e}")
            return "Rate limited — please try again in a moment."
        except groq.APIConnectionError as e:
            logger.error(f"Groq connection error: {e}")
            return "Cannot reach the language model. Check your connection."
        except groq.APIError as e:
            logger.error(f"Groq API error: {e}")
            return "The language model returned an error. Please try again."


if __name__ == "__main__":
    import sys
    from utils.logging_config import setup_logging
    setup_logging()

    question = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else \
        "Compare the revenue and risk exposure of Apple and Goldman Sachs"

    chain = ComparisonChain()
    result = chain.compare(question, entities=["Apple", "Goldman Sachs"])

    print(f"\n{'='*60}\nQuestion: {question}\n{'='*60}")
    print(f"\nAnswer:\n{result['answer']}")
    print(f"\nSources: {result['sources']}")