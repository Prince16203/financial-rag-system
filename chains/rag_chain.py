"""
chains/rag_chain.py

Base RAG chain: retrieves chunks, reranks, calls LLM, returns structured answer.
Used by conversational.py and (optionally) react_agent.py for fallback QA.

This is a BLOCKING FILE for the UI to launch.
Colleague (Phase 1 retrieval owner) must create this.

Usage:
    from chains.rag_chain import RAGChain

    rag = RAGChain()
    result = rag.run("What is revenue?", structured=True)
    print(result["answer"])
    print(result["sources"])
"""

import logging
from typing import Optional

from groq import Groq

import config
from retrieval.hybrid_retriever import HybridRetriever
from retrieval.reranker import Reranker
from chains.output_parsers import ChatAnswer, get_format_instructions

logger = logging.getLogger(__name__)

RAG_PROMPT = """You are a financial document analyst with deep knowledge of financial disclosures.

Your task: Answer the user's question using ONLY the provided context.
If the context does not contain enough information to answer the question,
say so clearly — do NOT make up or guess.

Be specific: cite numbers, years, and source documents where available.

{format_instructions}

Context from financial documents:
{context}

Question: {question}

{chat_history_block}

Answer:"""


class RAGChain:
    """
    Simple RAG pipeline: retrieve + rerank + LLM generation.
    
    Flow:
        1. Retrieve top-k chunks using hybrid (dense + sparse) retrieval
        2. Rerank chunks with Cohere for better relevance
        3. Format chunks as context
        4. Send to Groq LLM with context
        5. Parse and return structured answer + sources
    
    Args:
        hybrid_retriever: Optional pre-built HybridRetriever. Defaults to new instance.
        reranker:         Optional pre-built Reranker. Defaults to new instance.
        groq_client:      Optional pre-built Groq client. Defaults to new instance.
    """
    
    def __init__(
        self,
        hybrid_retriever: Optional[HybridRetriever] = None,
        reranker: Optional[Reranker] = None,
        groq_client: Optional[Groq] = None,
    ):
        self.retriever = hybrid_retriever or HybridRetriever()
        self.reranker = reranker or Reranker()
        
        if not config.GROQ_API_KEY:
            raise ValueError("GROQ_API_KEY not set in .env")
        self.client = groq_client or Groq(api_key=config.GROQ_API_KEY)
        
        logger.info("RAGChain initialised")
    
    # ------------------------------------------------------------------
    # Main: run the RAG pipeline
    # ------------------------------------------------------------------
    
    def run(
        self,
        question: str,
        chat_history: Optional[str] = None,
        structured: bool = False,
        k: int = 5,
    ) -> dict:
        """
        Run full RAG pipeline: retrieve → rerank → LLM → return answer + sources.
        
        Args:
            question:      User's question
            chat_history:  Optional multi-turn context (formatted prior turns)
            structured:    If True, parse output as ChatAnswer (Pydantic)
            k:             Number of final chunks to send to LLM
        
        Returns:
            Dict with keys:
                - "answer": str — the LLM's response
                - "sources": list[dict] — each {"source": str, "page": int}
        """
        logger.info(f"RAGChain.run: '{question[:80]}'")
        
        # 1. Retrieve
        raw = self.retriever.retrieve(question, k=k * 2)
        if not raw:
            logger.warning("Retriever returned no results")
        
        # 2. Rerank
        ranked = self.reranker.rerank(question, raw, top_n=k)
        if not ranked:
            logger.warning("Reranker returned no results")
        
        # 3. Format context
        context = self.retriever.format_context(ranked)
        if not context.strip():
            context = "[No relevant documents found]"
        
        # 4. Build prompt
        chat_hist_str = ""
        if chat_history:
            chat_hist_str = f"\nPrevious conversation:\n{chat_history}"
        
        format_inst = get_format_instructions() if structured else ""
        
        prompt = RAG_PROMPT.format(
            format_instructions=format_inst,
            context=context,
            question=question,
            chat_history_block=chat_hist_str,
        )
        
        # 5. Call LLM
        answer_text = self._call_groq(prompt)
        
        # 6. Extract sources
        sources = [
            {"source": r["source"], "page": r["page"]}
            for r in ranked
        ]
        
        logger.info(f"RAGChain returned answer + {len(sources)} sources")
        
        return {
            "answer": answer_text,
            "sources": sources,
        }
    
    # ------------------------------------------------------------------
    # LLM call with error handling
    # ------------------------------------------------------------------
    
    def _call_groq(self, prompt: str) -> str:
        """
        Call Groq API with graceful error handling.
        
        Args:
            prompt: Full prompt to send to LLM
        
        Returns:
            LLM response text, or error message
        """
        try:
            response = self.client.chat.completions.create(
                model=config.GROQ_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=config.LLM_TEMPERATURE,
            )
            return response.choices[0].message.content
        
        except Exception as e:
            logger.error(f"Groq API error: {e}")
            return (
                f"Error calling LLM: {e}. "
                "Please check your GROQ_API_KEY and try again."
            )
    
    # ------------------------------------------------------------------
    # Convenience: get collection info
    # ------------------------------------------------------------------
    
    def get_indexed_count(self) -> int:
        """Return number of chunks currently indexed in ChromaDB."""
        try:
            return self.retriever.dense.collection.count()
        except Exception:
            return 0


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    from utils.logging_config import setup_logging
    setup_logging()
    
    rag = RAGChain()
    
    print(f"\n{'='*60}")
    print("RAGChain Smoke Test")
    print(f"{'='*60}")
    
    print(f"Indexed chunks: {rag.get_indexed_count()}")
    
    if rag.get_indexed_count() == 0:
        print("\n⚠️  No documents indexed yet.")
        print("Run: python -m ingestion.indexer")
        sys.exit(1)
    
    question = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else \
        "What is the total revenue?"
    
    print(f"\nQuestion: {question}")
    result = rag.run(question, structured=False)
    
    print(f"\n{'='*60}")
    print("Answer:")
    print(f"{'='*60}")
    print(result["answer"])
    
    print(f"\n{'='*60}")
    print("Sources:")
    for src in result["sources"]:
        print(f"  - {src['source']} (page {src['page']})")
    print(f"{'='*60}")