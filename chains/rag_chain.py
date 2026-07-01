"""
chains/rag_chain.py

Core RAG chain: question -> hybrid retrieve -> rerank -> Groq -> answer.
"""

import logging
from typing import Optional

import groq
from groq import Groq

import config
from retrieval.hybrid_retriever import HybridRetriever
from retrieval.reranker import Reranker

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You are a financial document analyst. Answer using ONLY the context "
    "provided below. If the context doesn't contain the answer, say so "
    "explicitly — do not guess or use outside knowledge."
)

class RAGChain:
    def __init__(
        self,
        hybrid_retriever: Optional[HybridRetriever] = None,
        reranker: Optional[Reranker] = None,
        fetch_k: int = config.HYBRID_FETCH_K,
        top_n: int = config.RERANK_TOP_N,
        groq_client: Optional[Groq] = None,
    ):
        self.retriever = hybrid_retriever or HybridRetriever()
        self.reranker  = reranker or Reranker()
        self.fetch_k   = fetch_k
        self.top_n     = top_n

        if not config.GROQ_API_KEY:
            raise ValueError("GROQ_API_KEY not set. Add it to your .env file.")
        self.client = groq_client or Groq(api_key=config.GROQ_API_KEY)

        logger.info(f"RAGChain ready | fetch_k={fetch_k} top_n={top_n}")

    def run(self, question: str, chat_history: Optional[str] = None, structured: bool = False) -> dict:
        raw = self.retriever.retrieve(question, k=self.fetch_k)
        chunks = self.reranker.rerank(question, raw, top_n=self.top_n)

        if not chunks:
            return {
                "answer": "I couldn't find anything relevant in the indexed documents.",
                "sources": [],
                "chunks": [],
            }

        context = self.retriever.format_context(chunks)
        history_block = f"\n\nPrevious conversation:\n{chat_history}\n" if chat_history else ""

        format_block = ""
        if structured:
            from chains.output_parsers import get_format_instructions
            format_block = f"\n\n{get_format_instructions()}"

        prompt = f"{SYSTEM_PROMPT}\n\nContext:\n{context}{history_block}\n\nQuestion: {question}{format_block}"

        answer = self._call_groq(prompt)
        sources = [{"source": c["source"], "page": c["page"]} for c in chunks]

        logger.info(f"RAGChain answered '{question[:60]}...' using {len(chunks)} chunks")
        return {"answer": answer, "sources": sources, "chunks": chunks}

    def _call_groq(self, prompt: str) -> str:
        """
        Calls Groq with explicit handling for every failure mode that would
        otherwise crash Streamlit. Groq's SDK already retries connection
        errors/timeouts/429/5xx twice internally — this catches what survives that.
        """
        try:
            response = self.client.chat.completions.create(
                model=config.GROQ_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=config.LLM_TEMPERATURE,
            )
            return response.choices[0].message.content

        except groq.RateLimitError as e:
            logger.error(f"Groq rate limit exceeded: {e}")
            return "I'm getting rate-limited right now — please wait a moment and try again."

        except groq.APITimeoutError as e:
            logger.error(f"Groq request timed out: {e}")
            return "The language model took too long to respond. Please try again."

        except groq.APIConnectionError as e:
            logger.error(f"Could not reach Groq: {e}")
            return "I can't reach the language model service right now. Check your connection and try again."

        except groq.AuthenticationError as e:
            logger.error(f"Groq authentication failed: {e}")
            return "There's a configuration issue with the language model service (invalid API key)."

        except groq.APIStatusError as e:
            logger.error(f"Groq returned an error status ({e.status_code}): {e}")
            return "The language model service returned an error. Please try again shortly."

        except groq.APIError as e:
            # Catch-all for anything not covered above
            logger.error(f"Unexpected Groq API error: {e}")
            return "Something went wrong generating the answer. Please try again."


if __name__ == "__main__":
    import sys
    from utils.logging_config import setup_logging
    setup_logging()

    question = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "What is the total revenue?"
    chain = RAGChain()
    result = chain.run(question)

    print(f"\n{'='*60}\nQuestion: {question}\n{'='*60}")
    print(f"\nAnswer:\n{result['answer']}")
    print(f"\nSources: {result['sources']}")

    