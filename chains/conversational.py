"""
chains/conversational.py

Adds multi-turn memory on top of RAGChain: formats prior turns into a
chat_history string, asks RAGChain for a structured answer (via
output_parsers.py), and tracks the running conversation.

Usage:
    from chains.conversational import ConversationalRAGChain
    chain = ConversationalRAGChain()
    result = chain.ask("What was Q3 revenue?")
    result = chain.ask("How does that compare to Q2?")  # remembers prior turn
"""

import logging
from typing import Optional

from chains.rag_chain import RAGChain
from chains.output_parsers import ChatAnswer, SourceCitation, parse_chat_answer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

import config
MAX_HISTORY_TURNS = config.MAX_HISTORY_TURNS


class ConversationalRAGChain:
    """
    Wraps RAGChain with conversation memory.

    Args:
        rag_chain:          Optional pre-built RAGChain (avoids reloading models).
        max_history_turns:  How many prior (question, answer) pairs to keep in context.
    """

    def __init__(self, rag_chain: Optional[RAGChain] = None, max_history_turns: int = MAX_HISTORY_TURNS):
        self.chain = rag_chain or RAGChain()
        self.max_history_turns = max_history_turns
        self.history: list[dict] = []   # [{"question": ..., "answer": ...}, ...]

    def _format_history(self) -> Optional[str]:
        """Render recent turns as a text block for the prompt. None if no history yet."""
        if not self.history:
            return None
        recent = self.history[-self.max_history_turns:]
        lines = [f"Q: {t['question']}\nA: {t['answer']}" for t in recent]
        return "\n\n".join(lines)

    def ask(self, question: str) -> ChatAnswer:
        """
        Answer a question with conversation context.

        Args:
            question: Natural language question. May reference prior turns
                      ("what about last year?").

        Returns:
            Validated ChatAnswer (answer, sources, found_in_context).
        """
        chat_history = self._format_history()
        result = self.chain.run(question, chat_history=chat_history, structured=True)

        structured = parse_chat_answer(result["answer"])

        # Always trust the retrieval pipeline's actual sources over whatever
        # (if anything) the LLM claimed inside its JSON.
        structured.sources = [
            SourceCitation(source=s["source"], page=s["page"]) for s in result["sources"]
        ]

        self.history.append({"question": question, "answer": structured.answer})
        logger.info(f"Conversational turn {len(self.history)} | Q: '{question[:60]}...'")
        return structured

    def reset(self) -> None:
        """Clear conversation history (e.g. on 'new chat')."""
        self.history = []
        logger.info("Conversation history reset")


if __name__ == "__main__":
    chain = ConversationalRAGChain()
    print("Conversational RAG — type 'exit' to quit, 'reset' to clear history.\n")
    while True:
        q = input("You: ").strip()
        if q.lower() == "exit":
            break
        if q.lower() == "reset":
            chain.reset()
            continue
        result = chain.ask(q)
        print(f"\nAssistant: {result.answer}")
        if result.sources:
            print("Sources:", ", ".join(f"{s.source} (p.{s.page})" for s in result.sources))
        print()