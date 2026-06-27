"""
chains/output_parsers.py

Pydantic schemas defining the shape of LLM output for the conversational
RAG chain. conversational.py imports from here.

Usage:
    from chains.output_parsers import ChatAnswer, get_format_instructions, parse_chat_answer
"""

from pydantic import BaseModel, Field
from langchain_core.output_parsers import PydanticOutputParser


class SourceCitation(BaseModel):
    """A single source reference backing part of an answer."""
    source: str = Field(description="PDF filename the information came from")
    page: int = Field(description="Page number within that PDF")


class ChatAnswer(BaseModel):
    """Structured shape every conversational answer conforms to."""
    answer: str = Field(description="The answer to the user's question, grounded in context")
    sources: list[SourceCitation] = Field(
        default_factory=list,
        description="Documents/pages the answer drew from",
    )
    found_in_context: bool = Field(
        default=True,
        description="False if the context didn't contain the answer and the model said so",
    )


chat_answer_parser = PydanticOutputParser(pydantic_object=ChatAnswer)


def get_format_instructions() -> str:
    """Instructions to append to a prompt so the LLM returns valid ChatAnswer JSON."""
    return chat_answer_parser.get_format_instructions()


def parse_chat_answer(llm_output: str) -> ChatAnswer:
    """
    Parse raw LLM text into a validated ChatAnswer.
    Falls back to wrapping the raw text as a plain answer if the model
    didn't return valid JSON (Groq/llama models aren't always perfectly compliant).
    """
    try:
        return chat_answer_parser.parse(llm_output)
    except Exception:
        return ChatAnswer(answer=llm_output, sources=[], found_in_context=True)