"""
frontend/app.py — Financial Document Intelligence (Final)

Features:
- Hybrid retrieval (Dense + BM25)
- Cohere reranking
- Groq LLM
- Conversation memory
- Source display
- Reset conversation
- Empty index warning
- Streamlit session history

Run:
    streamlit run frontend/app.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils.logging_config import setup_logging
setup_logging()

import streamlit as st

from chains.rag_chain import RAGChain
from chains.conversational import ConversationalRAGChain

# -----------------------------------------------------------------------------
# Page Config
# -----------------------------------------------------------------------------

st.set_page_config(
    page_title="Financial Document Intelligence",
    page_icon="📄",
)

st.title("📄 Financial Document Intelligence")
st.caption(
    "Hybrid Retrieval (Dense + BM25) • Cohere Reranking • Groq LLM • Conversation Memory"
)

# -----------------------------------------------------------------------------
# Cached Components
# -----------------------------------------------------------------------------


@st.cache_resource
def load_rag_chain() -> RAGChain:
    """
    Heavy, stateless components are cached globally.

    Includes:
    - Embedding model
    - Chroma vector store
    - BM25 index
    - Cohere reranker
    - Groq client
    """
    return RAGChain()


rag_chain = load_rag_chain()

# -----------------------------------------------------------------------------
# Warn if no documents are indexed
# -----------------------------------------------------------------------------

try:
    if (
        hasattr(rag_chain, "dense_retriever")
        and rag_chain.dense_retriever.collection.count() == 0
    ):
        st.warning(
            "No documents indexed yet.\n\n"
            "Run:\n"
            "`python -m ingestion.indexer`"
        )
except Exception:
    pass

# -----------------------------------------------------------------------------
# Session State
# -----------------------------------------------------------------------------

if "conv_chain" not in st.session_state:
    # Conversation memory should NOT be cached globally.
    st.session_state.conv_chain = ConversationalRAGChain(rag_chain=rag_chain)

if "messages" not in st.session_state:
    st.session_state.messages = []

# -----------------------------------------------------------------------------
# Sidebar
# -----------------------------------------------------------------------------

with st.sidebar:
    st.header("Options")

    if st.button("🔄 Reset Conversation"):
        st.session_state.conv_chain.reset()
        st.session_state.messages = []
        st.rerun()

# -----------------------------------------------------------------------------
# Previous Messages
# -----------------------------------------------------------------------------

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

        if msg.get("sources"):
            st.caption("**Sources:** " + ", ".join(msg["sources"]))

# -----------------------------------------------------------------------------
# Chat Input
# -----------------------------------------------------------------------------

question = st.chat_input("Ask a question about your indexed financial documents...")

if question:

    # Store user message
    st.session_state.messages.append(
        {
            "role": "user",
            "content": question,
        }
    )

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):

        with st.spinner("Retrieving, reranking, and generating answer..."):
            result = st.session_state.conv_chain.ask(question)

        # ---------------------------------------------------------------------
        # Answer
        # ---------------------------------------------------------------------

        st.markdown(result.answer)

        # ---------------------------------------------------------------------
        # Sources
        # ---------------------------------------------------------------------

        source_strings = [
            f"{src.source} (p.{src.page})"
            for src in result.sources
        ]

        if source_strings:
            st.caption("**Sources:** " + ", ".join(source_strings))

        # ---------------------------------------------------------------------
        # Context warning
        # ---------------------------------------------------------------------

        if not result.found_in_context:
            st.info(
                "The indexed documents may not contain enough information "
                "to fully answer this question."
            )

        # ---------------------------------------------------------------------
        # Retrieved Chunks (optional debug)
        # ---------------------------------------------------------------------

        if hasattr(result, "sources") and result.sources:
            with st.expander(f"Retrieved {len(result.sources)} chunk(s)"):

                for chunk in result.sources:

                    st.markdown(
                        f"**{chunk.source} · page {chunk.page}**"
                    )

                    if hasattr(chunk, "score"):
                        st.caption(f"Score: {chunk.score:.4f}")

                    if hasattr(chunk, "text"):
                        st.text(chunk.text[:500])

    # Store assistant response

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": result.answer,
            "sources": source_strings,
        }
    )