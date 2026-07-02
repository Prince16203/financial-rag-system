"""
frontend/app.py — Financial Document Intelligence

Features:
- Toggle between Phase 2 Hybrid RAG and Phase 3 LangGraph ReAct Agent
- Hybrid retrieval (Dense + BM25)
- Cohere reranking
- Groq LLM
- Conversation memory
- Source citations
- Reset conversation
- Empty index warning

Run from repo root:
    streamlit run frontend/app.py
"""

import sys
from pathlib import Path

# Make repo root importable regardless of where Streamlit is launched from
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Setup logging ONCE before any other project imports
from utils.logging_config import setup_logging
setup_logging()

import streamlit as st

from chains.rag_chain import RAGChain
from chains.conversational import ConversationalRAGChain
from agents.react_agent import FinancialAgent

# -----------------------------------------------------------------------------
# Page config — must be first Streamlit call
# -----------------------------------------------------------------------------

st.set_page_config(
    page_title="Financial Document Intelligence",
    page_icon="📄",
    layout="wide",
)

st.title("📄 Financial Document Intelligence")

# -----------------------------------------------------------------------------
# Cached heavy components — shared across all sessions
# -----------------------------------------------------------------------------

@st.cache_resource
def load_rag_chain() -> RAGChain:
    return RAGChain()

@st.cache_resource
def load_financial_agent() -> FinancialAgent:
    return FinancialAgent()


rag_chain = load_rag_chain()
financial_agent = load_financial_agent()

# -----------------------------------------------------------------------------
# Empty index warning
# -----------------------------------------------------------------------------

try:
    doc_count = rag_chain.retriever.dense.collection.count()
    if doc_count == 0:
        st.warning(
            "⚠️ No documents indexed yet.\n\n"
            "Run this first:\n"
            "```\npython -m ingestion.indexer\n```"
        )
    else:
        st.sidebar.success(f"✅ {doc_count} chunks indexed")
except Exception:
    pass

# -----------------------------------------------------------------------------
# Session state initialization
# -----------------------------------------------------------------------------

if "conv_chain" not in st.session_state:
    st.session_state.conv_chain = ConversationalRAGChain(rag_chain=rag_chain)

if "messages" not in st.session_state:
    st.session_state.messages = []

# -----------------------------------------------------------------------------
# Sidebar Configuration
# -----------------------------------------------------------------------------

with st.sidebar:
    st.header("Options")

    # Mode Selector: Choose between standard RAG lookup or the ReAct tool-calling agent
    pipeline_mode = st.radio(
        "Select Pipeline Mode:",
        options=["Phase 3: ReAct Agent (SQL + Math + Search)", "Phase 2: Hybrid RAG Chain"],
        index=0,
        help="ReAct Agent leverages SQL tools and a calculator for exact data calculations. Hybrid RAG is vector-only."
    )

    st.divider()

    if st.button("🔄 Reset Conversation"):
        st.session_state.conv_chain.reset()
        st.session_state.messages = []
        st.rerun()

    st.divider()
    st.caption(f"Active Mode: {pipeline_mode}")

# Update main view caption based on selected mode
st.caption(
    "Hybrid Retrieval (Dense + BM25) · Cohere Reranking · Groq LLM · "
    f"{'LangGraph ReAct Loop' if 'Agent' in pipeline_mode else 'Conversational Memory'}"
)

# -----------------------------------------------------------------------------
# Replay previous messages
# -----------------------------------------------------------------------------

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            st.caption("**Sources:** " + ", ".join(msg["sources"]))

# -----------------------------------------------------------------------------
# Chat input & Routing Logic
# -----------------------------------------------------------------------------

question = st.chat_input("Ask a question about your indexed financial documents...")

if question:

    # -- User turn --
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    # -- Assistant turn --
    with st.chat_message("assistant"):
        
        # Scenario A: Running the ReAct Agent (Your Phase 3 work via FastAPI Backend)
        if "Agent" in pipeline_mode:
            with st.spinner("FastAPI Server executing agent reasoning and tools..."):
                try:
                    import requests
                    
                    # Send the question to your running backend server over local network
                    response = requests.post(
                        "http://localhost:8000/api/chat",
                        json={"question": question},
                        timeout=60
                    )
                    
                    if response.status_code == 200:
                        # Parse out the clean answer from your ChatResponse schema payload
                        final_answer = response.json().get("answer")
                    else:
                        final_answer = f"⚠️ Backend Server Error: {response.text}"
                        
                except Exception as e:
                    final_answer = f"⚠️ Could not reach FastAPI server: {str(e)}\nMake sure python api/main.py is running!"
            
            # Output response text
            st.markdown(final_answer)
            source_strings = []

        # Scenario B: Running the classic Hybrid RAG Chain (Your friend's Phase 2 work)
        else:
            with st.spinner("Retrieving text chunks, reranking context, generating answer..."):
                result = st.session_state.conv_chain.ask(question)

            # Output response text
            st.markdown(result.answer)

            # Map citation schema objects cleanly
            source_strings = [
                f"{src.source} (p.{src.page})"
                for src in result.sources
            ]
            if source_strings:
                st.caption("**Sources:** " + ", ".join(source_strings))

            # Context quality validation notification
            if not result.found_in_context:
                st.info(
                    "The indexed vector documents may not contain enough information "
                    "to fully answer this question."
                )

            # Source collapse container
            if result.sources:
                with st.expander(f"📎 {len(result.sources)} source(s) used"):
                    for src in result.sources:
                        st.markdown(f"**{src.source}** · page {src.page}")

    # Save to session history array for layout state rendering replays
    st.session_state.messages.append({
        "role": "assistant",
        "content": final_answer if "Agent" in pipeline_mode else result.answer,
        "sources": source_strings,
    })


# """
# frontend/app.py — Financial Document Intelligence

# Features:
# - Hybrid retrieval (Dense + BM25)
# - Cohere reranking
# - Groq LLM
# - Conversation memory
# - Source citations
# - Reset conversation
# - Empty index warning

# Run from repo root:
#     streamlit run frontend/app.py
# """

# import sys
# from pathlib import Path
# from agents.react_agent import FinancialAgent

# # Make repo root importable regardless of where Streamlit is launched from
# sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# # Setup logging ONCE before any other project imports
# from utils.logging_config import setup_logging
# setup_logging()

# import streamlit as st

# from chains.rag_chain import RAGChain
# from chains.conversational import ConversationalRAGChain

# # -----------------------------------------------------------------------------
# # Page config — must be first Streamlit call
# # -----------------------------------------------------------------------------

# st.set_page_config(
#     page_title="Financial Document Intelligence",
#     page_icon="📄",
#     layout="wide",
# )

# st.title("📄 Financial Document Intelligence")
# st.caption(
#     "Hybrid Retrieval (Dense + BM25) · Cohere Reranking · "
#     "Groq LLM · Conversation Memory"
# )

# # -----------------------------------------------------------------------------
# # Cached heavy components — shared across all sessions
# # Embedding model, BM25 index, ChromaDB connection, Groq client
# # are expensive to load; cache them once.
# # -----------------------------------------------------------------------------

# @st.cache_resource
# def load_rag_chain() -> RAGChain:
#     return RAGChain()


# rag_chain = load_rag_chain()

# # -----------------------------------------------------------------------------
# # Empty index warning
# # Correct path: RAGChain.retriever (HybridRetriever)
# #               → .dense (DenseRetriever)
# #               → .collection (ChromaDB collection)
# # -----------------------------------------------------------------------------

# try:
#     doc_count = rag_chain.retriever.dense.collection.count()
#     if doc_count == 0:
#         st.warning(
#             "⚠️ No documents indexed yet.\n\n"
#             "Run this first:\n"
#             "```\npython -m ingestion.indexer\n```"
#         )
#     else:
#         st.sidebar.success(f"✅ {doc_count} chunks indexed")
# except Exception:
#     pass

# # -----------------------------------------------------------------------------
# # Session state
# # ConversationalRAGChain holds per-user conversation history —
# # must NOT be @st.cache_resource (that would share history across all users).
# # -----------------------------------------------------------------------------

# if "conv_chain" not in st.session_state:
#     st.session_state.conv_chain = ConversationalRAGChain(rag_chain=rag_chain)

# if "messages" not in st.session_state:
#     st.session_state.messages = []

# # -----------------------------------------------------------------------------
# # Sidebar
# # -----------------------------------------------------------------------------

# with st.sidebar:
#     st.header("Options")

#     if st.button("🔄 Reset Conversation"):
#         st.session_state.conv_chain.reset()
#         st.session_state.messages = []
#         st.rerun()

#     st.divider()
#     st.caption("Phase 2 — Hybrid RAG")

# # -----------------------------------------------------------------------------
# # Replay previous messages
# # -----------------------------------------------------------------------------

# for msg in st.session_state.messages:
#     with st.chat_message(msg["role"]):
#         st.markdown(msg["content"])
#         if msg.get("sources"):
#             st.caption("**Sources:** " + ", ".join(msg["sources"]))

# # -----------------------------------------------------------------------------
# # Chat input
# # -----------------------------------------------------------------------------

# question = st.chat_input("Ask a question about your indexed financial documents...")

# if question:

#     # -- User turn --
#     st.session_state.messages.append({"role": "user", "content": question})
#     with st.chat_message("user"):
#         st.markdown(question)

#     # -- Assistant turn --
#     with st.chat_message("assistant"):

#         with st.spinner("Retrieving, reranking, generating answer..."):
#             result = st.session_state.conv_chain.ask(question)

#         # Answer text
#         st.markdown(result.answer)

#         # Source citations (result.sources are SourceCitation objects)
#         source_strings = [
#             f"{src.source} (p.{src.page})"
#             for src in result.sources
#         ]
#         if source_strings:
#             st.caption("**Sources:** " + ", ".join(source_strings))

#         # Context quality warning
#         if not result.found_in_context:
#             st.info(
#                 "The indexed documents may not contain enough information "
#                 "to fully answer this question."
#             )

#         # Debug expander — shows SourceCitation fields only (no text/score,
#         # those live on raw chunk dicts which are not stored on ChatAnswer)
#         if result.sources:
#             with st.expander(f"📎 {len(result.sources)} source(s) used"):
#                 for src in result.sources:
#                     st.markdown(f"**{src.source}** · page {src.page}")

#     # Store for replay
#     st.session_state.messages.append({
#         "role": "assistant",
#         "content": result.answer,
#         "sources": source_strings,
#     })