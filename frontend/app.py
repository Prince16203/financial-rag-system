"""
frontend/app.py

Minimal Streamlit chat UI for the financial RAG system.
Phase 1 version: query -> retrieve chunks -> Groq LLM -> answer with citations.

Run:
    streamlit run frontend/app.py

Requires:
    .env file in project root with:
        GROQ_API_KEY=your_key_here

    ChromaDB already populated:
        python -m ingestion.indexer   (or run_ingestion_pipeline)
"""

import os
import sys

import streamlit as st
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Path fix so imports work when run via `streamlit run frontend/app.py`
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

load_dotenv()

# ---------------------------------------------------------------------------
# Lazy imports (cached so model loads only once across reruns)
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="Loading retriever and embedding model...")
def load_retriever():
    from retrieval.dense_retriever import DenseRetriever
    # return DenseRetriever(default_k=5)
    return DenseRetriever(default_k=10)


@st.cache_resource(show_spinner="Connecting to Groq...")
def load_llm():
    from langchain_groq import ChatGroq
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        st.error("GROQ_API_KEY not found. Add it to your .env file.")
        st.stop()
    return ChatGroq(
        model="llama-3.1-8b-instant",
        api_key=api_key,
        temperature=0.1,         # low temp for factual financial answers
        max_tokens=1024,
    )


# ---------------------------------------------------------------------------
# RAG answer function
# ---------------------------------------------------------------------------

def get_answer(query: str, retriever, llm) -> tuple[str, list[dict]]:
    """
    Retrieve relevant chunks, build prompt, call LLM, return answer + sources.

    Returns:
        (answer_text, list_of_source_dicts)
    """
    # Step 1: Retrieve
    results = retriever.retrieve(query, k=5)

    if not results:
        return (
            "I could not find relevant information in the indexed documents. "
            "Please make sure PDFs are ingested first.",
            [],
        )

    # Step 2: Build context
    context = retriever.format_context(results, max_chars=4000)

    # Step 3: Prompt
    prompt = f"""You are a financial document analyst. Answer the question below
using ONLY the provided context from financial documents.

Rules:
- Be factual and concise.
- If the answer is not in the context, say "This information is not available in the provided documents."
- Do NOT make up numbers or facts.
- Keep your answer focused and under 300 words.

CONTEXT:
{context}

QUESTION: {query}

ANSWER:"""

    # Step 4: Call LLM
    from langchain_core.messages import HumanMessage
    response = llm.invoke([HumanMessage(content=prompt)])
    answer   = response.content.strip()

    return answer, results


# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Financial RAG",
    page_icon="📊",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.title("📊 Financial RAG")
    st.markdown("**Phase 1** — Dense Retrieval + Groq LLaMA")
    st.divider()

    st.markdown("### Settings")
    top_k = st.slider("Chunks to retrieve (k)", min_value=1, max_value=10, value=5)
    show_sources = st.toggle("Show source chunks", value=True)
    show_scores  = st.toggle("Show similarity scores", value=False)

    st.divider()
    st.markdown("### Ingest new PDFs")
    st.markdown(
        "Drop PDFs into `data/raw_pdfs/` then run:\n"
        "```\npython -m ingestion.indexer\n```"
    )

    # Collection stats
    st.divider()
    if st.button("Refresh collection stats"):
        try:
            from ingestion.indexer import Indexer
            idx   = Indexer()
            stats = idx.get_collection_stats()
            st.metric("Chunks indexed", stats["total_chunks"])
            st.markdown("**Sources:**")
            for s in stats["sources"]:
                st.markdown(f"- `{s}`")
        except Exception as e:
            st.error(f"ChromaDB not ready: {e}")

# ---------------------------------------------------------------------------
# Main chat area
# ---------------------------------------------------------------------------

st.title("Financial Document Q&A")
st.caption("Ask questions about your indexed financial documents.")

# Init session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Render chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and show_sources and msg.get("sources"):
            _render_sources(msg["sources"], show_scores)

# ---------------------------------------------------------------------------
# Source rendering helper (defined before use in history loop above)
# ---------------------------------------------------------------------------

def _render_sources(sources: list[dict], show_scores: bool) -> None:
    with st.expander(f"📄 Sources ({len(sources)} chunks)", expanded=False):
        for i, r in enumerate(sources, 1):
            score_str = f" | Score: {r['score']}" if show_scores else ""
            st.markdown(
                f"**[{i}] {r['source']} — Page {r['page']}**{score_str}"
            )
            st.caption(r["text"][:400] + ("..." if len(r["text"]) > 400 else ""))
            if i < len(sources):
                st.divider()


# Re-render sources in history (function now defined)
# (Streamlit rerenders top-to-bottom so we patch history display here)

# ---------------------------------------------------------------------------
# Query input
# ---------------------------------------------------------------------------

if query := st.chat_input("Ask a question about your financial documents..."):

    # Load resources (cached after first load)
    retriever = load_retriever()
    retriever.default_k = top_k
    llm       = load_llm()

    # Show user message
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    # Generate answer
    with st.chat_message("assistant"):
        with st.spinner("Searching documents and generating answer..."):
            try:
                answer, sources = get_answer(query, retriever, llm)
            except Exception as e:
                answer  = f"Error: {e}"
                sources = []

        st.markdown(answer)

        if show_sources and sources:
            _render_sources(sources, show_scores)

    # Store in history
    st.session_state.messages.append({
        "role":    "assistant",
        "content": answer,
        "sources": sources,
    })

# ---------------------------------------------------------------------------
# Empty state hint
# ---------------------------------------------------------------------------

if not st.session_state.messages:
    st.info(
        "No messages yet. Try asking:\n\n"
        "- *What is the total revenue reported?*\n"
        "- *What are the main risk factors disclosed?*\n"
        "- *Summarise the investment objectives.*"
    )