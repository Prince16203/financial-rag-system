"""
frontend/app.py — minimal Phase 1 chat UI.
Flow: question -> DenseRetriever -> format_context() -> Groq LLM -> answer.
No citations/memory/hybrid search yet — that's Phase 2.

Run from repo root:  streamlit run frontend/app.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # make root importable

import streamlit as st
from groq import Groq

import config
from retrieval.dense_retriever import DenseRetriever

st.set_page_config(page_title="Financial Doc Intelligence — Phase 1", page_icon="📄")
st.title("📄 Financial Document Intelligence — Phase 1")
st.caption("Dense retrieval only · no citations or memory yet")


@st.cache_resource
def load_retriever() -> DenseRetriever:
    return DenseRetriever(
        collection_name=config.COLLECTION_NAME,
        chroma_path=config.CHROMA_DB_PATH,
        default_k=config.DEFAULT_TOP_K,
        score_threshold=config.SCORE_THRESHOLD,
    )


@st.cache_resource
def load_groq_client() -> Groq:
    if not config.GROQ_API_KEY:
        st.error("GROQ_API_KEY is not set. Add it to your .env file.")
        st.stop()
    return Groq(api_key=config.GROQ_API_KEY)


retriever = load_retriever()
client = load_groq_client()

if retriever.collection.count() == 0:
    st.warning("No documents indexed yet. Run `python -m ingestion.indexer` first.")

if "history" not in st.session_state:
    st.session_state.history = []

for msg in st.session_state.history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

question = st.chat_input("Ask a question about your indexed documents...")

if question:
    st.session_state.history.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Retrieving relevant chunks..."):
            results = retriever.retrieve(question)
            context = retriever.format_context(results)

        if not results:
            answer = "I couldn't find anything relevant in the indexed documents."
        else:
            prompt = (
                "You are a financial document analyst. Answer using ONLY the "
                "context below. If the context doesn't contain the answer, "
                "say so explicitly — do not guess.\n\n"
                f"Context:\n{context}\n\nQuestion: {question}"
            )
            with st.spinner("Generating answer..."):
                response = client.chat.completions.create(
                    model=config.GROQ_MODEL,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.1,
                )
                answer = response.choices[0].message.content

        st.markdown(answer)
        with st.expander(f"Retrieved {len(results)} chunk(s)"):
            for r in results:
                st.markdown(f"**{r['source']} · page {r['page']}** (score: {r['score']})")
                st.text(r["text"][:300])

    st.session_state.history.append({"role": "assistant", "content": answer})