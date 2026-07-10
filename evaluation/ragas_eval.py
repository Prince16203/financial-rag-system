"""
evaluation/ragas_eval.py

Runs RAGAS evaluation over the current best pipeline (hybrid retrieval
+ Cohere reranker + ConversationalRAGChain) using the curated QA dataset.

Strategy:
  - Retrieve contexts separately (HybridRetriever + Reranker) so RAGAS
    has access to the raw text chunks it needs to score faithfulness
    and context recall.
  - Generate answer via ConversationalRAGChain.ask() — fresh instance
    per question so history does not bleed across eval turns.
  - Score with RAGAS: faithfulness, answer_relevancy, context_precision,
    context_recall.

Usage:
    python evaluation/ragas_eval.py
    python evaluation/ragas_eval.py --sample 10
    python evaluation/ragas_eval.py --type metric
    python evaluation/ragas_eval.py --difficulty easy
"""

import os
import sys
import time
import argparse
from pathlib import Path
from datetime import datetime
from unittest.mock import MagicMock

# ── Compatibility patch ───────────────────────────────────────────────────────
# RAGAS 0.4.x tries to import langchain_community.chat_models.vertexai which
# was removed in langchain-community 0.3+. Mock it so RAGAS loads cleanly.
for _mod in [
    "langchain_community.chat_models.vertexai",
    "langchain_community.chat_models.google_palm",
]:
    sys.modules.setdefault(_mod, MagicMock())
# ─────────────────────────────────────────────────────────────────────────────

import pandas as pd
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / ".env")

from retrieval.hybrid_retriever import HybridRetriever
from retrieval.reranker import Reranker
from chains.conversational import ConversationalRAGChain

try:
    from ragas import evaluate
    from ragas.metrics import (
        answer_relevancy,
        context_precision,
        context_recall,
    )
    from ragas.llms import LangchainLLMWrapper
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from datasets import Dataset
    from langchain_groq import ChatGroq
    from langchain_huggingface import HuggingFaceEmbeddings
except ImportError as e:
    print(f"[ERROR] Missing dependency: {e}")
    print("Run: pip install ragas datasets langchain-huggingface")
    sys.exit(1)

# ── Constants ─────────────────────────────────────────────────────────────────
QA_PATH     = ROOT / "evaluation" / "qa_dataset.csv"
RESULTS_DIR = ROOT / "evaluation" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

TOP_K              = 5
FETCH_K            = 15   # retrieve more, rerank down to TOP_K
INTER_QUERY_SLEEP  = 8    # seconds between questions (Groq rate limit)
RAGAS_SLEEP        = 20   # seconds before RAGAS scoring starts (Groq cooldown)


# ── Single-question pipeline run ──────────────────────────────────────────────

def run_pipeline_on_question(
    question: str,
    retriever: HybridRetriever,
    reranker: Reranker,
) -> dict:
    """
    1. Retrieve with hybrid retriever (large fetch_k)
    2. Rerank to top TOP_K
    3. Extract context strings from r["text"]
    4. Generate answer via a fresh ConversationalRAGChain (no history bleed)
    """
    try:
        # Step 1 — retrieve
        raw_results = retriever.retrieve(question, k=FETCH_K)

        # Step 2 — rerank
        reranked = reranker.rerank(question, raw_results, top_n=TOP_K)

        # Step 3 — contexts for RAGAS (raw text strings)
        contexts = [r["text"] for r in reranked]

        # Step 4 — generate answer (fresh chain = no history contamination)
        chain  = ConversationalRAGChain()
        result = chain.ask(question)
        answer = result.answer   # ChatAnswer.answer is a string

        return {"answer": answer, "contexts": contexts, "error": None}

    except Exception as exc:
        return {"answer": "", "contexts": [], "error": str(exc)}


# ── RAGAS evaluator setup ─────────────────────────────────────────────────────

def build_ragas_evaluator():
    """
    Use Groq llama-3.1-8b-instant as the RAGAS judge LLM and
    BAAI/bge as the embedding model — avoids OpenAI dependency.
    """
    groq_llm = ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0,
        groq_api_key=os.getenv("GROQ_API_KEY"),
    )
    embed_model = HuggingFaceEmbeddings(
        model_name="BAAI/bge-base-en-v1.5",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    return LangchainLLMWrapper(groq_llm), LangchainEmbeddingsWrapper(embed_model)


# ── Main ──────────────────────────────────────────────────────────────────────

def run_evaluation(
    sample: int | None = None,
    query_type: str | None = None,
    difficulty: str | None = None,
):
    print("\n" + "=" * 60)
    print("  Financial RAG — RAGAS Evaluation")
    print("=" * 60)

    # Load QA dataset
    df = pd.read_csv(QA_PATH)
    print(f"\n[INFO] Loaded {len(df)} questions")

    if query_type:
        df = df[df["query_type"] == query_type]
        print(f"[INFO] Filtered to type='{query_type}' → {len(df)} questions")
    if difficulty:
        df = df[df["difficulty"] == difficulty]
        print(f"[INFO] Filtered to difficulty='{difficulty}' → {len(df)} questions")
    if sample:
        df = df.sample(n=min(sample, len(df)), random_state=42)
        print(f"[INFO] Sampled {len(df)} questions")

    if df.empty:
        print("[ERROR] No questions after filtering.")
        sys.exit(1)

    # Initialise retrieval components once (expensive to reinit per question)
    print("\n[INFO] Initialising HybridRetriever...")
    retriever = HybridRetriever()

    print("[INFO] Initialising Reranker...")
    reranker = Reranker()

    # Run pipeline over each question
    print(f"\n[INFO] Running {len(df)} questions through pipeline...\n")
    records = []

    for i, (_, row) in enumerate(df.iterrows()):
        q_id     = row["question_id"]
        question = row["question"]
        gt       = row["ground_truth"]

        print(f"  [{q_id}] {question[:70]}...")
        result = run_pipeline_on_question(question, retriever, reranker)

        if result["error"]:
            print(f"         ⚠ Error: {result['error'][:80]}")

        records.append({
            "question_id":    q_id,
            "question":       question,
            "answer":         result["answer"],
            "contexts":       result["contexts"],
            "ground_truth":   gt,
            "query_type":     row["query_type"],
            "difficulty":     row["difficulty"],
            "source_document": row["source_document"],
            "pipeline_error": result["error"],
        })

        if i < len(df) - 1:
            time.sleep(INTER_QUERY_SLEEP)

    # Build RAGAS dataset — skip errored rows
    valid   = [r for r in records if r["answer"] and not r["pipeline_error"]]
    skipped = len(records) - len(valid)
    if skipped:
        print(f"\n[WARN] {skipped} questions skipped due to pipeline errors.")

    print(f"\n[INFO] Building RAGAS dataset from {len(valid)} valid results...")
    ragas_data = Dataset.from_dict({
        "question":     [r["question"]     for r in valid],
        "answer":       [r["answer"]       for r in valid],
        "contexts":     [r["contexts"]     for r in valid],
        "ground_truth": [r["ground_truth"] for r in valid],
    })

    # Configure RAGAS
    print("[INFO] Configuring RAGAS evaluator (Groq + BAAI/bge)...")
    ragas_llm, ragas_embed = build_ragas_evaluator()

    metrics = [answer_relevancy, context_precision, context_recall]
    for m in metrics:
        m.llm = ragas_llm
        if hasattr(m, "embeddings"):
            m.embeddings = ragas_embed

    print(f"\n[INFO] Running RAGAS scoring — please wait...\n")
    time.sleep(RAGAS_SLEEP)

    try:
        ragas_result = evaluate(
            dataset=ragas_data,
            metrics=metrics,
            raise_exceptions=False,
        )
    except Exception as exc:
        print(f"[ERROR] RAGAS failed: {exc}")
        sys.exit(1)

    # Merge scores with metadata
    scores_df = ragas_result.to_pandas()
    meta_df   = pd.DataFrame([{
        "question_id":    r["question_id"],
        "query_type":     r["query_type"],
        "difficulty":     r["difficulty"],
        "source_document": r["source_document"],
        "pipeline_error": r["pipeline_error"],
    } for r in valid])
    full_df = pd.concat(
        [meta_df.reset_index(drop=True), scores_df.reset_index(drop=True)],
        axis=1,
    )

    # Print summary
    metric_cols = ["answer_relevancy", "context_precision", "context_recall"]
    available   = [c for c in metric_cols if c in full_df.columns]

    print("\n" + "=" * 60)
    print("  RAGAS RESULTS SUMMARY")
    print("=" * 60)

    print("\n── Overall ─────────────────────────────────────────────────")
    for col in available:
        print(f"  {col:<28} {full_df[col].mean():.4f}")

    print("\n── By Query Type ───────────────────────────────────────────")
    print(full_df.groupby("query_type")[available].mean().round(4).to_string())

    print("\n── By Difficulty ───────────────────────────────────────────")
    print(full_df.groupby("difficulty")[available].mean().round(4).to_string())

    if "faithfulness" in full_df.columns and full_df["faithfulness"].notna().any():
        print("\n── Weakest Faithfulness (bottom 5) ─────────────────────────")
        weak = full_df.dropna(subset=["faithfulness"]).nsmallest(5, "faithfulness")
        for _, r in weak.iterrows():
            print(f"  [{r['question_id']}] {str(r.get('question', ''))[:60]}... → {r['faithfulness']:.3f}")
    elif "faithfulness" in full_df.columns:
        print("\n── Faithfulness: NaN — Groq rate limit hit during RAGAS scoring")
        print("  Re-run with --sample 5 or increase RAGAS_SLEEP")

    # Save
    timestamp    = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_path = RESULTS_DIR / f"ragas_results_{timestamp}.csv"
    summary_path = RESULTS_DIR / f"ragas_summary_{timestamp}.csv"

    full_df.to_csv(results_path, index=False)

    summary = {
        "timestamp":       timestamp,
        "total_questions": len(df),
        "evaluated":       len(valid),
        "skipped":         skipped,
        "retriever":       "hybrid_bm25_dense_cohere_rerank",
    }
    for col in available:
        summary[f"mean_{col}"] = round(full_df[col].mean(), 4)
    pd.DataFrame([summary]).to_csv(summary_path, index=False)

    print(f"\n[SAVED] {results_path}")
    print(f"[SAVED] {summary_path}")
    print("\n[DONE] RAGAS evaluation complete.\n")
    return full_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample",     type=int, default=None)
    parser.add_argument("--type",       type=str, default=None,
                        choices=["factual", "metric", "comparison", "multihop"])
    parser.add_argument("--difficulty", type=str, default=None,
                        choices=["easy", "medium", "hard"])
    args = parser.parse_args()

    run_evaluation(
        sample=args.sample,
        query_type=args.type,
        difficulty=args.difficulty,
    )