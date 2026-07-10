"""
evaluation/ablation_study.py

Compares three retrieval configurations on the same QA dataset to
quantify the contribution of each pipeline upgrade.

Config A — Dense only          (Phase 1 baseline)
Config B — Hybrid, no rerank   (Phase 2 retrieval only)
Config C — Hybrid + reranker   (Phase 2 current best)

All configs use the same simple RAG prompt + Groq LLM for generation
so that only the retrieval layer changes between configs.

Usage:
    python evaluation/ablation_study.py
    python evaluation/ablation_study.py --sample 15
    python evaluation/ablation_study.py --configs A C
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

from retrieval.dense_retriever import DenseRetriever
from retrieval.hybrid_retriever import HybridRetriever
from retrieval.reranker import Reranker

try:
    from ragas import evaluate
    from ragas.metrics import (
        faithfulness,
        answer_relevancy,
        context_precision,
        context_recall,
    )
    from ragas.llms import LangchainLLMWrapper
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from datasets import Dataset
    from langchain_groq import ChatGroq
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import StrOutputParser
except ImportError as e:
    print(f"[ERROR] Missing dependency: {e}")
    print("Run: pip install ragas datasets langchain-huggingface")
    sys.exit(1)

# ── Constants ─────────────────────────────────────────────────────────────────
QA_PATH            = ROOT / "evaluation" / "qa_dataset.csv"
RESULTS_DIR        = ROOT / "evaluation" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

TOP_K              = 5
FETCH_K            = 15
INTER_QUERY_SLEEP  = 8
INTER_CONFIG_SLEEP = 45

METRIC_COLS = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]

RAG_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are a financial document analyst. Answer the question using only "
     "the provided context. Be concise and factual. "
     "If the context does not contain the answer, say so.\n\nContext:\n{context}"),
    ("human", "{question}"),
])


# ── Shared answer generator ───────────────────────────────────────────────────

def build_answer(question: str, results: list[dict], llm) -> str:
    """Generate answer from retrieved result dicts using shared RAG prompt."""
    context = "\n\n---\n\n".join(
        f"[Source: {r['source']} | Page {r['page']}]\n{r['text']}"
        for r in results
    )
    chain = RAG_PROMPT | llm | StrOutputParser()
    try:
        return chain.invoke({"context": context, "question": question})
    except Exception as exc:
        return f"ERROR: {exc}"


# ── Config definitions ────────────────────────────────────────────────────────

class ConfigA:
    name        = "A_Dense_Only"
    description = "Dense retrieval only — ChromaDB similarity search, no BM25, no reranker"

    def __init__(self):
        self.retriever = DenseRetriever()

    def retrieve(self, question: str) -> list[dict]:
        return self.retriever.retrieve(question, k=TOP_K)


class ConfigB:
    name        = "B_Hybrid_No_Rerank"
    description = "Hybrid retrieval (BM25 + Dense via RRF) — no Cohere reranker"

    def __init__(self):
        self.retriever = HybridRetriever()

    def retrieve(self, question: str) -> list[dict]:
        return self.retriever.retrieve(question, k=TOP_K)


class ConfigC:
    name        = "C_Hybrid_Plus_Rerank"
    description = "Hybrid retrieval (BM25 + Dense via RRF) + Cohere reranker"

    def __init__(self):
        self.retriever = HybridRetriever()
        self.reranker  = Reranker()

    def retrieve(self, question: str) -> list[dict]:
        raw = self.retriever.retrieve(question, k=FETCH_K)
        return self.reranker.rerank(question, raw, top_n=TOP_K)


CONFIG_MAP = {"A": ConfigA, "B": ConfigB, "C": ConfigC}


# ── Run one config over full QA set ──────────────────────────────────────────

def run_config(config_obj, df: pd.DataFrame, llm) -> list[dict]:
    print(f"\n  {config_obj.name}")
    print(f"  {config_obj.description}")
    print(f"  Running {len(df)} questions...\n")

    records = []
    for i, (_, row) in enumerate(df.iterrows()):
        question = row["question"]
        print(f"    [{row['question_id']}] {question[:65]}...")

        try:
            results  = config_obj.retrieve(question)
            answer   = build_answer(question, results, llm)
            contexts = [r["text"] for r in results]
            error    = None
        except Exception as exc:
            answer   = ""
            contexts = []
            error    = str(exc)
            print(f"           ⚠ {error[:80]}")

        records.append({
            "question_id":   row["question_id"],
            "question":      question,
            "answer":        answer,
            "contexts":      contexts,
            "ground_truth":  row["ground_truth"],
            "query_type":    row["query_type"],
            "difficulty":    row["difficulty"],
            "error":         error,
        })

        if i < len(df) - 1:
            time.sleep(INTER_QUERY_SLEEP)

    return records


# ── Score one config with RAGAS ───────────────────────────────────────────────

def ragas_score(records: list[dict], ragas_llm, ragas_embed, config_name: str) -> pd.DataFrame:
    valid = [r for r in records if r["answer"] and not r["error"]]
    if not valid:
        print(f"  [WARN] No valid records for {config_name}")
        return pd.DataFrame()

    ds = Dataset.from_dict({
        "question":     [r["question"]     for r in valid],
        "answer":       [r["answer"]       for r in valid],
        "contexts":     [r["contexts"]     for r in valid],
        "ground_truth": [r["ground_truth"] for r in valid],
    })

    metrics = [faithfulness, answer_relevancy, context_precision, context_recall]
    for m in metrics:
        m.llm = ragas_llm
        if hasattr(m, "embeddings"):
            m.embeddings = ragas_embed

    result = evaluate(dataset=ds, metrics=metrics, raise_exceptions=False)
    scores = result.to_pandas()

    meta = pd.DataFrame([{
        "question_id": r["question_id"],
        "query_type":  r["query_type"],
        "difficulty":  r["difficulty"],
    } for r in valid])

    out = pd.concat([meta.reset_index(drop=True), scores.reset_index(drop=True)], axis=1)
    out.insert(0, "config", config_name)
    return out


# ── Main ──────────────────────────────────────────────────────────────────────

def run_ablation(
    sample: int | None = None,
    configs_to_run: list[str] | None = None,
):
    print("\n" + "=" * 65)
    print("  Financial RAG — Ablation Study")
    print("=" * 65)

    if configs_to_run is None:
        configs_to_run = ["A", "B", "C"]

    df = pd.read_csv(QA_PATH)
    if sample:
        df = df.sample(n=min(sample, len(df)), random_state=42)
    print(f"\n[INFO] Evaluating {len(df)} questions across configs: {configs_to_run}")

    # Shared LLM for generation + RAGAS scoring
    llm = ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0,
        groq_api_key=os.getenv("GROQ_API_KEY"),
    )
    embed_model = HuggingFaceEmbeddings(
        model_name="BAAI/bge-base-en-v1.5",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    ragas_llm   = LangchainLLMWrapper(llm)
    ragas_embed = LangchainEmbeddingsWrapper(embed_model)

    all_scores   = {}
    summary_rows = []

    for key in configs_to_run:
        if key not in CONFIG_MAP:
            print(f"[WARN] Unknown config '{key}' — skipping.")
            continue

        print(f"\n{'─' * 65}")
        config_obj = CONFIG_MAP[key]()
        records    = run_config(config_obj, df, llm)

        print(f"\n  [INFO] RAGAS scoring for {config_obj.name}...")
        time.sleep(INTER_CONFIG_SLEEP)

        scores_df = ragas_score(records, ragas_llm, ragas_embed, config_obj.name)
        all_scores[key] = scores_df

        row = {
            "config":      config_obj.name,
            "description": config_obj.description,
            "evaluated":   len(scores_df),
        }
        for col in METRIC_COLS:
            row[f"mean_{col}"] = round(scores_df[col].mean(), 4) if col in scores_df.columns else None
        summary_rows.append(row)

        # Cooldown before next config
        if key != configs_to_run[-1]:
            print(f"\n  [WAIT] {INTER_CONFIG_SLEEP}s cooldown before next config...")
            time.sleep(INTER_CONFIG_SLEEP)

    # ── Comparison table ──────────────────────────────────────────────
    summary_df = pd.DataFrame(summary_rows)
    display    = ["config"] + [f"mean_{c}" for c in METRIC_COLS if f"mean_{c}" in summary_df.columns]

    print("\n\n" + "=" * 65)
    print("  ABLATION RESULTS — COMPARISON TABLE")
    print("=" * 65)
    print("\n" + summary_df[display].to_string(index=False))

    # Delta vs Config A
    baseline_rows = summary_df[summary_df["config"] == "A_Dense_Only"]
    if not baseline_rows.empty and len(summary_rows) > 1:
        baseline = baseline_rows.iloc[0]
        print("\n── Improvement over Config A (Dense Only) baseline ─────────")
        for _, row in summary_df.iterrows():
            if row["config"] == "A_Dense_Only":
                continue
            print(f"\n  vs {row['config']}:")
            for col in METRIC_COLS:
                key_col = f"mean_{col}"
                if key_col in row and row[key_col] is not None and baseline[key_col] is not None:
                    delta = row[key_col] - baseline[key_col]
                    sign  = "+" if delta >= 0 else ""
                    print(f"    {col:<28} {sign}{delta:.4f}  "
                          f"({baseline[key_col]:.4f} → {row[key_col]:.4f})")

    # Context recall by query type
    if all_scores:
        first_valid = next(
            (v for v in all_scores.values() if "context_recall" in v.columns), None
        )
        if first_valid is not None:
            print("\n── Context Recall by Query Type ────────────────────────────")
            frames = []
            for key, sdf in all_scores.items():
                if "context_recall" in sdf.columns:
                    tmp = (sdf.groupby("query_type")["context_recall"]
                             .mean().round(4)
                             .rename(CONFIG_MAP[key].name))
                    frames.append(tmp)
            if frames:
                print(pd.concat(frames, axis=1).to_string())

    # Save
    timestamp   = datetime.now().strftime("%Y%m%d_%H%M%S")
    detail_path = RESULTS_DIR / f"ablation_detail_{timestamp}.csv"
    summ_path   = RESULTS_DIR / f"ablation_summary_{timestamp}.csv"

    combined = pd.concat(list(all_scores.values()), ignore_index=True)
    combined.to_csv(detail_path, index=False)
    summary_df.to_csv(summ_path, index=False)

    print(f"\n[SAVED] {detail_path}")
    print(f"[SAVED] {summ_path}")
    print("\n[DONE] Ablation study complete.\n")
    return summary_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample",  type=int,    default=None)
    parser.add_argument("--configs", nargs="+",   default=["A", "B", "C"],
                        choices=["A", "B", "C"])
    args = parser.parse_args()

    run_ablation(sample=args.sample, configs_to_run=args.configs)