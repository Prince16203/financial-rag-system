# Financial Document Intelligence System — Phase 1

Dense retrieval RAG pipeline for querying financial PDFs (10-K filings,
prospectuses, risk reports) using natural language.

---

## What Phase 1 Does

```
PDF(s)
   │
   ▼
PDF Parser          [ingestion/pdf_parser.py]
   │                Extracts text + tables from each page
   ▼
Chunker             [ingestion/chunker.py]
   │                Splits pages into ~1000-char overlapping chunks
   ▼
Embedding Model     [ingestion/embedder.py]
   │                BAAI/bge-base-en-v1.5 → 768-dim vectors (runs locally)
   ▼
ChromaDB            [data/chroma_db/]
   │                Persistent vector store on disk
   ▼
Dense Retriever     [retrieval/dense_retriever.py]
   │                Cosine similarity search → top-k relevant chunks
   ▼
Groq LLaMA 3.1      [frontend/app.py]
   │                Generates grounded answer from retrieved context
   ▼
Streamlit Chat UI   [frontend/app.py]
                    Browser-based Q&A interface with source citations
```

---

## Project Structure

```
financial-rag-system/
├── ingestion/
│   ├── __init__.py
│   ├── pdf_parser.py        # PDF text + table extraction
│   ├── chunker.py           # Text splitting
│   ├── embedder.py          # BGE embedding model
│   └── indexer.py           # ChromaDB writer + full pipeline runner
├── retrieval/
│   ├── __init__.py
│   └── dense_retriever.py   # Query → top-k chunks from ChromaDB
├── frontend/
│   └── app.py               # Streamlit chat UI
├── data/
│   ├── raw_pdfs/            # Drop your PDFs here
│   └── chroma_db/           # Auto-created by indexer (do not commit)
├── .env                     # Your API key (do not commit)
├── .env.example             # Template for .env
├── .gitignore
└── requirements.txt
```

---

## Setup

### 1. Clone and activate environment

```bash
git clone <your-repo-url>
cd financial-rag-system
conda activate .\.conda\          # Windows
# or
source repo_env/bin/activate      # Mac/Linux
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

First install takes 5-10 minutes — PyTorch and the BGE model are large.

### 3. Add your Groq API key

Create a `.env` file in the project root:

```
GROQ_API_KEY=your_key_here
```

Get a free key at: https://console.groq.com

### 4. Add PDFs

Drop one or more PDF files into `data/raw_pdfs/`.

Good sources for financial PDFs:
- SEC EDGAR: https://efts.sec.gov/LATEST/search-index?forms=10-K
- Company investor relations pages (Apple, Microsoft, JPMorgan etc.)

Any PDF works for testing — the filename becomes the citation label.

---

## Running the Pipeline

### Step 1 — Ingest PDFs into ChromaDB

Run once (or again whenever you add new PDFs):

```bash
python -m ingestion.indexer
```

This runs all four stages in sequence:
- Parses all PDFs in `data/raw_pdfs/`
- Chunks text into ~1000-char segments
- Embeds each chunk with BGE (downloads model on first run, ~400MB)
- Writes everything to `data/chroma_db/`

Expected output:
```
Starting full ingestion pipeline
  PDF dir:    data/raw_pdfs
  ChromaDB:   data/chroma_db
Parsing: test_doc.pdf
  Total pages: 106
  Extracted 104 non-blank pages
Chunking complete: 104 pages → 312 chunks
Loading embedding model: BAAI/bge-base-en-v1.5
Embedding 312 chunks...
Indexing complete: 312 chunks (total: 312)
```

### Step 2 — Test retrieval (optional)

Check that ChromaDB is returning relevant chunks before launching the UI:

```bash
python -m retrieval.dense_retriever "What is the total revenue?"
```

You should see 5 scored results with source filename and page number.

### Step 3 — Launch the chat UI

```bash
streamlit run frontend/app.py
```

Opens at http://localhost:8501 in your browser.

---

## Using the Chat UI

**Sidebar controls:**
- `Chunks to retrieve (k)` — how many document chunks are passed to the LLM (default 5)
- `Show source chunks` — toggle the expandable citation panel below each answer
- `Show similarity scores` — show cosine similarity score per chunk
- `Refresh collection stats` — see how many chunks are indexed and which PDFs

**Example questions to try:**
- What is the total revenue reported?
- What are the main risk factors disclosed?
- Summarise the key business segments.
- What did management say about cloud growth?
- What is the company's stated investment objective?

**How answers work:**
Every answer is generated only from the retrieved document chunks.
If the answer is not in your PDFs, the system says so rather than
making something up.

---

## Adding More PDFs

```bash
# Drop new PDFs into data/raw_pdfs/ then re-run:
python -m ingestion.indexer
```

The indexer uses upsert — re-running on the same PDF is safe and will
not create duplicates.

To remove a specific PDF from the index, open a Python shell:

```python
from ingestion.indexer import Indexer
idx = Indexer()
idx.delete_source("old_filing.pdf")
```

---

## Smoke-Testing Individual Stages

Run each file directly to test one stage at a time:

```bash
# Test PDF parsing only
python -m ingestion.pdf_parser data/raw_pdfs/test_doc.pdf

# Test chunking only
python -m ingestion.chunker data/raw_pdfs/test_doc.pdf

# Test embedding only (embeds first 20 chunks as a quick check)
python -m ingestion.embedder data/raw_pdfs/test_doc.pdf

# Test retrieval only (ChromaDB must be populated first)
python -m retrieval.dense_retriever "What is the Sharpe ratio?"
```

---

## What NOT to Commit

The `.gitignore` already excludes these — do not force-add them:

```
.env                    # contains your API key
data/chroma_db/         # generated vector store (can be large)
data/raw_pdfs/*.pdf     # raw PDFs (often confidential or large)
repo_env/               # virtual environment
__pycache__/
```

---

## Files Already Pushed to GitHub

```
ingestion/
retrieval/
frontend/
requirements.txt
```

### Still to push (create these manually):

```bash
# 1. .env.example  (template without real key)
echo GROQ_API_KEY=your_groq_api_key_here > .env.example

# 2. .gitignore
# (see contents below)

# 3. README.md  (this file, renamed)
```

**.gitignore contents:**
```
repo_env/
.conda/
.env
data/chroma_db/
data/raw_pdfs/*.pdf
__pycache__/
*.pyc
.ipynb_checkpoints/
*.egg-info/
dist/
```

---

## Troubleshooting

**`No module named 'ingestion'`**
Run from the project root directory, not from inside a subfolder.
Always use `python -m ingestion.indexer` not `python ingestion/indexer.py`.

**`SyntaxError: source code string cannot contain null bytes`**
Your `__init__.py` was created with `echo` on Windows (writes UTF-16).
Fix: `python -c "open('ingestion/__init__.py', 'w').write('')"`

**`Collection 'financial_docs' not found`**
ChromaDB has not been populated yet. Run `python -m ingestion.indexer` first.

**`GROQ_API_KEY not found`**
Create a `.env` file in the project root with `GROQ_API_KEY=your_key`.

**BGE model download hangs**
The model is ~400MB. Let it finish once — it caches to
`~/.cache/huggingface/` and loads instantly on subsequent runs.

**Streamlit page is blank / spinner won't stop**
ChromaDB is probably empty. Run the indexer first, then restart Streamlit.

---

## Phase Roadmap

| Phase | What's Added                                    | Status      |
|-------|-------------------------------------------------|-------------|
| 1     | Ingestion + Dense Retrieval + Streamlit UI      | ✅ Complete |
| 2     | Hybrid Search (BM25 + Dense) + Reranking + Memory | ⏳ Next   |
| 3     | Metric Extraction + SQL Agent + LangGraph       | Planned     |
| 4     | RAGAS Evaluation + Ablation Study               | Planned     |
| 5     | FastAPI Backend + Docker + Kubernetes           | Planned     |
| 6     | Grafana Monitoring + Airflow Ingestion DAG      | Planned     |
| 7     | Documentation + Architecture Diagram + Demo     | Planned     |