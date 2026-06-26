"""
config.py — centralized paths, model names, tunable defaults.
Every other module imports from here instead of hardcoding values.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent

DATA_DIR       = BASE_DIR / "data"
RAW_PDF_DIR    = DATA_DIR / "raw"
PROCESSED_DIR  = DATA_DIR / "processed"
EMBEDDINGS_DIR = DATA_DIR / "embeddings"     # ChromaDB persists here
EVALUATION_DIR = DATA_DIR / "evaluation"

for _dir in (RAW_PDF_DIR, PROCESSED_DIR, EMBEDDINGS_DIR, EVALUATION_DIR):
    _dir.mkdir(parents=True, exist_ok=True)

CHROMA_DB_PATH  = str(EMBEDDINGS_DIR)
COLLECTION_NAME = "financial_docs"

EMBEDDING_MODEL      = "BAAI/bge-base-en-v1.5"
EMBEDDING_BATCH_SIZE = 64

CHUNK_SIZE    = 1000
CHUNK_OVERLAP = 150

DEFAULT_TOP_K   = 5
SCORE_THRESHOLD = 0.0

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL   = "llama-3.3-70b-versatile"