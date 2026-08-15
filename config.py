import os
from pathlib import Path
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent

# Load local .env without requiring an additional dependency.
# This is intentionally done before reading GROQ_API_KEY below.
ENV_FILE = BASE_DIR / ".env"
if ENV_FILE.exists():
    try:
        for raw_line in ENV_FILE.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and value and key not in os.environ:
                os.environ[key] = value
    except Exception:
        pass

DATA_DIR = BASE_DIR / "data"
PDF_DIR = DATA_DIR / "pdfs"
CHROMA_DIR = DATA_DIR / "chroma_db"
LOCAL_MODEL_DIR = BASE_DIR / "models" / "qwen2.5-1.5b-instruct"

COLLECTION_NAME = "apple_watch_static_docs"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LOCAL_LLM_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"

CHUNK_SIZE = 900
CHUNK_OVERLAP = 150
RETRIEVAL_K = 4
MAX_CONTEXT_CHARS = 10000
MAX_HISTORY_MESSAGES = 8

GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
APPLE_REGION = "in"
APPLE_BASE = "https://www.apple.com/in"


def get_groq_api_key():
    try:
        if hasattr(st, "secrets") and "GROQ_API_KEY" in st.secrets:
            value = str(st.secrets["GROQ_API_KEY"]).strip()
            if value:
                return value
    except Exception:
        pass
    return os.getenv("GROQ_API_KEY", "").strip() or None


def ensure_dirs():
    for path in (DATA_DIR, PDF_DIR, CHROMA_DIR, LOCAL_MODEL_DIR.parent):
        path.mkdir(parents=True, exist_ok=True)


ensure_dirs()
