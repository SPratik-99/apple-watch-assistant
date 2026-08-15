# Apple Watch Expert v2

A Streamlit conversational Apple Watch assistant with a clean separation between **static knowledge** and **live information**.

## Architecture

- **PDF RAG:** manuals, technical information, setup, troubleshooting and other stable knowledge.
- **Apple.com client:** current prices, current lineup, availability and other changing information.
- **Router:** decides whether a question needs PDF evidence, live Apple evidence, or both.
- **LLM:** exactly two choices — Groq or an offline Hugging Face model.
- **Conversation layer:** keeps recent chat context so follow-up questions work naturally.

## Important data rule

Do **not** put prices, offers, availability or other dynamic data into `data/pdfs/`. Those are fetched from Apple.com when needed.

Put only stable reference material into `data/pdfs/`.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Copy `.env.example` to `.env` and add a Groq key if you want Groq.

```bash
streamlit run app.py
```

## Offline Hugging Face

The first time the local provider is used, Transformers downloads `Qwen/Qwen2.5-1.5B-Instruct`. After the model files are available locally, inference runs through PyTorch on the selected device and does not call a hosted LLM API.

## PDF indexing

The vector index is rebuilt automatically when the set of PDF files changes. ChromaDB persists the resulting embeddings under `data/chroma_db/`.

## Provider behavior

There is deliberately no Ollama, no expert-response LLM fallback, and no third provider. If the selected provider is unavailable, the app reports the setup problem instead of silently changing the AI model.
