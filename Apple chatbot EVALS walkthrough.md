# ⌚ Apple Watch RAG Chatbot — Evaluation Walkthrough

I have successfully designed and implemented a **3-dimension evaluation framework** for your Apple Watch RAG assistant. This elevates your project from a standard RAG prototype to a rigorously tested system — a major green flag for AI/Data science recruiters.

## What Was Built

All evaluation code is completely isolated in the `evals/` directory and can be run repeatably.

### 1. Router Unit Tests (`evals/test_router.py`)
- **Methodology**: 36 `pytest` assertions testing the keyword-based router offline.
- **Why it matters**: Tests the deterministic logic of the system (does a "price" question correctly trigger the web scraper?).
- **Results**: **36/36 passed.** We also identified and documented two interesting edge-case bugs (e.g., "What is the best iPhone" incorrectly triggers the `comparison` intent because "best" is a comparison keyword).

### 2. Retrieval Quality Tests (`evals/test_retrieval.py`)
- **Methodology**: 15 `pytest` assertions checking ChromaDB vector search against known queries.
- **Why it matters**: Proves you understand embedding distances. Tests assert that chunks returned have an acceptable cosine distance (`< 1.2`) and contain expected semantic keywords.
- **Results**: **15/15 passed.** 
- *Note: `SentenceTransformer` requires network access on its first initialization to check HuggingFace templates, even if the model weights are cached locally.*

### 3. LLM-as-Judge Response Quality (`evals/eval_responses.py`)
- **Methodology**: A Python script runs 15 curated "golden test" queries through the full RAG pipeline, then sends the response + retrieved context to a judge LLM (Groq `llama-3.3-70b-versatile`) to score on a 1-5 scale.
- **Criteria Evaluated**:
  1. **Groundedness**: Are facts backed by the context?
  2. **Faithfulness**: Did it avoid hallucinating prices or models?
  3. **Completeness**: Did it answer the user's question?
- **Results Generated**: A CSV of raw scores (`eval_results.csv`) and a beautiful markdown report ([`eval_report.md`](file:///Users/pratik/Downloads/v2%203/evals/report/eval_report.md)).

---

## 📊 Key Findings from the Eval Report

The LLM-as-judge evaluation highlighted exactly where the system is strong and where it fails:

**✅ The Strong Points (Static & Comparison)**
- The chatbot is excellent at static PDF questions (e.g., "What is ECG?"). It scored a perfect **5.0/5.0** on several of these.
- Comparisons ("SE vs Ultra for fitness") also score highly (**~4.3/5.0**).

**⚠️ The Weak Points (Live Price/Lineup Hallucinations)**
- **Live web queries performed poorly (Average 1.89/5.0).** 
- **The Issue**: When the Apple.com scraper fails to extract a verified price, the LLM sometimes ignores its strict system prompt and *hallucinates* realistic-sounding prices (e.g., inventing ₹46,900 for Series 11) or models that weren't in the context.
- **Why this is great for your CV**: Discovering this failure mode is the whole point of evaluations. You can now talk about how you identified hallucination risks in live-pricing scenarios.

---

## 🚀 Recommended Next Steps (How to use this for your CV)

You now have the data needed to improve the system. I recommend the following iterative improvements (which you can mention in interviews):

1. **Tighten the System Prompt**: Add a stricter constraint: *"If the LIVE APPLE.COM EVIDENCE section is empty, you MUST reply with 'I cannot verify the current price/lineup.' Do not attempt to guess."*
2. **Improve the Scraper**: The conservative regex in `apple_client.py` might be failing to extract prices from the current Apple India HTML layout, leaving the context empty. Updating the DOM parsing logic would fix the root cause.
3. **Run Evals Again**: After tweaking the prompt/scraper, re-run `python evals/eval_responses.py` to prove that Groundedness scores on `live` queries go up.

## To run the tests yourself:
```bash
# Run router tests
python -m pytest evals/test_router.py -v

# Run retrieval tests
python -m pytest evals/test_retrieval.py -v

# Run the full LLM judge evaluation (requires GROQ_API_KEY)
python evals/eval_responses.py
```
