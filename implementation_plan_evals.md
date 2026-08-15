# Apple Watch RAG Chatbot — Evaluation Plan

## Project Architecture (Quick Recap)

| Layer | Tech |
|---|---|
| **UI** | Streamlit |
| **Router** | Keyword-based (`router.py`) → `needs_pdf`, `needs_web`, `intent` |
| **PDF Retriever** | ChromaDB + `all-MiniLM-L6-v2` embeddings (RETRIEVAL_K = 4) |
| **Web Scraper** | `web/apple_client.py` → Apple India page |
| **LLM** | Groq (`llama-3.3-70b-versatile`) or local HuggingFace (Qwen2.5-1.5B) |
| **Prompt** | Detailed 12-section system prompt + evidence-injected user message |

---

## Why Evals Matter for Your CV

> [!IMPORTANT]
> As a fresher applying for AI/Data roles, **running evaluations and writing about them is a bigger signal than the chatbot itself.** Most freshers build the system and stop. You'll stand out because you can *measure* what you built. Recruiters and hiring managers for AI roles look for: (1) metric-driven thinking, (2) awareness of failure modes, (3) ability to iterate from data.

---

## Evaluation Strategy: What to Evaluate (and What to Skip)

We focus on **3 core evaluation dimensions** — each directly mappable to what a RAG system can fail at, and each explainable in an interview.

### Dimension 1 — Router Accuracy (Unit Tests, fully automated)
*"Does the keyword router send the query to the right source?"*

- This is **pure Python logic** — 100% testable without calling any LLM.
- A bad router = wasted web calls + wrong context fed to LLM.
- Recruiters love this: it shows you understood the system end-to-end.

### Dimension 2 — Retrieval Quality (Semi-automated)
*"Does the vector search surface relevant PDF chunks for a given query?"*

- We measure **chunk relevance** using cosine distance scores already returned by ChromaDB.
- We also write a small set of `(query → expected_keyword_in_chunk)` test cases.
- No LLM needed; pure embedding-based.

### Dimension 3 — Response Quality / Human-Model Critique Alignment (LLM-as-judge)
*"Is the final answer factually grounded, helpful, and appropriately uncertain?"*

- We create a **golden test set** of ~10 questions with expected answer characteristics.
- We use the LLM itself (Groq) as a judge to score responses on 3 rubric criteria.
- We log all scores to a CSV so you have a report artifact.

> [!NOTE]
> We deliberately skip: BLEU/ROUGE (not meaningful for RAG chat), full integration tests needing live internet, and latency benchmarking. These add complexity without proportionate CV value.

---

## Rubric Criteria (Dimension 3)

| # | Criterion | What it checks | Score (1–5) |
|---|---|---|---|
| 1 | **Groundedness** | Is every factual claim supported by retrieved context? | 1–5 |
| 2 | **Faithfulness / No Hallucination** | Does the response avoid inventing prices, models, or specs? | 1–5 |
| 3 | **Answer Completeness** | Does the response fully address what the user asked? | 1–5 |

---

## Proposed Files

### New `evals/` directory

```
evals/
├── __init__.py
├── test_router.py          # pytest unit tests for router.py
├── test_retrieval.py       # pytest tests for retrieval quality
├── eval_responses.py       # LLM-as-judge response evaluation runner
├── golden_test_set.py      # Curated Q&A test cases (the ground truth)
└── report/
    └── eval_report.md      # Generated markdown evaluation report
```

---

## Proposed Changes

### `evals/golden_test_set.py` [NEW]
A curated set of test questions organized by category:
- **Static (PDF)** questions: pairing, battery, ECG, fall detection
- **Live (Web)** questions: current lineup, pricing
- **Hybrid** questions: "Should I upgrade?" type
- **Adversarial** questions: off-topic, ambiguous — expect graceful refusal

Each entry: `{ query, expected_intent, expected_source, expected_keywords_in_answer, should_not_contain }`.

---

### `evals/test_router.py` [NEW]
~20 pytest unit tests covering:
- Standard static queries → `needs_pdf=True, needs_web=False`
- Standard live queries → `needs_pdf=False, needs_web=True`
- Comparison queries → both sources
- Follow-up handling (short query + history context)
- Edge cases (empty query, mixed signals)

These run **instantly**, no API key needed, and give you a green pass/fail badge.

---

### `evals/test_retrieval.py` [NEW]
~8 pytest tests covering:
- Known queries return `k=4` chunks
- Cosine distance stays below a threshold (< 1.0 for all-MiniLM)
- Specific keyword present in at least one returned chunk (e.g., "ECG" query → chunk contains "ECG")
- Empty / nonsense query returns results gracefully (no crash)

---

### `evals/eval_responses.py` [NEW]
A standalone script (not pytest) that:
1. Loads `golden_test_set.py` entries
2. For each entry, calls `assistant.answer(query)`
3. Sends both the answer and context to Groq with an **LLM-as-judge prompt**
4. Collects scores for Groundedness, Faithfulness, Completeness
5. Writes results to `evals/report/eval_results.csv`
6. Prints a summary table to stdout

---

### `evals/report/eval_report.md` [NEW]
A final markdown report artifact containing:
- Summary table of all 3 dimensions with pass rates
- Per-question scores from LLM judge
- Failure analysis: which questions scored lowest and why
- Identified improvement areas

> [!IMPORTANT]
> **This report is the most important artifact for your CV.** It demonstrates that you ran rigorous evals, understood failures, and could propose improvements — the exact thinking loop that real ML engineers do.

---

## Open Questions

> [!NOTE]
> No blocking questions — the plan is ready to execute. One optional decision below.

1. **Groq API key availability**: The `eval_responses.py` script needs your Groq API key to run the LLM-as-judge step. Do you want a fallback that scores manually instead? (Recommended: keep Groq for automation.)
2. **Number of golden test cases**: I'm planning ~12 questions. More = better report, but more API calls. Shall I keep it at 12 or expand to ~20?

---

## Verification Plan

### Automated Tests
```bash
cd "v2 3"
pytest evals/test_router.py -v           # router unit tests
pytest evals/test_retrieval.py -v        # retrieval quality tests
python evals/eval_responses.py           # LLM-as-judge eval runner
```

### Manual Verification
- Review `evals/report/eval_results.csv` for scores
- Read generated `evals/report/eval_report.md`

---

## What This Looks Like on Your CV

> "Designed and implemented a 3-dimension evaluation framework for a RAG-based Apple Watch assistant:
> automated router unit tests (pytest), retrieval quality tests using embedding distance thresholds,
> and LLM-as-judge response scoring (Groundedness, Faithfulness, Completeness) with a final
> evaluation report identifying failure modes and improvement paths."

That's a strong, specific bullet that demonstrates real ML engineering discipline.
