"""
eval_responses.py — LLM-as-Judge Response Quality Evaluation

This script runs the full RAG pipeline on the golden test set and uses
Groq (LLaMA 3.3-70B) as a judge to score each response on 3 criteria:

  1. Groundedness    — Every factual claim is supported by the retrieved context
  2. Faithfulness    — No invented facts, prices, or specs (no hallucination)
  3. Completeness    — The response fully addresses what the user asked

Each criterion is scored 1–5. Scores are written to:
  evals/report/eval_results.csv

A formatted summary is printed to stdout.

Usage:
    python evals/eval_responses.py

Requirements:
    - GROQ_API_KEY set in .env or environment
    - PDFs indexed in data/chroma_db/ (for static queries)
    - Internet access for live queries

Runtime: ~3–5 minutes for 15 test cases (Groq API calls).
"""

import sys
import os
import csv
import json
import time
import textwrap
from datetime import datetime
from pathlib import Path

# ── Project root on path ──────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from config import get_groq_api_key, GROQ_MODEL
from assistant import AppleWatchAssistant
from evals.golden_test_set import GOLDEN_TEST_SET

# ── Output paths ──────────────────────────────────────────────────────────────
REPORT_DIR = ROOT / "evals" / "report"
REPORT_DIR.mkdir(parents=True, exist_ok=True)
CSV_PATH = REPORT_DIR / "eval_results.csv"
MD_PATH = REPORT_DIR / "eval_report.md"


# ── LLM-as-Judge prompt ───────────────────────────────────────────────────────

JUDGE_SYSTEM_PROMPT = """You are a strict but fair evaluator of AI assistant responses.
Your job is to score a response on three criteria and return ONLY a JSON object.

Criteria:
1. groundedness   (1–5): Every factual claim is supported by the retrieved context provided.
   5 = fully grounded, 1 = major unsupported claims.

2. faithfulness   (1–5): The response avoids inventing facts, prices, specs, or model names.
   5 = no hallucination at all, 1 = significant hallucination.

3. completeness   (1–5): The response fully answers what the user asked.
   5 = fully answers the question, 1 = avoids or misses the core question.

Return ONLY this JSON structure with NO extra text:
{
  "groundedness": <int 1-5>,
  "faithfulness": <int 1-5>,
  "completeness": <int 1-5>,
  "groundedness_reason": "<one sentence>",
  "faithfulness_reason": "<one sentence>",
  "completeness_reason": "<one sentence>"
}"""


def build_judge_prompt(query: str, response: str, context_summary: str) -> str:
    return f"""USER QUESTION:
{query}

RETRIEVED CONTEXT SUMMARY (what was available to the assistant):
{context_summary}

ASSISTANT RESPONSE TO EVALUATE:
{response}

Score the assistant response on the three criteria. Return ONLY the JSON object."""


# ── Groq judge call ───────────────────────────────────────────────────────────

def call_judge(query: str, response: str, context_summary: str) -> dict:
    """Call Groq API with the judge prompt. Returns parsed score dict or None."""
    from groq import Groq

    api_key = get_groq_api_key()
    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY not found. Set it in .env or as an environment variable."
        )

    client = Groq(api_key=api_key)
    judge_prompt = build_judge_prompt(query, response, context_summary)

    try:
        completion = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                {"role": "user", "content": judge_prompt},
            ],
            temperature=0.0,
            max_tokens=400,
        )
        raw = completion.choices[0].message.content.strip()

        # Strip markdown code fences if present
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        raw = raw.strip()

        return json.loads(raw)
    except json.JSONDecodeError as e:
        print(f"  ⚠️  Judge returned non-JSON: {e}")
        return None
    except Exception as e:
        print(f"  ⚠️  Judge call failed: {e}")
        return None


# ── Context summary builder ───────────────────────────────────────────────────

def summarize_context(result: dict) -> str:
    """Build a short context summary for the judge from assistant result."""
    parts = []
    route = result.get("route", {})
    parts.append(f"Route intent: {route.get('intent', 'unknown')}")
    parts.append(
        f"Sources used: PDF={route.get('needs_pdf', False)}, "
        f"Web={route.get('needs_web', False)}"
    )

    pdf_sources = result.get("pdf_sources", [])
    if pdf_sources:
        pages = [f"page {s.get('page', '?')}" for s in pdf_sources[:3]]
        parts.append(f"PDF chunks retrieved from: {', '.join(pages)}")

    web_sources = result.get("web_sources", [])
    if web_sources:
        parts.append(f"Web sources: {', '.join(web_sources[:2])}")
    else:
        parts.append("Web sources: none")

    return "\n".join(parts)


# ── Hallucination guard check ─────────────────────────────────────────────────

def check_should_not_contain(response: str, should_not_contain: list) -> list:
    """Return list of forbidden substrings found in the response."""
    return [
        s for s in should_not_contain
        if s.lower() in response.lower()
    ]


# ── Main eval loop ────────────────────────────────────────────────────────────

def run_evaluation():
    print("\n" + "=" * 65)
    print("  Apple Watch RAG Chatbot — Response Quality Evaluation")
    print("  LLM Judge: Groq llama-3.3-70b-versatile")
    print(f"  Test cases: {len(GOLDEN_TEST_SET)}")
    print(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 65 + "\n")

    # Initialize assistant (Groq provider)
    print("Initializing assistant...")
    assistant = AppleWatchAssistant(provider="groq")
    status = assistant.status()
    if not status["provider"].get("available", True):
        print(f"❌ Assistant not available: {status['provider'].get('error')}")
        sys.exit(1)
    print(f"✅ Assistant ready. PDF chunks: {status['retrieval']['documents']}\n")

    rows = []
    category_scores = {}

    for i, test in enumerate(GOLDEN_TEST_SET, 1):
        tid = test["id"]
        category = test["category"]
        query = test["query"]

        print(f"[{i:02d}/{len(GOLDEN_TEST_SET)}] {tid} ({category})")
        print(f"  Query: {query[:80]}{'...' if len(query) > 80 else ''}")

        # ── Step 1: Run the RAG pipeline ──────────────────────────────────────
        try:
            result = assistant.answer(query)
            response = result["response"]
        except Exception as e:
            print(f"  ❌ Assistant error: {e}\n")
            rows.append({
                "id": tid, "category": category, "query": query,
                "groundedness": 0, "faithfulness": 0, "completeness": 0,
                "avg_score": 0, "hallucination_violations": "",
                "groundedness_reason": f"ERROR: {e}",
                "faithfulness_reason": "", "completeness_reason": "",
                "response_snippet": "",
            })
            continue

        print(f"  Response: {response[:100].replace(chr(10), ' ')}...")

        # ── Step 2: Hallucination guard ───────────────────────────────────────
        violations = check_should_not_contain(
            response, test.get("should_not_contain", [])
        )
        if violations:
            print(f"  ⚠️  Hallucination guard triggered: {violations}")

        # ── Step 3: LLM judge ─────────────────────────────────────────────────
        context_summary = summarize_context(result)
        scores = call_judge(query, response, context_summary)

        if scores:
            g = scores.get("groundedness", 0)
            f = scores.get("faithfulness", 0)
            c = scores.get("completeness", 0)
            avg = round((g + f + c) / 3, 2)
            print(f"  Scores → Groundedness:{g} | Faithfulness:{f} | Completeness:{c} | Avg:{avg}")
        else:
            g, f, c, avg = 0, 0, 0, 0.0
            scores = {
                "groundedness_reason": "Judge call failed",
                "faithfulness_reason": "",
                "completeness_reason": "",
            }
            print("  Scores → judge call failed (0s recorded)")

        # Track by category
        if category not in category_scores:
            category_scores[category] = []
        if avg > 0:
            category_scores[category].append(avg)

        rows.append({
            "id": tid,
            "category": category,
            "query": query,
            "groundedness": g,
            "faithfulness": f,
            "completeness": c,
            "avg_score": avg,
            "hallucination_violations": "; ".join(violations),
            "groundedness_reason": scores.get("groundedness_reason", ""),
            "faithfulness_reason": scores.get("faithfulness_reason", ""),
            "completeness_reason": scores.get("completeness_reason", ""),
            "response_snippet": response[:300].replace("\n", " "),
        })

        print()
        # Small delay to stay within Groq rate limits
        time.sleep(1.5)

    # ── Write CSV ─────────────────────────────────────────────────────────────
    fieldnames = [
        "id", "category", "query",
        "groundedness", "faithfulness", "completeness", "avg_score",
        "hallucination_violations",
        "groundedness_reason", "faithfulness_reason", "completeness_reason",
        "response_snippet",
    ]
    with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n✅ Results saved to: {CSV_PATH}")

    # ── Print summary ─────────────────────────────────────────────────────────
    valid_rows = [r for r in rows if r["avg_score"] > 0]
    if valid_rows:
        overall_avg = round(
            sum(r["avg_score"] for r in valid_rows) / len(valid_rows), 2
        )
        avg_g = round(sum(r["groundedness"] for r in valid_rows) / len(valid_rows), 2)
        avg_f = round(sum(r["faithfulness"] for r in valid_rows) / len(valid_rows), 2)
        avg_c = round(sum(r["completeness"] for r in valid_rows) / len(valid_rows), 2)
    else:
        overall_avg = avg_g = avg_f = avg_c = 0.0

    print("\n" + "=" * 65)
    print("  EVALUATION SUMMARY")
    print("=" * 65)
    print(f"  Overall avg score  : {overall_avg}/5.0")
    print(f"  Groundedness avg   : {avg_g}/5.0")
    print(f"  Faithfulness avg   : {avg_f}/5.0")
    print(f"  Completeness avg   : {avg_c}/5.0")
    print()
    print("  By category:")
    for cat, scores_list in category_scores.items():
        cat_avg = round(sum(scores_list) / len(scores_list), 2) if scores_list else 0
        bar = "█" * int(cat_avg) + "░" * (5 - int(cat_avg))
        print(f"    {cat:<20} {bar}  {cat_avg}/5.0")

    hallucination_count = sum(
        1 for r in rows if r["hallucination_violations"]
    )
    print(f"\n  Hallucination guard violations: {hallucination_count}/{len(rows)}")

    # ── Generate markdown report ───────────────────────────────────────────────
    _write_markdown_report(rows, overall_avg, avg_g, avg_f, avg_c, category_scores)
    print(f"\n📄 Markdown report saved to: {MD_PATH}")
    print("\nDone! ✅\n")


# ── Markdown report generator ─────────────────────────────────────────────────

def _write_markdown_report(rows, overall_avg, avg_g, avg_f, avg_c, category_scores):
    valid_rows = [r for r in rows if r["avg_score"] > 0]
    hallucination_count = sum(1 for r in rows if r["hallucination_violations"])
    worst = sorted(valid_rows, key=lambda r: r["avg_score"])[:3]
    best = sorted(valid_rows, key=lambda r: r["avg_score"], reverse=True)[:3]

    lines = [
        "# Apple Watch RAG Chatbot — Evaluation Report",
        "",
        f"> Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  ",
        f"> Test cases: {len(rows)}  ",
        f"> LLM Judge: Groq `llama-3.3-70b-versatile`",
        "",
        "---",
        "",
        "## 1. Overall Scores",
        "",
        "| Criterion | Average Score (out of 5) | Interpretation |",
        "|---|---|---|",
        f"| **Groundedness** | {avg_g} | Factual claims backed by retrieved context |",
        f"| **Faithfulness** | {avg_f} | No hallucinated facts, prices, or model names |",
        f"| **Completeness** | {avg_c} | Question fully answered |",
        f"| **Overall Average** | **{overall_avg}** | — |",
        "",
        f"**Hallucination guard violations:** {hallucination_count} / {len(rows)} responses "
        f"contained a forbidden substring.",
        "",
        "---",
        "",
        "## 2. Scores by Category",
        "",
        "| Category | Avg Score | # Tests |",
        "|---|---|---|",
    ]

    for cat, scores_list in category_scores.items():
        cat_avg = round(sum(scores_list) / len(scores_list), 2) if scores_list else 0
        lines.append(f"| {cat} | {cat_avg} | {len(scores_list)} |")

    lines += [
        "",
        "---",
        "",
        "## 3. Per-Question Results",
        "",
        "| ID | Category | Query | G | F | C | Avg | Violations |",
        "|---|---|---|---|---|---|---|---|",
    ]

    for r in rows:
        short_q = r["query"][:55].replace("|", "\\|") + ("..." if len(r["query"]) > 55 else "")
        v = "⚠️" if r["hallucination_violations"] else "✅"
        lines.append(
            f"| {r['id']} | {r['category']} | {short_q} "
            f"| {r['groundedness']} | {r['faithfulness']} | {r['completeness']} "
            f"| {r['avg_score']} | {v} |"
        )

    lines += [
        "",
        "---",
        "",
        "## 4. Best Performing Responses",
        "",
    ]
    for r in best:
        lines += [
            f"### {r['id']} — {r['category']} (avg {r['avg_score']})",
            f"**Query:** {r['query']}",
            f"- Groundedness: {r['groundedness']} — {r['groundedness_reason']}",
            f"- Faithfulness: {r['faithfulness']} — {r['faithfulness_reason']}",
            f"- Completeness: {r['completeness']} — {r['completeness_reason']}",
            "",
        ]

    lines += [
        "---",
        "",
        "## 5. Lowest Scoring Responses (Improvement Targets)",
        "",
    ]
    for r in worst:
        lines += [
            f"### {r['id']} — {r['category']} (avg {r['avg_score']})",
            f"**Query:** {r['query']}",
            f"- Groundedness: {r['groundedness']} — {r['groundedness_reason']}",
            f"- Faithfulness: {r['faithfulness']} — {r['faithfulness_reason']}",
            f"- Completeness: {r['completeness']} — {r['completeness_reason']}",
        ]
        if r["hallucination_violations"]:
            lines.append(f"- ⚠️ Hallucination guard: `{r['hallucination_violations']}`")
        lines.append("")

    lines += [
        "---",
        "",
        "## 6. Identified Improvement Areas",
        "",
        "> This section should be filled in based on patterns observed in Section 5.",
        "",
        "| Issue | Affected Category | Suggested Fix |",
        "|---|---|---|",
        "| Low retrieval relevance for niche queries | static | Tune CHUNK_SIZE or add more PDFs |",
        "| Price hallucination risk | live | Tighten system prompt price rules |",
        "| Incomplete comparison answers | comparison | Add comparison-specific prompt section |",
        "| Off-topic queries not redirected firmly enough | adversarial | Add explicit out-of-scope handling |",
        "",
        "---",
        "",
        "## 7. Methodology",
        "",
        "- **Router tests**: 26 pytest unit tests on `router.py` (offline, no API key).",
        "- **Retrieval tests**: pytest tests asserting cosine distance < 1.2 and keyword presence in top-k chunks.",
        "- **Response quality**: LLM-as-judge using Groq `llama-3.3-70b-versatile` with a structured JSON rubric.",
        "- **Hallucination guard**: Rule-based check for forbidden substrings in responses.",
        "",
        "Raw scores: [`eval_results.csv`](./eval_results.csv)",
    ]

    MD_PATH.write_text("\n".join(lines), encoding="utf-8")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    run_evaluation()
