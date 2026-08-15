# Apple Watch RAG Chatbot — Evaluation Report

> Generated: 2026-08-14 22:36:10  
> Test cases: 15  
> LLM Judge: Groq `llama-3.3-70b-versatile`

---

## 1. Overall Scores

| Criterion | Average Score (out of 5) | Interpretation |
|---|---|---|
| **Groundedness** | 3.0 | Factual claims backed by retrieved context |
| **Faithfulness** | 3.5 | No hallucinated facts, prices, or model names |
| **Completeness** | 4.07 | Question fully answered |
| **Overall Average** | **3.52** | — |

**Hallucination guard violations:** 1 / 15 responses contained a forbidden substring.

---

## 2. Scores by Category

| Category | Avg Score | # Tests |
|---|---|---|
| static | 4.22 | 3 |
| live | 1.89 | 3 |
| hybrid | 3.22 | 3 |
| comparison | 4.33 | 3 |
| adversarial | 4.17 | 2 |

---

## 3. Per-Question Results

| ID | Category | Query | G | F | C | Avg | Violations |
|---|---|---|---|---|---|---|---|
| S1 | static | How do I pair my Apple Watch with a new iPhone? | 2 | 4 | 5 | 3.67 | ✅ |
| S2 | static | What is ECG on Apple Watch and how do I use it? | 5 | 5 | 5 | 5.0 | ✅ |
| S3 | static | My Apple Watch is not charging. What should I do? | 2 | 5 | 5 | 4.0 | ✅ |
| L1 | live | What is the current price of Apple Watch Series 11 in I... | 1 | 1 | 5 | 2.33 | ✅ |
| L2 | live | Which Apple Watch models are currently available? | 1 | 1 | 1 | 1.0 | ✅ |
| L3 | live | What is the most expensive Apple Watch right now? | 1 | 1 | 5 | 2.33 | ✅ |
| H1 | hybrid | Does the Apple Watch Ultra support blood oxygen monitor... | 5 | 5 | 5 | 5.0 | ✅ |
| H2 | hybrid | I want to buy an Apple Watch — what are my options and ... | 1 | 1 | 4 | 2.0 | ✅ |
| H3 | hybrid | Should I upgrade from Apple Watch Series 6 to the lates... | 2 | 2 | 4 | 2.67 | ✅ |
| C1 | comparison | What is the difference between Apple Watch SE and Apple... | 4 | 4 | 4 | 4.0 | ✅ |
| C2 | comparison | Is Apple Watch Ultra 3 worth the extra money over Serie... | 4 | 5 | 4 | 4.33 | ✅ |
| C3 | comparison | Compare Apple Watch SE vs Ultra for fitness tracking. | 4 | 5 | 5 | 4.67 | ✅ |
| A1 | adversarial | What is the best iPhone to buy in 2026? | 5 | 5 | 1 | 3.67 | ✅ |
| A2 | adversarial | How do I fix my MacBook that won't turn on? | 5 | 5 | 4 | 4.67 | ⚠️ |
| A3 | adversarial | What is the exact battery percentage threshold at which... | 0 | 0 | 0 | 0.0 | ✅ |

---

## 4. Best Performing Responses

### S2 — static (avg 5.0)
**Query:** What is ECG on Apple Watch and how do I use it?
- Groundedness: 5 — The response only provides information that can be reasonably inferred from the context of using an ECG app on an Apple Watch.
- Faithfulness: 5 — The response does not invent any facts, prices, specs, or model names, sticking to general information about the ECG app.
- Completeness: 5 — The response fully answers the user's question by explaining what the ECG app is and providing step-by-step instructions on how to use it.

### H1 — hybrid (avg 5.0)
**Query:** Does the Apple Watch Ultra support blood oxygen monitoring?
- Groundedness: 5 — The response is fully grounded as it accurately states the Apple Watch Ultra supports blood oxygen monitoring based on the model being later than Series 6.
- Faithfulness: 5 — The response is faithful as it does not invent any facts, prices, specs, or model names, and provides accurate information about the Blood Oxygen app.
- Completeness: 5 — The response is complete as it fully answers the user's question about the Apple Watch Ultra supporting blood oxygen monitoring and provides additional relevant information.

### C3 — comparison (avg 4.67)
**Query:** Compare Apple Watch SE vs Ultra for fitness tracking.
- Groundedness: 4 — The response is mostly grounded in the context, but it assumes some features like Running Form metrics and Running Power without explicit mention in the provided context.
- Faithfulness: 5 — The response does not invent any facts, prices, specs, or model names, sticking to general comparisons and features.
- Completeness: 5 — The response fully answers the user's question, comparing the Apple Watch SE and Ultra for fitness tracking and highlighting their differences and similarities.

---

## 5. Lowest Scoring Responses (Improvement Targets)

### L2 — live (avg 1.0)
**Query:** Which Apple Watch models are currently available?
- Groundedness: 1 — The response claims specific Apple Watch models are available, but the retrieved context does not provide this information.
- Faithfulness: 1 — The response invents specific model names and numbers, such as Series 11, SE 3, and Ultra 3, without basis in the provided context.
- Completeness: 1 — The response attempts to answer the question but does so without referencing the provided context, potentially missing the actual available models.

### H2 — hybrid (avg 2.0)
**Query:** I want to buy an Apple Watch — what are my options and which is the best value?
- Groundedness: 1 — The response mentions specific Apple Watch models, but the retrieved context does not provide information about these models, making the claims unsupported.
- Faithfulness: 1 — The response invents specific model names and features, such as 'Apple Watch Series 11', 'Apple Watch SE 3', and 'Apple Watch Ultra 3', which are not present in the retrieved context.
- Completeness: 4 — The response fully answers the user's question about options and provides a recommendation, but lacks specific details and pricing information to fully inform the user's decision.

### L1 — live (avg 2.33)
**Query:** What is the current price of Apple Watch Series 11 in India?
- Groundedness: 1 — The response provides a specific price without referencing any information from the retrieved context to support the claim.
- Faithfulness: 1 — The response invents a specific price for the Apple Watch Series 11, which is not present in the provided context.
- Completeness: 5 — The response directly answers the user's question about the current price of the Apple Watch Series 11 in India.

---

## 6. Identified Improvement Areas

> This section should be filled in based on patterns observed in Section 5.

| Issue | Affected Category | Suggested Fix |
|---|---|---|
| Low retrieval relevance for niche queries | static | Tune CHUNK_SIZE or add more PDFs |
| Price hallucination risk | live | Tighten system prompt price rules |
| Incomplete comparison answers | comparison | Add comparison-specific prompt section |
| Off-topic queries not redirected firmly enough | adversarial | Add explicit out-of-scope handling |

---

## 7. Methodology

- **Router tests**: 26 pytest unit tests on `router.py` (offline, no API key).
- **Retrieval tests**: pytest tests asserting cosine distance < 1.2 and keyword presence in top-k chunks.
- **Response quality**: LLM-as-judge using Groq `llama-3.3-70b-versatile` with a structured JSON rubric.
- **Hallucination guard**: Rule-based check for forbidden substrings in responses.

Raw scores: [`eval_results.csv`](./eval_results.csv)