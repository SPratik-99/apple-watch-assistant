"""
Golden test set for Apple Watch RAG chatbot evaluation.

15 curated questions across 5 categories:
  - static   : questions answered purely from the PDF user guide
  - live     : questions requiring live Apple.com data
  - hybrid   : questions needing both PDF and web context
  - comparison: model vs model questions
  - adversarial: off-topic / ambiguous — the bot should handle gracefully

Each entry:
    query               : the user question
    expected_intent     : route intent the router MUST produce
    expected_needs_pdf  : bool
    expected_needs_web  : bool
    category            : for grouping in the report
    eval_criteria       : natural-language rubric hints for the LLM judge
    should_not_contain  : substrings that must NOT appear in the response
                          (hallucination guards)
"""

GOLDEN_TEST_SET = [

    # ── STATIC (PDF) ──────────────────────────────────────────────────────────

    {
        "id": "S1",
        "category": "static",
        "query": "How do I pair my Apple Watch with a new iPhone?",
        "expected_intent": "static",
        "expected_needs_pdf": True,
        "expected_needs_web": False,
        "eval_criteria": (
            "The answer should provide clear numbered steps for pairing. "
            "It should mention the Apple Watch app or proximity pairing. "
            "It must NOT invent steps not found in a standard user guide."
        ),
        "should_not_contain": ["₹", "price", "Series 11", "SE 3", "Ultra 3"],
    },
    {
        "id": "S2",
        "category": "static",
        "query": "What is ECG on Apple Watch and how do I use it?",
        "expected_intent": "static",
        "expected_needs_pdf": True,
        "expected_needs_web": False,
        "eval_criteria": (
            "The response should explain what ECG measures (electrical heart activity), "
            "how to take a reading (Digital Crown contact), and note any regional availability. "
            "It should NOT invent model numbers or prices."
        ),
        "should_not_contain": ["₹", "I don't know", "cannot help"],
    },
    {
        "id": "S3",
        "category": "static",
        "query": "My Apple Watch is not charging. What should I do?",
        "expected_intent": "static",
        "expected_needs_pdf": True,
        "expected_needs_web": False,
        "eval_criteria": (
            "The answer should provide troubleshooting steps: check cable alignment, "
            "clean contacts, restart the watch, try a different power adapter. "
            "Steps should be actionable and ordered logically."
        ),
        "should_not_contain": ["₹", "price", "buy"],
    },

    # ── LIVE (Web) ────────────────────────────────────────────────────────────

    {
        "id": "L1",
        "category": "live",
        "query": "What is the current price of Apple Watch Series 11 in India?",
        "expected_intent": "live",
        "expected_needs_pdf": False,
        "expected_needs_web": True,
        "eval_criteria": (
            "The response must only state a price if Apple.com evidence explicitly "
            "associates a ₹ price with Series 11. "
            "If the price cannot be reliably extracted, the bot must say so clearly "
            "rather than guessing. It must NOT invent a price."
        ),
        "should_not_contain": ["According to the PDF", "user guide"],
    },
    {
        "id": "L2",
        "category": "live",
        "query": "Which Apple Watch models are currently available?",
        "expected_intent": "live",
        "expected_needs_pdf": False,
        "expected_needs_web": True,
        "eval_criteria": (
            "The response should list current models based on Apple.com evidence. "
            "It should not mention discontinued models as current. "
            "It must not say 'I don't know' if Apple.com was reachable."
        ),
        "should_not_contain": ["Series 6", "Series 7", "Series 8", "Series 9"],
    },
    {
        "id": "L3",
        "category": "live",
        "query": "What is the most expensive Apple Watch right now?",
        "expected_intent": "live",
        "expected_needs_pdf": False,
        "expected_needs_web": True,
        "eval_criteria": (
            "The bot should identify the most expensive model (Ultra 3 is expected) "
            "IF the evidence explicitly supports it. "
            "If price data is ambiguous, it must say so honestly. "
            "Must not invent a rupee amount."
        ),
        "should_not_contain": ["According to the PDF"],
    },

    # ── HYBRID ────────────────────────────────────────────────────────────────

    {
        "id": "H1",
        "category": "hybrid",
        "query": "Does the Apple Watch Ultra support blood oxygen monitoring?",
        "expected_intent": "static",          # router may vary; acceptable as static or hybrid
        "expected_needs_pdf": True,
        "expected_needs_web": False,
        "eval_criteria": (
            "The response should confirm whether blood oxygen (SpO2) is available "
            "and mention any relevant setup steps from the user guide. "
            "It should not confuse blood oxygen with ECG."
        ),
        "should_not_contain": ["I don't know", "cannot confirm"],
    },
    {
        "id": "H2",
        "category": "hybrid",
        "query": "I want to buy an Apple Watch — what are my options and which is the best value?",
        "expected_intent": "comparison_or_recommendation",
        "expected_needs_pdf": True,
        "expected_needs_web": True,
        "eval_criteria": (
            "The response should give an actual recommendation, not just a list. "
            "It should contrast at least two models on price vs features. "
            "It should not say 'I cannot recommend' or just list specs."
        ),
        "should_not_contain": ["I cannot recommend", "I'm unable to"],
    },
    {
        "id": "H3",
        "category": "hybrid",
        "query": "Should I upgrade from Apple Watch Series 6 to the latest model?",
        "expected_intent": "comparison_or_recommendation",
        "expected_needs_pdf": True,
        "expected_needs_web": True,
        "eval_criteria": (
            "The response should highlight meaningful feature improvements over Series 6 "
            "(e.g., crash detection, always-on display improvements, newer health sensors). "
            "It should give an opinion, not sit on the fence."
        ),
        "should_not_contain": ["Series 6 is the latest", "Series 6 is current"],
    },

    # ── COMPARISON ────────────────────────────────────────────────────────────

    {
        "id": "C1",
        "category": "comparison",
        "query": "What is the difference between Apple Watch SE and Apple Watch Series 11?",
        "expected_intent": "comparison_or_recommendation",
        "expected_needs_pdf": True,
        "expected_needs_web": True,
        "eval_criteria": (
            "The response should clearly explain key differences: "
            "display (always-on on Series 11 vs absent on SE), health sensors, price tier. "
            "A table or bullet list is appropriate. "
            "It must not fabricate specs."
        ),
        "should_not_contain": ["I cannot compare", "I don't have information"],
    },
    {
        "id": "C2",
        "category": "comparison",
        "query": "Is Apple Watch Ultra 3 worth the extra money over Series 11?",
        "expected_intent": "comparison_or_recommendation",
        "expected_needs_pdf": True,
        "expected_needs_web": True,
        "eval_criteria": (
            "The response should explain Ultra-specific features (titanium, 60-hour battery, "
            "Action Button, dual-frequency GPS, depth gauge) and give a clear recommendation "
            "based on use-case (adventure/sport vs everyday). "
            "Must not answer with 'it depends' only — should give a directional opinion."
        ),
        "should_not_contain": ["I cannot say", "it depends on you"],
    },
    {
        "id": "C3",
        "category": "comparison",
        "query": "Compare Apple Watch SE vs Ultra for fitness tracking.",
        "expected_intent": "comparison_or_recommendation",
        "expected_needs_pdf": True,
        "expected_needs_web": True,
        "eval_criteria": (
            "The response should compare fitness-relevant features: "
            "GPS, heart rate, workout detection, battery life for long workouts. "
            "Should give a clear winner for fitness-focused users."
        ),
        "should_not_contain": ["I don't have enough information", "I cannot compare"],
    },

    # ── ADVERSARIAL ───────────────────────────────────────────────────────────

    {
        "id": "A1",
        "category": "adversarial",
        "query": "What is the best iPhone to buy in 2026?",
        "expected_intent": "general_conversation",
        "expected_needs_pdf": False,
        "expected_needs_web": False,
        "eval_criteria": (
            "The bot should politely acknowledge it is an Apple Watch assistant "
            "and decline to answer iPhone questions in detail. "
            "It must NOT provide detailed iPhone recommendations or specs. "
            "A brief, polite redirect is ideal."
        ),
        "should_not_contain": ["iPhone 15", "iPhone 16", "iPhone 17", "A17", "A18"],
    },
    {
        "id": "A2",
        "category": "adversarial",
        "query": "How do I fix my MacBook that won't turn on?",
        "expected_intent": "general_conversation",
        "expected_needs_pdf": False,
        "expected_needs_web": False,
        "eval_criteria": (
            "The bot should acknowledge it only covers Apple Watch topics "
            "and cannot help with MacBook issues. "
            "It must NOT invent MacBook troubleshooting steps."
        ),
        "should_not_contain": ["SMC reset", "NVRAM", "MacBook"],
    },
    {
        "id": "A3",
        "category": "adversarial",
        "query": "What is the exact battery percentage threshold at which Apple Watch enters Power Reserve mode?",
        "expected_intent": "static",
        "expected_needs_pdf": True,
        "expected_needs_web": False,
        "eval_criteria": (
            "If the PDF does not contain this specific threshold, the bot must honestly say "
            "it couldn't find that specific detail rather than guessing a number. "
            "If it does contain the value (typically ~10%), state it directly. "
            "Must not fabricate a percentage not found in evidence."
        ),
        "should_not_contain": ["I made up", "approximately 5%", "exactly 3%"],
    },
]
