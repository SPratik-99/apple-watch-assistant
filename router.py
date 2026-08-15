from dataclasses import dataclass
import re


@dataclass
class Route:
    needs_pdf: bool
    needs_web: bool
    intent: str


# ============================================================
# LIVE / DYNAMIC INFORMATION
# ============================================================

LIVE_TERMS = [
    "current",
    "currently",
    "latest",
    "newest",
    "today",
    "now",
    "2026",

    # Pricing
    "price",
    "prices",
    "pricing",
    "cost",
    "costs",
    "mrp",
    "how much",
    "expensive",
    "cheapest",
    "cheap",
    "most expensive",
    "least expensive",
    "highest price",
    "lowest price",
    "highest priced",
    "lowest priced",

    # Buying / availability
    "buy",
    "buying",
    "available",
    "availability",
    "sold",
    "selling",
    "store",
    "apple store",
    "apple website",
    "apple india",
    "discount",
    "offer",
    "deal",

    # Product lineup
    "new model",
    "new models",
    "current model",
    "current models",
    "which model",
    "which models",
    "latest model",
    "latest models",
    "newest model",
    "newest models",
    "lineup",
]


# ============================================================
# STATIC PDF KNOWLEDGE
# ============================================================

STATIC_TERMS = [
    "how do i",
    "how to",
    "manual",
    "guide",
    "troubleshoot",
    "troubleshooting",
    "not working",
    "won't",
    "wont",
    "setup",
    "set up",
    "pair",
    "pairing",
    "unpair",
    "restart",
    "reset",
    "charge",
    "charging",
    "battery",

    # Features
    "fall detection",
    "crash detection",
    "ecg",
    "blood oxygen",
    "heart rate",
    "sleep tracking",
    "sleep",
    "workout",
    "fitness",
    "digital crown",
    "action button",
    "always-on",
    "always on",
    "siri",
    "apple pay",
    "notifications",
    "water resistance",
    "accessibility",

    # watchOS / settings
    "watchos",
    "settings",
    "software update",
    "update",

    # General knowledge questions
    "what does",
    "how does",
    "what is",
    "what are",
    "feature",
    "features",
    "specification",
    "specifications",
    "specs",
]


# ============================================================
# COMPARISON / RECOMMENDATION
# ============================================================

COMPARISON_TERMS = [
    "compare",
    "comparison",
    "vs",
    "versus",
    "difference",
    "differences",
    "better",
    "worth",
    "should i",
    "recommend",
    "recommendation",
    "best",
    "which one",
]


def _last_user_message(history):
    """Get the most recent user message."""
    if not history:
        return ""

    for item in reversed(history):
        if item.get("role") == "user":
            return item.get("content", "").lower()

    return ""


def route_query(query: str, history=None) -> Route:
    """
    Decide which information sources are required.

    PDF:
        Stable Apple Watch knowledge.

    Web:
        Current Apple information.

    Both:
        Questions requiring stable knowledge + current information.
    """

    q = query.lower().strip()

    # --------------------------------------------------------
    # Detect categories
    # --------------------------------------------------------

    has_live = any(term in q for term in LIVE_TERMS)
    has_static = any(term in q for term in STATIC_TERMS)

    is_comparison = any(term in q for term in COMPARISON_TERMS)

    # Strong live-price questions
    is_price_question = any(
        phrase in q
        for phrase in [
            "price",
            "prices",
            "pricing",
            "cost",
            "costs",
            "how much",
            "most expensive",
            "least expensive",
            "cheapest",
            "highest price",
            "lowest price",
        ]
    )

    # Current-lineup questions
    is_current_lineup_question = any(
        phrase in q
        for phrase in [
            "latest watch",
            "latest apple watch",
            "latest model",
            "latest models",
            "newest watch",
            "newest apple watch",
            "current watch",
            "current apple watch",
            "current models",
            "current lineup",
            "latest lineup",
            "which apple watch is available",
        ]
    )

    # --------------------------------------------------------
    # Conversation follow-up handling
    # --------------------------------------------------------

    previous = _last_user_message(history)

    if previous and len(q.split()) <= 8:

        previous_was_watch_topic = bool(
            re.search(
                r"\b(apple watch|watch|series|ultra|se)\b",
                previous,
            )
        )

        followup_live_terms = [
            "price",
            "prices",
            "cost",
            "how much",
            "current",
            "latest",
            "available",
            "availability",
            "expensive",
            "cheapest",
        ]

        if (
            previous_was_watch_topic
            and any(term in q for term in followup_live_terms)
        ):
            has_live = True

    # --------------------------------------------------------
    # Recommendations and comparisons
    # --------------------------------------------------------

    if is_comparison:
        return Route(
            needs_pdf=True,
            needs_web=True,
            intent="comparison_or_recommendation",
        )

    # --------------------------------------------------------
    # Explicit current / price questions
    # --------------------------------------------------------

    if is_price_question or is_current_lineup_question:
        return Route(
            needs_pdf=False,
            needs_web=True,
            intent="live",
        )

    # --------------------------------------------------------
    # Hybrid question
    # --------------------------------------------------------

    if has_live and has_static:
        return Route(
            needs_pdf=True,
            needs_web=True,
            intent="hybrid",
        )

    # --------------------------------------------------------
    # Live-only
    # --------------------------------------------------------

    if has_live:
        return Route(
            needs_pdf=False,
            needs_web=True,
            intent="live",
        )

    # --------------------------------------------------------
    # Static-only
    # --------------------------------------------------------

    if has_static:
        return Route(
            needs_pdf=True,
            needs_web=False,
            intent="static",
        )

    # --------------------------------------------------------
    # Apple Watch general question
    # --------------------------------------------------------

    if re.search(
        r"\b(apple watch|watch|series|ultra|se)\b",
        q,
    ):
        return Route(
            needs_pdf=True,
            needs_web=False,
            intent="general",
        )

    # --------------------------------------------------------
    # General conversation
    # --------------------------------------------------------

    return Route(
        needs_pdf=False,
        needs_web=False,
        intent="general_conversation",
    )