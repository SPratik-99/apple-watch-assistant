import logging
import re
import time
from typing import Dict, List, Optional, Tuple

import requests
from bs4 import BeautifulSoup, Tag

logger = logging.getLogger(__name__)


class AppleClient:
    """
    Retrieves current information from Apple India.

    IMPORTANT:
    This class is responsible for FACT COLLECTION.
    It does not generate natural-language answers.

    The LLM receives only structured/relevant evidence.
    """

    URLS = {
        "watch": "https://www.apple.com/in/watch/",
        "buy": "https://www.apple.com/in/shop/buy-watch",
        "compare": "https://www.apple.com/in/watch/compare/",
        "support": "https://support.apple.com/en-in/watch",
        "watchos": "https://www.apple.com/in/watchos/",
    }

    # Current lineup names we expect to find on Apple India.
    # These are NOT used as a source of truth by themselves.
    # Apple.com must still contain them.
    KNOWN_MODELS = [
        "Apple Watch Series 11",
        "Apple Watch SE 3",
        "Apple Watch Ultra 3",
    ]

    def __init__(self, ttl_seconds: int = 600):
        self.session = requests.Session()

        self.session.headers.update({
            "User-Agent": (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/151 Safari/537.36"
            ),
            "Accept": (
                "text/html,application/xhtml+xml,"
                "application/xml;q=0.9,image/webp,*/*;q=0.8"
            ),
            "Accept-Language": "en-IN,en;q=0.9",
        })

        self.ttl_seconds = ttl_seconds

        # key -> (timestamp, BeautifulSoup)
        self.cache: Dict[str, Tuple[float, BeautifulSoup]] = {}

    # ========================================================
    # BASIC HTTP
    # ========================================================

    def _fetch(self, key: str) -> Optional[BeautifulSoup]:
        now = time.time()

        cached = self.cache.get(key)

        if cached:
            timestamp, soup = cached

            if now - timestamp < self.ttl_seconds:
                return soup

        try:
            response = self.session.get(
                self.URLS[key],
                timeout=15,
            )

            response.raise_for_status()

            soup = BeautifulSoup(
                response.text,
                "html.parser",
            )

            self.cache[key] = (now, soup)

            return soup

        except Exception as exc:
            logger.warning(
                "Apple fetch failed for %s: %s",
                self.URLS[key],
                exc,
            )

            return None

    # ========================================================
    # TEXT HELPERS
    # ========================================================

    @staticmethod
    def _clean(text: str) -> str:
        return re.sub(
            r"\s+",
            " ",
            text or "",
        ).strip()

    @staticmethod
    def _extract_prices(text: str) -> List[str]:
        """
        Extract INR prices.

        Example:
            ₹46,900
            ₹25,900
        """

        matches = re.findall(
            r"₹\s*[0-9][0-9,]*(?:\.\d+)?",
            text or "",
        )

        # Deduplicate while preserving order.
        return list(dict.fromkeys(matches))

    @staticmethod
    def _price_value(price: str) -> int:
        digits = re.sub(
            r"\D",
            "",
            price,
        )

        return int(digits) if digits else 0

    # ========================================================
    # MODEL DETECTION
    # ========================================================

    @staticmethod
    def _canonical_model(text: str) -> Optional[str]:
        """
        Convert model mentions into canonical names.
        """

        t = text.lower()

        # Series 11
        if re.search(
            r"\bapple\s+watch\s+series\s*11\b",
            t,
        ):
            return "Apple Watch Series 11"

        if re.search(
            r"\bseries\s*11\b",
            t,
        ):
            return "Apple Watch Series 11"

        # SE 3
        if re.search(
            r"\bapple\s+watch\s+se\s*3\b",
            t,
        ):
            return "Apple Watch SE 3"

        if re.search(
            r"\bse\s*3\b",
            t,
        ):
            return "Apple Watch SE 3"

        # Ultra 3
        if re.search(
            r"\bapple\s+watch\s+ultra\s*3\b",
            t,
        ):
            return "Apple Watch Ultra 3"

        if re.search(
            r"\bultra\s*3\b",
            t,
        ):
            return "Apple Watch Ultra 3"

        return None

    # ========================================================
    # MODEL / PRICE EXTRACTION
    # ========================================================

    def _find_model_price_pairs(
        self,
        soup: BeautifulSoup,
    ) -> List[Dict]:
        """
        Search Apple Store HTML for model/price relationships.

        We only accept a relationship when the model and price
        appear in the SAME relatively small DOM block.

        This prevents unrelated page prices from being assigned
        to Apple Watch models.
        """

        results = []

        # Search all elements containing model names.
        text_nodes = soup.find_all(
            string=re.compile(
                r"(Series\s*11|SE\s*3|Ultra\s*3)",
                re.I,
            )
        )

        for node in text_nodes:

            if not isinstance(node, str):
                continue

            model = self._canonical_model(node)

            if not model:
                continue

            parent = node.parent # type: ignore

            # Search progressively larger containers.
            for _ in range(4):

                if not isinstance(parent, Tag):
                    break

                text = self._clean(
                    parent.get_text(
                        " ",
                        strip=True,
                    )
                )

                # Avoid giant page sections.
                if len(text) > 1200:
                    parent = parent.parent
                    continue

                prices = self._extract_prices(text)

                if prices:

                    # We only accept a price if there is exactly
                    # one clearly relevant price in the block.
                    #
                    # If multiple unrelated prices are present,
                    # we refuse to guess.
                    if len(prices) == 1:

                        results.append({
                            "model": model,
                            "price": prices[0],
                            "price_value": self._price_value(
                                prices[0]
                            ),
                            "evidence": text,
                        })

                    break

                parent = parent.parent

        # Deduplicate model entries.
        unique = {}

        for item in results:
            model = item["model"]

            if model not in unique:
                unique[model] = item

        return list(unique.values())

    # ========================================================
    # CURRENT LINEUP
    # ========================================================

    def _get_current_lineup(
        self,
        soup: BeautifulSoup,
    ) -> List[str]:

        text = self._clean(
            soup.get_text(
                " ",
                strip=True,
            )
        )

        lineup = []

        for model in self.KNOWN_MODELS:

            if re.search(
                re.escape(model),
                text,
                re.I,
            ):
                lineup.append(model)

        return lineup

    # ========================================================
    # PRICE EVIDENCE
    # ========================================================

    def _build_price_evidence(
        self,
        soup: BeautifulSoup,
        query: str,
    ) -> str:

        pairs = self._find_model_price_pairs(soup)

        q = query.lower()

        # ----------------------------------------------------
        # User asked about a specific model
        # ----------------------------------------------------

        requested_models = []

        if re.search(
            r"\bseries\s*11\b",
            q,
        ):
            requested_models.append(
                "Apple Watch Series 11"
            )

        if re.search(
            r"\bse(?:\s*3)?\b",
            q,
        ):
            requested_models.append(
                "Apple Watch SE 3"
            )

        if re.search(
            r"\bultra(?:\s*3)?\b",
            q,
        ):
            requested_models.append(
                "Apple Watch Ultra 3"
            )

        if requested_models:

            matched = [
                item
                for item in pairs
                if item["model"] in requested_models
            ]

            if matched:

                return (
                    "VERIFIED APPLE INDIA PRICE DATA:\n"
                    + "\n".join(
                        f"- {item['model']}: "
                        f"{item['price']}"
                        for item in matched
                    )
                )

            return (
                "APPLE INDIA PRICE STATUS:\n"
                f"Apple India was reached, but a reliable "
                f"model-specific price for "
                f"{', '.join(requested_models)} "
                f"could not be extracted."
            )

        # ----------------------------------------------------
        # User asks for most expensive / cheapest
        # ----------------------------------------------------

        if pairs and (
            "most expensive" in q
            or "highest price" in q
            or "highest priced" in q
        ):

            if len(pairs) >= 2:

                most_expensive = max(
                    pairs,
                    key=lambda x: x["price_value"],
                )

                return (
                    "VERIFIED APPLE INDIA PRICE DATA:\n"
                    f"- Most expensive verified model: "
                    f"{most_expensive['model']} "
                    f"({most_expensive['price']})"
                )

        if pairs and (
            "cheapest" in q
            or "least expensive" in q
            or "lowest price" in q
            or "lowest priced" in q
        ):

            if len(pairs) >= 2:

                cheapest = min(
                    pairs,
                    key=lambda x: x["price_value"],
                )

                return (
                    "VERIFIED APPLE INDIA PRICE DATA:\n"
                    f"- Cheapest verified model: "
                    f"{cheapest['model']} "
                    f"({cheapest['price']})"
                )

        # ----------------------------------------------------
        # User asks for all current prices
        # ----------------------------------------------------

        if pairs:

            return (
                "VERIFIED APPLE INDIA PRICE DATA:\n"
                + "\n".join(
                    f"- {item['model']}: "
                    f"{item['price']}"
                    for item in pairs
                )
            )

        # ----------------------------------------------------
        # IMPORTANT:
        # Never return a raw list of prices.
        # ----------------------------------------------------

        return (
            "APPLE INDIA PRICE STATUS:\n"
            "Apple India was reached, but the current page "
            "did not expose reliable model-specific prices "
            "that could be safely associated with Apple Watch models."
        )

    # ========================================================
    # GENERAL LIVE EVIDENCE
    # ========================================================

    def _build_lineup_evidence(
        self,
        soup: BeautifulSoup,
    ) -> str:

        lineup = self._get_current_lineup(soup)

        if not lineup:
            return (
                "Apple India was reached, but the current "
                "Apple Watch lineup could not be reliably extracted."
            )

        return (
            "VERIFIED APPLE INDIA CURRENT LINEUP:\n"
            + "\n".join(
                f"- {model}"
                for model in lineup
            )
        )

    # ========================================================
    # FETCH
    # ========================================================

    def fetch(self, query: str) -> Dict:

        q = query.lower().strip()

        wants_price = any(
            term in q
            for term in [
                "price",
                "prices",
                "pricing",
                "cost",
                "costs",
                "how much",
                "expensive",
                "cheapest",
                "highest price",
                "lowest price",
            ]
        )

        wants_current = any(
            term in q
            for term in [
                "current",
                "currently",
                "latest",
                "newest",
                "new model",
                "new models",
                "current model",
                "current models",
                "current lineup",
                "latest lineup",
                "available",
                "availability",
                "which model",
                "which models",
            ]
        )

        wants_compare = any(
            term in q
            for term in [
                "compare",
                "comparison",
                "difference",
                " vs ",
                "versus",
            ]
        )

        wants_support = any(
            term in q
            for term in [
                "support",
                "troubleshoot",
                "not working",
            ]
        )

        wants_watchos = (
            "watchos" in q
            or "software update" in q
        )

        pages = []

        # Current/price information comes primarily
        # from Apple's store and current watch page.
        if wants_price:
            pages.append("buy")

        if wants_current:
            pages.append("watch")

        if wants_compare:
            pages.append("compare")

        if wants_support:
            pages.append("support")

        if wants_watchos:
            pages.append("watchos")

        if not pages:
            pages.append("watch")

        evidence = []
        sources = []

        for page_key in dict.fromkeys(pages):

            soup = self._fetch(page_key)

            if not soup:
                continue

            # --------------------------------------------
            # Apple Store
            # --------------------------------------------

            if page_key == "buy":

                summary = self._build_price_evidence(
                    soup,
                    q,
                )

            # --------------------------------------------
            # Current Apple Watch page
            # --------------------------------------------

            elif page_key == "watch":

                summary = self._build_lineup_evidence(
                    soup,
                )

            # --------------------------------------------
            # Other pages
            # --------------------------------------------

            else:

                page_text = self._clean(
                    soup.get_text(
                        " ",
                        strip=True,
                    )
                )

                summary = page_text[:3000]

            if summary:

                evidence.append(summary)
                sources.append(
                    self.URLS[page_key]
                )

        return {
            "available": bool(evidence),
            "evidence": "\n\n".join(evidence),
            "sources": sources,
        }

    # ========================================================
    # HEALTH CHECK
    # ========================================================

    def health(self) -> bool:

        try:

            response = self.session.get(
                self.URLS["watch"],
                timeout=10,
            )

            return response.ok

        except Exception:

            return False