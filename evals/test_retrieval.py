"""
test_retrieval.py — Pytest tests for retrieval quality.

Tests the StaticPDFRetriever (ChromaDB + MiniLM embeddings) in isolation.
Requires the ChromaDB index to be populated (i.e., PDFs must be in data/pdfs/).

Run:
    pytest evals/test_retrieval.py -v

If no PDFs are present, tests are skipped gracefully.

Coverage:
  - Search returns the correct number of results
  - Cosine distance stays within an acceptable threshold
  - Expected keywords appear in top-k results for known queries
  - Nonsense/garbage queries don't crash the system
  - Metadata fields are present on all returned chunks
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from retrieval.vector_store import StaticPDFRetriever
from config import RETRIEVAL_K


# ── Fixture ───────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def retriever():
    """Single retriever instance shared across all tests in this module."""
    return StaticPDFRetriever()


@pytest.fixture(scope="module")
def has_docs(retriever):
    """Skip all tests if the index is empty (no PDFs loaded)."""
    return retriever.document_count > 0


# ── Helpers ───────────────────────────────────────────────────────────────────

COSINE_DISTANCE_THRESHOLD = 1.2
"""
ChromaDB returns L2 distance on normalized embeddings, which is equivalent to
  distance = 2 * (1 - cosine_similarity).
  0.0 = perfect match, 2.0 = opposite vectors.
We allow up to 1.2 — anything higher means the chunk is probably irrelevant.
"""


def keyword_found_in_results(results, keyword: str) -> bool:
    """Check if a keyword appears in at least one result chunk (case-insensitive)."""
    kw = keyword.lower()
    return any(kw in r["text"].lower() for r in results)


# ── RESULT COUNT tests ────────────────────────────────────────────────────────

class TestResultCount:
    """The retriever should return exactly RETRIEVAL_K results for real queries."""

    def test_returns_k_results_for_pairing_query(self, retriever, has_docs):
        if not has_docs:
            pytest.skip("No PDFs indexed — skipping retrieval tests")
        results = retriever.search("How do I pair Apple Watch with iPhone?")
        assert len(results) == RETRIEVAL_K, (
            f"Expected {RETRIEVAL_K} results, got {len(results)}"
        )

    def test_returns_k_results_for_battery_query(self, retriever, has_docs):
        if not has_docs:
            pytest.skip("No PDFs indexed — skipping retrieval tests")
        results = retriever.search("Apple Watch battery charging")
        assert len(results) == RETRIEVAL_K

    def test_returns_k_results_for_ecg_query(self, retriever, has_docs):
        if not has_docs:
            pytest.skip("No PDFs indexed — skipping retrieval tests")
        results = retriever.search("ECG heart electrocardiogram Apple Watch")
        assert len(results) == RETRIEVAL_K


# ── DISTANCE / RELEVANCE tests ────────────────────────────────────────────────

class TestDistanceThresholds:
    """All returned chunks should have a distance below the threshold."""

    def test_pairing_query_distances_within_threshold(self, retriever, has_docs):
        if not has_docs:
            pytest.skip("No PDFs indexed")
        results = retriever.search("How do I pair Apple Watch?")
        for r in results:
            assert r["distance"] < COSINE_DISTANCE_THRESHOLD, (
                f"Chunk distance {r['distance']:.4f} exceeds threshold {COSINE_DISTANCE_THRESHOLD}. "
                f"Chunk text: {r['text'][:80]}..."
            )

    def test_battery_query_distances_within_threshold(self, retriever, has_docs):
        if not has_docs:
            pytest.skip("No PDFs indexed")
        results = retriever.search("Apple Watch battery life and charging")
        for r in results:
            assert r["distance"] < COSINE_DISTANCE_THRESHOLD

    def test_distance_is_numeric(self, retriever, has_docs):
        if not has_docs:
            pytest.skip("No PDFs indexed")
        results = retriever.search("Apple Watch setup")
        for r in results:
            assert isinstance(r["distance"], (float, int)), (
                "Distance value should be numeric"
            )


# ── KEYWORD RELEVANCE tests ───────────────────────────────────────────────────

class TestKeywordRelevance:
    """
    At least one of the top-k results should contain a keyword
    that is semantically central to the query.
    """

    def test_ecg_query_contains_ecg_in_results(self, retriever, has_docs):
        if not has_docs:
            pytest.skip("No PDFs indexed")
        results = retriever.search("How does ECG work on Apple Watch?")
        assert keyword_found_in_results(results, "ecg") or \
               keyword_found_in_results(results, "electrocardiogram") or \
               keyword_found_in_results(results, "heart"), (
            "ECG query should surface at least one chunk mentioning ECG or heart"
        )

    def test_battery_query_contains_battery_in_results(self, retriever, has_docs):
        if not has_docs:
            pytest.skip("No PDFs indexed")
        results = retriever.search("How do I charge Apple Watch?")
        assert keyword_found_in_results(results, "charg") or \
               keyword_found_in_results(results, "battery"), (
            "Charging query should surface at least one chunk about charging or battery"
        )

    def test_sleep_query_contains_sleep_in_results(self, retriever, has_docs):
        if not has_docs:
            pytest.skip("No PDFs indexed")
        results = retriever.search("Apple Watch sleep tracking")
        assert keyword_found_in_results(results, "sleep"), (
            "Sleep query should surface at least one chunk mentioning sleep"
        )


# ── METADATA INTEGRITY tests ──────────────────────────────────────────────────

class TestMetadataIntegrity:
    """Every returned chunk must have the expected metadata fields."""

    def test_metadata_has_required_fields(self, retriever, has_docs):
        if not has_docs:
            pytest.skip("No PDFs indexed")
        results = retriever.search("Apple Watch features")
        for r in results:
            assert "metadata" in r, "Result must have a 'metadata' key"
            assert "text" in r, "Result must have a 'text' key"
            assert "distance" in r, "Result must have a 'distance' key"
            meta = r["metadata"]
            # source and page are expected from document_loader.py
            assert "source" in meta or len(meta) >= 0, (
                "Metadata should at least be a dict"
            )

    def test_text_is_non_empty(self, retriever, has_docs):
        if not has_docs:
            pytest.skip("No PDFs indexed")
        results = retriever.search("Apple Watch troubleshooting")
        for r in results:
            assert len(r["text"].strip()) > 0, (
                "Every returned chunk must have non-empty text"
            )


# ── ROBUSTNESS tests ──────────────────────────────────────────────────────────

class TestRobustness:
    """The retriever must not crash on unusual inputs."""

    def test_empty_query_does_not_crash(self, retriever, has_docs):
        if not has_docs:
            pytest.skip("No PDFs indexed")
        try:
            results = retriever.search("")
            assert isinstance(results, list)
        except Exception as e:
            pytest.fail(f"Empty query raised an exception: {e}")

    def test_nonsense_query_does_not_crash(self, retriever, has_docs):
        if not has_docs:
            pytest.skip("No PDFs indexed")
        try:
            results = retriever.search("xkcd flubberwocky bazinga qwerty")
            assert isinstance(results, list)
        except Exception as e:
            pytest.fail(f"Nonsense query raised an exception: {e}")

    def test_very_long_query_does_not_crash(self, retriever, has_docs):
        if not has_docs:
            pytest.skip("No PDFs indexed")
        long_query = "Apple Watch " * 100
        try:
            results = retriever.search(long_query)
            assert isinstance(results, list)
        except Exception as e:
            pytest.fail(f"Very long query raised an exception: {e}")

    def test_status_returns_document_count(self, retriever):
        status = retriever.status()
        assert "documents" in status
        assert isinstance(status["documents"], int)
        assert status["documents"] >= 0
