"""
Tests for Phase 3 — Query-Time Pipeline.
Tests query classifier, guardrails, prompt templates, thread manager,
hybrid search, and the full RAG pipeline (retrieval only, no LLM).
"""

import json
import os
import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.generation.query_classifier import (
    classify_query, QueryIntent, ClassificationResult,
)
from src.generation.prompt_templates import (
    build_user_prompt, get_refusal_response, build_system_prompt,
    SYSTEM_PROMPT, REFUSAL_PII,
)
from src.generation.guardrails import (
    check_advisory_language,
    check_citation_present,
    check_response_length,
    check_pii_leakage,
    check_hallucination,
    validate_response,
)
from src.chat.thread_manager import ThreadManager, Thread, Message


# =============================================================================
# Query Classifier Tests
# =============================================================================

class TestQueryClassifier:
    """Tests for query_classifier.py."""

    # ── PII Detection ────────────────────────────────────────────────────

    def test_detects_pan_number(self):
        result = classify_query("My PAN is ABCPD1234E, check my portfolio")
        assert result.intent == QueryIntent.PII_DETECTED
        assert result.pii_type == "pan"

    def test_detects_aadhaar_number(self):
        result = classify_query("My Aadhaar is 1234 5678 9012")
        assert result.intent == QueryIntent.PII_DETECTED
        assert result.pii_type == "aadhaar"

    def test_detects_email(self):
        result = classify_query("Send details to user@example.com")
        assert result.intent == QueryIntent.PII_DETECTED
        assert result.pii_type == "email"

    def test_detects_phone_number(self):
        result = classify_query("Call me at 9876543210")
        assert result.intent == QueryIntent.PII_DETECTED
        assert result.pii_type == "phone"

    # ── Advisory Detection ───────────────────────────────────────────────

    def test_detects_advisory_should_invest(self):
        result = classify_query("Should I invest in HDFC Mid-Cap Fund?")
        assert result.intent == QueryIntent.ADVISORY

    def test_detects_advisory_recommend(self):
        result = classify_query("Can you recommend a good mutual fund?")
        assert result.intent == QueryIntent.ADVISORY

    def test_detects_advisory_suggest(self):
        result = classify_query("Please suggest the best fund for me")
        assert result.intent == QueryIntent.ADVISORY

    def test_detects_advisory_is_it_safe(self):
        result = classify_query("Is it safe to invest in this fund?")
        assert result.intent == QueryIntent.ADVISORY

    # ── Comparative Detection ────────────────────────────────────────────

    def test_detects_comparative_which_better(self):
        result = classify_query("Which is better: HDFC or SBI fund?")
        assert result.intent == QueryIntent.COMPARATIVE

    def test_detects_comparative_compare(self):
        result = classify_query("Compare HDFC Mid-Cap with HDFC Small-Cap")
        assert result.intent == QueryIntent.COMPARATIVE

    # ── Off-Topic Detection ──────────────────────────────────────────────

    def test_detects_off_topic(self):
        result = classify_query("What is the weather today in Mumbai?")
        assert result.intent == QueryIntent.OFF_TOPIC

    def test_detects_off_topic_joke(self):
        result = classify_query("Tell me a funny joke please")
        assert result.intent == QueryIntent.OFF_TOPIC

    # ── Factual Detection ────────────────────────────────────────────────

    def test_detects_factual_expense_ratio(self):
        result = classify_query("What is the expense ratio of HDFC Mid-Cap Fund?")
        assert result.intent == QueryIntent.FACTUAL

    def test_detects_factual_exit_load(self):
        result = classify_query("What is the exit load for HDFC ELSS?")
        assert result.intent == QueryIntent.FACTUAL

    def test_detects_factual_nav(self):
        result = classify_query("What is the NAV of HDFC Equity Fund?")
        assert result.intent == QueryIntent.FACTUAL

    def test_detects_factual_sip(self):
        result = classify_query("What is the minimum SIP amount?")
        assert result.intent == QueryIntent.FACTUAL

    def test_detects_factual_fund_manager(self):
        result = classify_query("Who is the fund manager of HDFC Mid-Cap?")
        assert result.intent == QueryIntent.FACTUAL


# =============================================================================
# Prompt Templates Tests
# =============================================================================

class TestPromptTemplates:
    """Tests for prompt_templates.py."""

    def test_system_prompt_contains_rules(self):
        assert "3 sentences" in SYSTEM_PROMPT
        assert "NEVER" in SYSTEM_PROMPT
        assert "financial advisor" in SYSTEM_PROMPT

    def test_build_system_prompt_injects_date(self):
        prompt = build_system_prompt("2026-04-13")
        assert "2026-04-13" in prompt

    def test_build_user_prompt_formats_context(self):
        chunks = [
            {
                "text": "NAV is ₹215.55",
                "metadata": {
                    "scheme_name": "HDFC Mid-Cap Fund",
                    "section": "fund_details",
                    "source_url": "https://groww.in/mutual-funds/hdfc-mid-cap-fund-direct-growth",
                },
            }
        ]
        prompt = build_user_prompt("What is the NAV?", chunks)
        assert "NAV is ₹215.55" in prompt
        assert "SOURCE 1" in prompt
        assert "fund_details" in prompt
        assert "Question: What is the NAV?" in prompt

    def test_refusal_advisory_response(self):
        response = get_refusal_response("advisory", "2026-04-13")
        assert "investment advice" in response
        assert "amfiindia.com" in response
        assert "2026-04-13" in response

    def test_refusal_pii_response(self):
        response = get_refusal_response("pii_detected", "2026-04-13")
        assert "personal information" in response

    def test_refusal_off_topic_response(self):
        response = get_refusal_response("off_topic", "2026-04-13")
        assert "mutual fund" in response


# =============================================================================
# Guardrails Tests
# =============================================================================

class TestGuardrails:
    """Tests for guardrails.py."""

    # ── Advisory Language Check ──────────────────────────────────────────

    def test_detects_advisory_in_output(self):
        response = "I would recommend investing in this fund for long-term growth."
        violations = check_advisory_language(response)
        assert len(violations) > 0

    def test_passes_clean_factual_output(self):
        response = "The expense ratio of HDFC Mid-Cap Fund is 0.77% per annum."
        violations = check_advisory_language(response)
        assert len(violations) == 0

    def test_detects_should_invest(self):
        response = "You should invest in HDFC ELSS for tax saving."
        violations = check_advisory_language(response)
        assert len(violations) > 0

    # ── Citation Check ───────────────────────────────────────────────────

    def test_passes_with_citation(self):
        response = "The exit load is 1%. Source: https://groww.in/mutual-funds/test"
        violations = check_citation_present(response)
        assert len(violations) == 0

    def test_fails_without_citation(self):
        response = "The exit load is 1%."
        violations = check_citation_present(response)
        assert len(violations) > 0

    # ── Length Check ─────────────────────────────────────────────────────

    def test_passes_short_response(self):
        response = "The expense ratio is 0.77%. Source: https://example.com\nLast updated from sources: 2026-04-13"
        violations = check_response_length(response)
        assert len(violations) == 0

    def test_fails_long_response(self):
        response = (
            "First sentence about the fund. "
            "Second sentence with more details. "
            "Third sentence with additional info. "
            "Fourth sentence that exceeds the limit. "
            "Fifth sentence that is definitely too much."
        )
        violations = check_response_length(response)
        assert len(violations) > 0

    # ── PII Leakage Check ────────────────────────────────────────────────

    def test_detects_pan_in_output(self):
        response = "Your PAN ABCPD1234E has been noted."
        violations = check_pii_leakage(response)
        assert len(violations) > 0

    def test_passes_no_pii(self):
        response = "The minimum SIP amount is ₹100."
        violations = check_pii_leakage(response)
        assert len(violations) == 0

    def test_allows_official_emails(self):
        response = "Contact support@groww.in for more details."
        violations = check_pii_leakage(response)
        assert len(violations) == 0

    # ── Hallucination Check ──────────────────────────────────────────────

    def test_passes_when_numbers_match_context(self):
        response = "The expense ratio is 0.77%."
        chunks = [{"text": "Expense ratio: 0.77%"}]
        violations = check_hallucination(response, chunks)
        assert len(violations) == 0

    def test_flags_hallucinated_numbers(self):
        response = "The expense ratio is 99.99%."
        chunks = [{"text": "Expense ratio: 0.77%"}]
        violations = check_hallucination(response, chunks)
        assert len(violations) > 0

    # ── Full Validation ──────────────────────────────────────────────────

    def test_validate_passes_clean_response(self):
        response = (
            "The expense ratio of HDFC Mid-Cap Fund is 0.77% per annum. "
            "Source: https://groww.in/mutual-funds/hdfc-mid-cap-fund-direct-growth\n"
            "Last updated from sources: 2026-04-13"
        )
        chunks = [{"text": "Expense ratio: 0.77%. Scrape date: 2026-04-13"}]
        result = validate_response(response, context_chunks=chunks)
        assert result.passed, f"Violations: {result.violations}"

    def test_validate_fails_advisory_response(self):
        response = (
            "I would recommend this fund for long-term investors. "
            "Source: https://groww.in/test\n"
            "Last updated from sources: 2026-04-13"
        )
        result = validate_response(response)
        assert not result.passed
        assert any("Advisory" in v for v in result.violations)


# =============================================================================
# Thread Manager Tests
# =============================================================================

class TestThreadManager:
    """Tests for thread_manager.py."""

    def _get_temp_manager(self):
        """Create a thread manager with a temp database."""
        tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        tmp.close()
        return ThreadManager(db_path=tmp.name), tmp.name

    def test_create_thread(self):
        mgr, path = self._get_temp_manager()
        try:
            thread = mgr.create_thread("Test Chat")
            assert thread.thread_id
            assert thread.title == "Test Chat"
            assert len(thread.messages) == 0
        finally:
            mgr.close()
            os.unlink(path)

    def test_add_and_retrieve_messages(self):
        mgr, path = self._get_temp_manager()
        try:
            thread = mgr.create_thread()
            mgr.add_message(thread.thread_id, "user", "What is the NAV?")
            mgr.add_message(
                thread.thread_id, "assistant", "The NAV is ₹215.55.",
                citations=["https://groww.in/test"]
            )

            retrieved = mgr.get_thread(thread.thread_id)
            assert len(retrieved.messages) == 2
            assert retrieved.messages[0].role == "user"
            assert retrieved.messages[1].role == "assistant"
            assert "https://groww.in/test" in retrieved.messages[1].citations
        finally:
            mgr.close()
            os.unlink(path)

    def test_auto_title_from_first_message(self):
        mgr, path = self._get_temp_manager()
        try:
            thread = mgr.create_thread()
            mgr.add_message(thread.thread_id, "user", "What is the expense ratio?")

            retrieved = mgr.get_thread(thread.thread_id)
            assert "expense ratio" in retrieved.title.lower()
        finally:
            mgr.close()
            os.unlink(path)

    def test_list_threads(self):
        mgr, path = self._get_temp_manager()
        try:
            mgr.create_thread("Chat 1")
            mgr.create_thread("Chat 2")
            mgr.create_thread("Chat 3")

            threads = mgr.list_threads()
            assert len(threads) == 3
        finally:
            mgr.close()
            os.unlink(path)

    def test_delete_thread(self):
        mgr, path = self._get_temp_manager()
        try:
            thread = mgr.create_thread()
            mgr.add_message(thread.thread_id, "user", "Test message")

            assert mgr.delete_thread(thread.thread_id)
            assert mgr.get_thread(thread.thread_id) is None
        finally:
            mgr.close()
            os.unlink(path)

    def test_get_recent_history(self):
        mgr, path = self._get_temp_manager()
        try:
            thread = mgr.create_thread()
            for i in range(5):
                mgr.add_message(thread.thread_id, "user", f"Question {i}")
                mgr.add_message(thread.thread_id, "assistant", f"Answer {i}")

            history = mgr.get_recent_history(thread.thread_id, max_pairs=3)
            assert len(history) == 6  # 3 pairs
            assert history[0]["role"] == "user"
            assert "Question 2" in history[0]["content"]
        finally:
            mgr.close()
            os.unlink(path)

    def test_get_nonexistent_thread(self):
        mgr, path = self._get_temp_manager()
        try:
            assert mgr.get_thread("nonexistent-id") is None
        finally:
            mgr.close()
            os.unlink(path)


# =============================================================================
# Hybrid Search Tests (requires data in ChromaDB + SQLite)
# =============================================================================

class TestHybridSearch:
    """Tests for hybrid_search.py (integration — requires Phase 2 data)."""

    def test_search_returns_results(self):
        """Test that hybrid search returns results from existing data."""
        from config import VECTORSTORE_DIR, CHUNKS_DB_FILE
        if not VECTORSTORE_DIR.exists() or not CHUNKS_DB_FILE.exists():
            pytest.skip("Vector store or chunks DB not available")

        from src.retrieval.hybrid_search import HybridSearcher
        searcher = HybridSearcher()
        results = searcher.search("expense ratio HDFC Mid-Cap", top_k=3)

        assert len(results) > 0
        for r in results:
            assert "chunk_id" in r
            assert "text" in r
            assert "rrf_score" in r

    def test_semantic_search_returns_results(self):
        """Test semantic-only search."""
        from config import VECTORSTORE_DIR
        if not VECTORSTORE_DIR.exists():
            pytest.skip("Vector store not available")

        from src.retrieval.hybrid_search import HybridSearcher
        searcher = HybridSearcher()
        results = searcher.semantic_search("exit load", top_k=3)

        assert len(results) > 0
        assert all(r["source"] == "semantic" for r in results)

    def test_bm25_search_returns_results(self):
        """Test BM25-only search."""
        from config import CHUNKS_DB_FILE
        if not CHUNKS_DB_FILE.exists():
            pytest.skip("Chunks DB not available")

        from src.retrieval.hybrid_search import HybridSearcher
        searcher = HybridSearcher()
        results = searcher.bm25_search("HDFC Mid-Cap Fund", top_k=3)

        assert len(results) > 0
        assert all(r["source"] == "bm25" for r in results)


# =============================================================================
# RAG Pipeline Tests (retrieval only — no LLM calls)
# =============================================================================

class TestRAGPipeline:
    """Tests for rag_pipeline.py (no LLM calls)."""

    def test_pipeline_refuses_advisory(self):
        from src.rag_pipeline import RAGPipeline
        pipeline = RAGPipeline(use_reranker=False)
        response = pipeline.answer("Should I invest in HDFC Mid-Cap?")
        assert response.intent == "advisory"
        assert "investment advice" in response.answer.lower() or "advice" in response.answer.lower()

    def test_pipeline_refuses_pii(self):
        from src.rag_pipeline import RAGPipeline
        pipeline = RAGPipeline(use_reranker=False)
        response = pipeline.answer("My PAN is ABCPD1234E")
        assert response.intent == "pii_detected"
        assert "personal information" in response.answer.lower()

    def test_pipeline_refuses_off_topic(self):
        from src.rag_pipeline import RAGPipeline
        pipeline = RAGPipeline(use_reranker=False)
        response = pipeline.answer("What is the weather today in Delhi?")
        assert response.intent == "off_topic"

    def test_pipeline_search_only(self):
        """Test retrieval without LLM call."""
        from config import VECTORSTORE_DIR
        if not VECTORSTORE_DIR.exists():
            pytest.skip("Vector store not available")

        from src.rag_pipeline import RAGPipeline
        pipeline = RAGPipeline(use_reranker=False)
        results = pipeline.search_only("expense ratio", top_k=3)
        assert len(results) > 0
