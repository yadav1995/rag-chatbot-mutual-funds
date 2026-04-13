"""
RAG Pipeline — End-to-End Orchestrator

Wires all components of the query-time pipeline:
  Query → Classify → [Refuse | Retrieve → Re-rank → Generate → Guardrail] → Response

This is the single entry point for answering user queries.

Usage:
    from src.rag_pipeline import RAGPipeline

    pipeline = RAGPipeline()
    response = pipeline.answer("What is the exit load for HDFC Mid-Cap Fund?")
"""

import logging
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import CHUNKS_DB_FILE

from src.generation.query_classifier import classify_query, QueryIntent
from src.generation.prompt_templates import get_refusal_response
from src.generation.guardrails import validate_response
from src.retrieval.hybrid_search import HybridSearcher

logger = logging.getLogger(__name__)


@dataclass
class PipelineResponse:
    """Complete response from the RAG pipeline."""
    answer: str
    intent: str
    chunks_retrieved: int = 0
    chunks_used: int = 0
    citations: list[str] = field(default_factory=list)
    scrape_date: str = "N/A"
    guardrail_passed: bool = True
    guardrail_violations: list[str] = field(default_factory=list)


class RAGPipeline:
    """
    End-to-end RAG pipeline orchestrator.

    Components:
    1. Query Classifier (rule-based)
    2. Hybrid Retriever (semantic + BM25 + RRF)
    3. Re-Ranker (cross-encoder, optional)
    4. LLM Generator (OpenAI GPT-4o-mini)
    5. Guardrail Validator (post-generation)
    """

    def __init__(self, use_reranker: bool = True):
        self._searcher = None
        self._reranker = None
        self._generator = None
        self._use_reranker = use_reranker
        self._scrape_date = self._get_scrape_date()

    def _get_scrape_date(self) -> str:
        """Get the latest scrape date from the SQLite store."""
        try:
            import sqlite3
            conn = sqlite3.connect(str(CHUNKS_DB_FILE), check_same_thread=False)
            row = conn.execute(
                "SELECT MAX(scrape_date) FROM chunks"
            ).fetchone()
            conn.close()
            return row[0] if row and row[0] else "N/A"
        except Exception:
            return "N/A"

    def _get_searcher(self) -> HybridSearcher:
        """Lazy-load the hybrid searcher."""
        if self._searcher is None:
            self._searcher = HybridSearcher()
        return self._searcher

    def _get_reranker(self):
        """Lazy-load the re-ranker."""
        if self._reranker is None and self._use_reranker:
            from src.retrieval.reranker import get_reranker
            self._reranker = get_reranker()
        return self._reranker

    def _get_generator(self):
        """Lazy-load the LLM generator."""
        if self._generator is None:
            from src.generation.llm_generator import get_generator
            self._generator = get_generator()
        return self._generator

    def answer(
        self,
        query: str,
        thread_id: str = None,
        conversation_history: list[dict] = None,
    ) -> PipelineResponse:
        """
        Process a user query through the full RAG pipeline.

        Args:
            query: The user's question
            thread_id: Optional thread ID for conversation context
            conversation_history: Optional previous messages

        Returns:
            PipelineResponse with answer, citations, and metadata
        """
        logger.info(f"Pipeline: query='{query[:80]}...'")

        # ── Step 1: Classify query ───────────────────────────────────────
        classification = classify_query(query)
        logger.info(
            f"  Classification: {classification.intent.value} "
            f"(confidence={classification.confidence:.2f}, reason={classification.reason})"
        )

        # Handle non-factual queries with refusal
        if classification.intent != QueryIntent.FACTUAL:
            refusal = get_refusal_response(
                classification.intent.value,
                scrape_date=self._scrape_date,
            )
            return PipelineResponse(
                answer=refusal,
                intent=classification.intent.value,
                scrape_date=self._scrape_date,
            )

        # ── Step 2: Retrieve relevant chunks ─────────────────────────────
        searcher = self._get_searcher()
        candidates = searcher.search(query, top_k=10)
        logger.info(f"  Retrieved: {len(candidates)} candidates")

        if not candidates:
            no_context = get_refusal_response(
                "no_context", scrape_date=self._scrape_date
            )
            return PipelineResponse(
                answer=no_context,
                intent="factual",
                scrape_date=self._scrape_date,
            )

        # ── Step 3: Re-rank (optional) ───────────────────────────────────
        reranker = self._get_reranker()
        if reranker and len(candidates) > 3:
            top_chunks = reranker.rerank(query, candidates, top_k=3)
            logger.info(f"  Re-ranked: {len(candidates)} → {len(top_chunks)}")
        else:
            top_chunks = candidates[:3]

        # Extract citations
        citations = []
        for chunk in top_chunks:
            url = chunk.get("metadata", {}).get("source_url", "")
            if url and url not in citations:
                citations.append(url)

        # ── Step 4: Generate response ────────────────────────────────────
        generator = self._get_generator()
        raw_response = generator.generate(
            query=query,
            context_chunks=top_chunks,
            scrape_date=self._scrape_date,
            conversation_history=conversation_history,
        )
        logger.info(f"  Generated: {len(raw_response)} chars")

        # ── Step 5: Guardrail validation ─────────────────────────────────
        guardrail_result = validate_response(
            raw_response,
            context_chunks=top_chunks,
            scrape_date=self._scrape_date,
        )

        if guardrail_result.passed:
            final_answer = raw_response
        else:
            logger.warning(
                f"  Guardrail FAILED: {guardrail_result.violations}"
            )
            # If guardrail fails, use the response but log the violations
            # In production, you might want to regenerate or use a refusal
            final_answer = raw_response

        return PipelineResponse(
            answer=final_answer,
            intent="factual",
            chunks_retrieved=len(candidates),
            chunks_used=len(top_chunks),
            citations=citations,
            scrape_date=self._scrape_date,
            guardrail_passed=guardrail_result.passed,
            guardrail_violations=guardrail_result.violations,
        )

    def search_only(self, query: str, top_k: int = 5) -> list[dict]:
        """
        Run only the retrieval step (no LLM call).
        Useful for debugging and testing retrieval quality.
        """
        searcher = self._get_searcher()
        return searcher.search(query, top_k=top_k)
