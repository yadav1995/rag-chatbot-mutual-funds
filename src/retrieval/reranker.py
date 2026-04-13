"""
Re-Ranker — Cross-Encoder Re-Ranking

Implements the re-ranking step per RAGArchitecture.md §4.4:
- Takes top-N candidates from hybrid search
- Re-scores each (query, chunk) pair using a cross-encoder
- Returns the top-K most relevant chunks

Model: cross-encoder/ms-marco-MiniLM-L-6-v2
"""

import logging
import time
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

logger = logging.getLogger(__name__)

# Default re-ranker model
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class ReRanker:
    """
    Cross-encoder re-ranker that scores (query, document) pairs
    for more precise relevance ranking than bi-encoder similarity.
    """

    def __init__(self, model_name: str = RERANKER_MODEL):
        self._model = None
        self._model_name = model_name

    def _load_model(self):
        """Lazy-load the cross-encoder model."""
        if self._model is None:
            logger.info(f"Loading re-ranker model: {self._model_name}")
            start = time.time()

            from sentence_transformers import CrossEncoder
            self._model = CrossEncoder(self._model_name)

            elapsed = time.time() - start
            logger.info(f"Re-ranker loaded in {elapsed:.1f}s")

        return self._model

    def rerank(
        self,
        query: str,
        candidates: list[dict],
        top_k: int = 3,
    ) -> list[dict]:
        """
        Re-rank candidate chunks using the cross-encoder.

        Args:
            query: The user query
            candidates: List of chunk dicts from hybrid search
            top_k: Number of top results to return after re-ranking

        Returns:
            Top-K chunks sorted by cross-encoder relevance score
        """
        if not candidates:
            return []

        if len(candidates) <= top_k:
            # No need to re-rank if we have fewer candidates than top_k
            return candidates

        model = self._load_model()

        # Create (query, document) pairs for scoring
        pairs = [(query, c["text"]) for c in candidates]

        # Score all pairs
        scores = model.predict(pairs)

        # Attach scores and sort
        scored = []
        for candidate, score in zip(candidates, scores):
            entry = candidate.copy()
            entry["rerank_score"] = float(score)
            scored.append(entry)

        scored.sort(key=lambda x: x["rerank_score"], reverse=True)

        results = scored[:top_k]

        logger.info(
            f"Re-ranked {len(candidates)} candidates → top {len(results)}, "
            f"best_score={results[0]['rerank_score']:.4f}"
        )

        return results


# Singleton for reuse
_reranker_instance = None


def get_reranker() -> ReRanker:
    """Get or create the singleton re-ranker."""
    global _reranker_instance
    if _reranker_instance is None:
        _reranker_instance = ReRanker()
    return _reranker_instance
