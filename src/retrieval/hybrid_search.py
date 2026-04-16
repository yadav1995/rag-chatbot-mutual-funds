"""
Hybrid Search — Semantic + BM25 with Reciprocal Rank Fusion

Implements the retrieval layer per RAGArchitecture.md §4.3:
- Semantic Search: Cosine similarity via ChromaDB embeddings
- BM25 Keyword Search: TF-IDF sparse retrieval on raw text
- Score Fusion: Reciprocal Rank Fusion (RRF) merges both result sets

Usage:
    searcher = HybridSearcher()
    results = searcher.search("What is the exit load for HDFC Mid-Cap Fund?", top_k=10)
"""

import json
import logging
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from config import (
    CHROMA_COLLECTION_NAME,
    CHROMA_DISTANCE_METRIC,
    VECTORSTORE_DIR,
    CHUNKS_DB_FILE,
    EMBEDDING_MODEL,
    EMBEDDING_BATCH_SIZE,
)

logger = logging.getLogger(__name__)

# RRF constant — controls how quickly rank importance decays
RRF_K = 60


class HybridSearcher:
    """
    Hybrid retriever combining semantic (dense) and BM25 (sparse) search
    with Reciprocal Rank Fusion.
    """

    def __init__(self):
        self._chroma_collection = None
        self._embedding_service = None
        self._bm25 = None
        self._bm25_corpus = None  # list of (chunk_id, text, metadata_json)

    # ── Lazy initialization ──────────────────────────────────────────────

    def _get_chroma(self):
        """Lazy-load ChromaDB collection."""
        if self._chroma_collection is None:
            import chromadb
            client = chromadb.PersistentClient(path=str(VECTORSTORE_DIR))
            try:
                self._chroma_collection = client.get_collection(
                    name=CHROMA_COLLECTION_NAME,
                )
                logger.info(
                    f"ChromaDB loaded: {self._chroma_collection.count()} vectors"
                )
            except Exception as e:
                logger.error(f"Failed to load ChromaDB collection '{CHROMA_COLLECTION_NAME}' from {VECTORSTORE_DIR}: {e}")
                # Provide a more helpful error for deployment environments
                raise RuntimeError(
                    f"ChromaDB collection '{CHROMA_COLLECTION_NAME}' not found in {VECTORSTORE_DIR}. "
                    "Ensure the data/vectorstore directory is present in the repository."
                ) from e
        return self._chroma_collection

    def _get_embedding_service(self):
        """Lazy-load embedding model for query encoding."""
        if self._embedding_service is None:
            from src.ingestion.embedder import EmbeddingService
            self._embedding_service = EmbeddingService()
        return self._embedding_service

    def _get_bm25_index(self):
        """Lazy-build BM25 index from SQLite chunks."""
        if self._bm25 is None:
            from rank_bm25 import BM25Okapi

            conn = sqlite3.connect(str(CHUNKS_DB_FILE), check_same_thread=False)
            rows = conn.execute(
                "SELECT chunk_id, full_text, metadata_json FROM chunks"
            ).fetchall()
            conn.close()

            if not rows:
                logger.warning("No chunks found in SQLite for BM25 index")
                return None

            self._bm25_corpus = []
            tokenized_corpus = []

            for chunk_id, text, meta_json in rows:
                self._bm25_corpus.append((chunk_id, text, meta_json))
                # Simple whitespace tokenization for BM25
                tokenized_corpus.append(text.lower().split())

            self._bm25 = BM25Okapi(tokenized_corpus)
            logger.info(f"BM25 index built: {len(self._bm25_corpus)} documents")

        return self._bm25

    # ── Semantic Search ──────────────────────────────────────────────────

    def semantic_search(self, query: str, top_k: int = 10) -> list[dict]:
        """
        Cosine similarity search via ChromaDB.
        Returns ranked list of {chunk_id, text, metadata, score}.
        """
        collection = self._get_chroma()
        emb_service = self._get_embedding_service()

        query_embedding = emb_service.embed_query(query)

        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=min(top_k, collection.count()),
            include=["documents", "metadatas", "distances"],
        )

        hits = []
        if results["ids"] and results["ids"][0]:
            for i, chunk_id in enumerate(results["ids"][0]):
                # ChromaDB returns distances; for cosine, similarity = 1 - distance
                distance = results["distances"][0][i]
                similarity = 1.0 - distance

                hits.append({
                    "chunk_id": chunk_id,
                    "text": results["documents"][0][i],
                    "metadata": results["metadatas"][0][i],
                    "score": similarity,
                    "source": "semantic",
                })

        logger.debug(f"Semantic search: {len(hits)} hits for '{query[:50]}...'")
        return hits

    # ── BM25 Keyword Search ──────────────────────────────────────────────

    def bm25_search(self, query: str, top_k: int = 10) -> list[dict]:
        """
        BM25 keyword search on the SQLite document store.
        Returns ranked list of {chunk_id, text, metadata, score}.
        """
        bm25 = self._get_bm25_index()
        if bm25 is None:
            return []

        tokenized_query = query.lower().split()
        scores = bm25.get_scores(tokenized_query)

        # Get top-k indices by score
        ranked_indices = sorted(
            range(len(scores)), key=lambda i: scores[i], reverse=True
        )[:top_k]

        hits = []
        for rank, idx in enumerate(ranked_indices):
            if scores[idx] <= 0:
                continue  # Skip zero-score matches

            chunk_id, text, meta_json = self._bm25_corpus[idx]
            metadata = json.loads(meta_json) if meta_json else {}

            hits.append({
                "chunk_id": chunk_id,
                "text": text,
                "metadata": metadata,
                "score": float(scores[idx]),
                "source": "bm25",
            })

        logger.debug(f"BM25 search: {len(hits)} hits for '{query[:50]}...'")
        return hits

    # ── Reciprocal Rank Fusion ───────────────────────────────────────────

    def _rrf_fusion(
        self,
        semantic_hits: list[dict],
        bm25_hits: list[dict],
        k: int = RRF_K,
    ) -> list[dict]:
        """
        Merge results from semantic and BM25 search using RRF.

        RRF_score(doc) = Σ 1 / (k + rank_i(doc))

        where k = 60 (standard constant) and rank_i is the rank in each
        result list (1-based).
        """
        fused_scores: dict[str, float] = {}
        chunk_data: dict[str, dict] = {}

        # Score semantic results
        for rank, hit in enumerate(semantic_hits, start=1):
            cid = hit["chunk_id"]
            fused_scores[cid] = fused_scores.get(cid, 0.0) + 1.0 / (k + rank)
            chunk_data[cid] = hit

        # Score BM25 results
        for rank, hit in enumerate(bm25_hits, start=1):
            cid = hit["chunk_id"]
            fused_scores[cid] = fused_scores.get(cid, 0.0) + 1.0 / (k + rank)
            if cid not in chunk_data:
                chunk_data[cid] = hit

        # Sort by fused score
        ranked = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)

        results = []
        for chunk_id, rrf_score in ranked:
            entry = chunk_data[chunk_id].copy()
            entry["rrf_score"] = rrf_score
            entry["source"] = "hybrid"
            results.append(entry)

        return results

    # ── Main Search Entry Point ──────────────────────────────────────────

    def search(
        self,
        query: str,
        top_k: int = 10,
        semantic_weight: int = 10,
        bm25_weight: int = 10,
        scheme_filter: str = None,
    ) -> list[dict]:
        """
        Hybrid search: semantic + BM25 with RRF fusion.

        Args:
            query: User query string
            top_k: Number of final results to return
            semantic_weight: Number of candidates from semantic search
            bm25_weight: Number of candidates from BM25 search
            scheme_filter: Optional scheme_slug to filter results

        Returns:
            List of chunk dicts ranked by RRF score
        """
        # Run both searches
        semantic_hits = self.semantic_search(query, top_k=semantic_weight)
        bm25_hits = self.bm25_search(query, top_k=bm25_weight)

        # Fuse results
        fused = self._rrf_fusion(semantic_hits, bm25_hits)

        # Optional scheme filter
        if scheme_filter:
            fused = [
                h for h in fused
                if h.get("metadata", {}).get("scheme_slug") == scheme_filter
            ]

        # Return top-k
        results = fused[:top_k]

        logger.info(
            f"Hybrid search: query='{query[:50]}...', "
            f"semantic={len(semantic_hits)}, bm25={len(bm25_hits)}, "
            f"fused={len(fused)}, returned={len(results)}"
        )

        return results


# =============================================================================
# CLI Entry Point — for testing
# =============================================================================

if __name__ == "__main__":
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    query = " ".join(sys.argv[1:]) or "What is the expense ratio of HDFC Mid-Cap Fund?"
    print(f"\nQuery: {query}\n")

    searcher = HybridSearcher()
    results = searcher.search(query, top_k=5)

    for i, r in enumerate(results, 1):
        meta = r.get("metadata", {})
        print(f"  [{i}] {r['chunk_id']} (RRF={r.get('rrf_score', 0):.4f})")
        print(f"      Section: {meta.get('section', 'N/A')}")
        print(f"      Scheme:  {meta.get('scheme_name', 'N/A')}")
        print(f"      Text:    {r['text'][:120]}...")
        print()
