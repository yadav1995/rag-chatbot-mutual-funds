"""
Embedder Module — Embedding Generation + Dual-Store (ChromaDB + SQLite)

Implements the embedding and storage pipeline as specified in
ChunkingEmbeddingArchitecture.md §5-6:

- Embedding generation using sentence-transformers/all-MiniLM-L6-v2
- Batch encoding with L2 normalization
- ChromaDB upsert (vectors + metadata + document text)
- SQLite upsert (full text + metadata + content hashes for audit)
- Incremental update strategy: insert/update/skip/delete

Usage:
    python src/ingestion/embedder.py                  # Embed all chunks from processed/
    python src/ingestion/embedder.py --rebuild        # Rebuild entire vector store from scratch
"""

import argparse
import json
import logging
import sqlite3
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from config import (
    EMBEDDING_MODEL,
    EMBEDDING_DIMENSION,
    EMBEDDING_BATCH_SIZE,
    CHROMA_COLLECTION_NAME,
    CHROMA_DISTANCE_METRIC,
    VECTORSTORE_DIR,
    CHUNKS_DB_FILE,
    PROCESSED_DIR,
    ensure_directories,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Embedding Service
# =============================================================================

class EmbeddingService:
    """
    Generates embeddings using sentence-transformers.
    Primary model: all-MiniLM-L6-v2 (384-dim, free, local, fast).
    """

    def __init__(self, model_name: str = EMBEDDING_MODEL):
        logger.info(f"Loading embedding model: {model_name}")
        start = time.time()

        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name)
        # Use new method name (v3+), fallback to deprecated for older versions
        try:
            self.dimension = self.model.get_embedding_dimension()
        except AttributeError:
            self.dimension = self.model.get_sentence_embedding_dimension()

        elapsed = time.time() - start
        logger.info(f"Model loaded in {elapsed:.1f}s (dimension={self.dimension})")

    def embed_texts(self, texts: list[str], batch_size: int = EMBEDDING_BATCH_SIZE) -> list[list[float]]:
        """
        Generate embeddings for a list of texts in batches.
        Returns list of float vectors (L2-normalized for cosine similarity).
        """
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=len(texts) > 10,
            normalize_embeddings=True,  # L2 normalization → cosine = dot product
            convert_to_numpy=True,
        )
        return [emb.tolist() for emb in embeddings]

    def embed_query(self, query: str) -> list[float]:
        """Embed a single user query for similarity search."""
        embedding = self.model.encode(
            query,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        return embedding.tolist()


# =============================================================================
# ChromaDB Store
# =============================================================================

class ChromaStore:
    """
    Manages the ChromaDB vector store.
    Collection: groww_mf_chunks with cosine distance.
    """

    def __init__(self, persist_dir: str = None):
        import chromadb

        persist_dir = persist_dir or str(VECTORSTORE_DIR)
        logger.info(f"Initializing ChromaDB at: {persist_dir}")

        self.client = chromadb.PersistentClient(path=persist_dir)
        self.collection = self.client.get_or_create_collection(
            name=CHROMA_COLLECTION_NAME,
            metadata={
                "hnsw:space": CHROMA_DISTANCE_METRIC,
                "hnsw:M": 16,
                "hnsw:construction_ef": 100,
            },
        )
        logger.info(f"ChromaDB collection '{CHROMA_COLLECTION_NAME}': {self.collection.count()} existing vectors")

    def upsert(self, chunk_ids: list[str], embeddings: list[list[float]],
               documents: list[str], metadatas: list[dict]) -> None:
        """Upsert chunks into ChromaDB (handles both insert and update)."""
        if not chunk_ids:
            return

        # ChromaDB metadata values must be str, int, float, or bool
        clean_metadatas = []
        for meta in metadatas:
            clean = {}
            for k, v in meta.items():
                if isinstance(v, (str, int, float, bool)):
                    clean[k] = v
                elif isinstance(v, list):
                    clean[k] = ",".join(str(x) for x in v)
                else:
                    clean[k] = str(v)
            clean_metadatas.append(clean)

        # Upsert in batches to avoid memory issues
        batch_size = 100
        for i in range(0, len(chunk_ids), batch_size):
            end = min(i + batch_size, len(chunk_ids))
            self.collection.upsert(
                ids=chunk_ids[i:end],
                embeddings=embeddings[i:end],
                documents=documents[i:end],
                metadatas=clean_metadatas[i:end],
            )

        logger.info(f"Upserted {len(chunk_ids)} vectors to ChromaDB")

    def delete_stale(self, valid_ids: set[str]) -> int:
        """Delete chunks that are no longer in the current scrape."""
        existing = self.collection.get()
        existing_ids = set(existing["ids"]) if existing["ids"] else set()
        stale_ids = existing_ids - valid_ids

        if stale_ids:
            self.collection.delete(ids=list(stale_ids))
            logger.info(f"Deleted {len(stale_ids)} stale vectors from ChromaDB")
        return len(stale_ids)

    def get_existing_hashes(self) -> dict[str, str]:
        """Get existing chunk_id → content_hash mapping for diff detection."""
        existing = self.collection.get(include=["metadatas"])
        hashes = {}
        if existing["ids"]:
            for id_, meta in zip(existing["ids"], existing["metadatas"]):
                hashes[id_] = meta.get("content_hash", "")
        return hashes

    def count(self) -> int:
        return self.collection.count()

    def reset(self) -> None:
        """Delete and recreate the collection."""
        self.client.delete_collection(CHROMA_COLLECTION_NAME)
        self.collection = self.client.get_or_create_collection(
            name=CHROMA_COLLECTION_NAME,
            metadata={
                "hnsw:space": CHROMA_DISTANCE_METRIC,
                "hnsw:M": 16,
                "hnsw:construction_ef": 100,
            },
        )
        logger.info("ChromaDB collection reset")


# =============================================================================
# SQLite Document Store
# =============================================================================

class SQLiteStore:
    """
    SQLite document store for full-text storage, BM25 search, and audit trail.
    Stores: chunk text, metadata, content hashes.
    """

    def __init__(self, db_path: str = None):
        db_path = db_path or str(CHUNKS_DB_FILE)
        logger.info(f"Initializing SQLite at: {db_path}")

        self.conn = sqlite3.connect(db_path)
        self._create_tables()

    def _create_tables(self) -> None:
        """Create the chunks and fund_facts tables if they don't exist."""
        self.conn.executescript("""
            CREATE TABLE IF NOT EXISTS chunks (
                chunk_id       TEXT PRIMARY KEY,
                scheme_name    TEXT NOT NULL,
                scheme_slug    TEXT NOT NULL,
                amc            TEXT NOT NULL DEFAULT 'HDFC Mutual Fund',
                category       TEXT NOT NULL,
                section        TEXT NOT NULL,
                full_text      TEXT NOT NULL,
                source_url     TEXT NOT NULL,
                scrape_date    TEXT NOT NULL,
                token_count    INTEGER NOT NULL,
                content_hash   TEXT NOT NULL,
                metadata_json  TEXT NOT NULL,
                created_at     TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at     TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );

            CREATE INDEX IF NOT EXISTS idx_chunks_scheme ON chunks(scheme_slug);
            CREATE INDEX IF NOT EXISTS idx_chunks_section ON chunks(section);
            CREATE INDEX IF NOT EXISTS idx_chunks_category ON chunks(category);

            CREATE TABLE IF NOT EXISTS fund_facts (
                scheme_slug    TEXT PRIMARY KEY,
                scheme_name    TEXT NOT NULL,
                category       TEXT,
                nav            TEXT,
                nav_date       TEXT,
                min_sip        TEXT,
                fund_size      TEXT,
                expense_ratio  TEXT,
                rating         TEXT,
                source_url     TEXT NOT NULL,
                scrape_date    TEXT NOT NULL,
                updated_at     TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)
        self.conn.commit()

    def upsert_chunks(self, chunks: list[dict]) -> tuple[int, int, int]:
        """
        Upsert chunks into SQLite.
        Returns (inserted, updated, skipped) counts.
        """
        inserted = updated = skipped = 0
        now = datetime.now(timezone.utc).isoformat()

        for chunk in chunks:
            meta = chunk["metadata"]
            chunk_id = meta["chunk_id"]
            content_hash = meta["content_hash"]

            # Check if chunk already exists
            existing = self.conn.execute(
                "SELECT content_hash FROM chunks WHERE chunk_id = ?", (chunk_id,)
            ).fetchone()

            if existing:
                if existing[0] == content_hash:
                    # Content unchanged, but still refresh the scrape_date and updated_at
                    # to reflect that we successfully verified this data today.
                    self.conn.execute("""
                        UPDATE chunks SET
                            scrape_date = ?, updated_at = ?
                        WHERE chunk_id = ?
                    """, (meta["scrape_date"], now, chunk_id))
                    skipped += 1
                    continue
                else:
                    # Update existing chunk
                    self.conn.execute("""
                        UPDATE chunks SET
                            full_text = ?, content_hash = ?, token_count = ?,
                            scrape_date = ?, metadata_json = ?, updated_at = ?
                        WHERE chunk_id = ?
                    """, (
                        chunk["text"], content_hash, meta["token_count"],
                        meta["scrape_date"], json.dumps(meta), now, chunk_id,
                    ))
                    updated += 1
            else:
                # Insert new chunk
                self.conn.execute("""
                    INSERT INTO chunks (
                        chunk_id, scheme_name, scheme_slug, amc, category,
                        section, full_text, source_url, scrape_date,
                        token_count, content_hash, metadata_json, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    chunk_id, meta["scheme_name"], meta["scheme_slug"],
                    meta["amc"], meta["category"], meta["section"],
                    chunk["text"], meta["source_url"], meta["scrape_date"],
                    meta["token_count"], content_hash, json.dumps(meta),
                    now, now,
                ))
                inserted += 1

        self.conn.commit()
        logger.info(f"SQLite: {inserted} inserted, {updated} updated, {skipped} unchanged")
        return inserted, updated, skipped

    def upsert_fund_facts(self, facts_data: dict) -> None:
        """Upsert structured fund facts for a scheme."""
        now = datetime.now(timezone.utc).isoformat()
        facts = facts_data.get("fund_facts", {})

        self.conn.execute("""
            INSERT OR REPLACE INTO fund_facts (
                scheme_slug, scheme_name, category, nav, nav_date,
                min_sip, fund_size, expense_ratio, rating,
                source_url, scrape_date, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            facts_data["scheme_slug"],
            facts_data["scheme_name"],
            facts_data.get("category", ""),
            facts.get("nav", ""),
            facts.get("nav_date", ""),
            facts.get("min_sip", ""),
            facts.get("fund_size", ""),
            facts.get("expense_ratio", ""),
            facts.get("rating", ""),
            facts_data["source_url"],
            facts_data["scrape_date"],
            now,
        ))
        self.conn.commit()

    def delete_stale(self, valid_ids: set[str]) -> int:
        """Delete chunks not in the valid_ids set."""
        if not valid_ids:
            return 0
        placeholders = ",".join("?" for _ in valid_ids)
        cursor = self.conn.execute(
            f"DELETE FROM chunks WHERE chunk_id NOT IN ({placeholders})",
            list(valid_ids),
        )
        self.conn.commit()
        deleted = cursor.rowcount
        if deleted > 0:
            logger.info(f"Deleted {deleted} stale rows from SQLite")
        return deleted

    def count(self) -> int:
        return self.conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]

    def reset(self) -> None:
        """Drop and recreate tables."""
        self.conn.executescript("DROP TABLE IF EXISTS chunks; DROP TABLE IF EXISTS fund_facts;")
        self._create_tables()
        logger.info("SQLite tables reset")

    def close(self) -> None:
        self.conn.close()


# =============================================================================
# Full Embedding Pipeline
# =============================================================================

def run_embedding_pipeline(rebuild: bool = False) -> dict:
    """
    Full embedding pipeline:
    1. Load chunks from data/processed/all_chunks.json
    2. Generate embeddings
    3. Upsert to ChromaDB + SQLite
    4. Delete stale entries
    5. Load fund facts into SQLite

    Args:
        rebuild: If True, wipe and rebuild the stores from scratch.

    Returns dict with pipeline stats.
    """
    ensure_directories()

    logger.info("=" * 60)
    logger.info("EMBEDDING PIPELINE — Starting")
    logger.info("=" * 60)

    start_time = time.time()

    # --- Load chunks ---
    chunks_file = PROCESSED_DIR / "all_chunks.json"
    if not chunks_file.exists():
        logger.error(f"Chunks file not found: {chunks_file}")
        logger.info("Run the chunker first: python src/ingestion/chunker.py")
        return {"status": "error", "message": "No chunks file found"}

    with open(chunks_file, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    logger.info(f"Loaded {len(chunks)} chunks from {chunks_file}")

    if not chunks:
        logger.warning("No chunks to embed")
        return {"status": "warning", "message": "No chunks found"}

    # --- Initialize stores ---
    chroma_store = ChromaStore()
    sqlite_store = SQLiteStore()

    if rebuild:
        logger.info("REBUILD mode: resetting stores...")
        chroma_store.reset()
        sqlite_store.reset()

    # --- Diff detection: skip unchanged chunks ---
    existing_hashes = chroma_store.get_existing_hashes() if not rebuild else {}
    chunks_to_embed = []

    for chunk in chunks:
        chunk_id = chunk["metadata"]["chunk_id"]
        content_hash = chunk["metadata"]["content_hash"]

        if chunk_id in existing_hashes and existing_hashes[chunk_id] == content_hash:
            continue  # Skip unchanged
        chunks_to_embed.append(chunk)

    skipped = len(chunks) - len(chunks_to_embed)
    logger.info(
        f"Diff check: {len(chunks_to_embed)} to embed, {skipped} unchanged (skipped)"
    )

    # --- Generate embeddings ---
    embeddings = []
    if chunks_to_embed:
        embedding_service = EmbeddingService()
        texts = [c["text"] for c in chunks_to_embed]

        logger.info(f"Generating embeddings for {len(texts)} chunks...")
        embed_start = time.time()
        embeddings = embedding_service.embed_texts(texts)
        embed_time = time.time() - embed_start
        logger.info(f"Embeddings generated in {embed_time:.2f}s")

        # --- Upsert to ChromaDB ---
        chroma_store.upsert(
            chunk_ids=[c["metadata"]["chunk_id"] for c in chunks_to_embed],
            embeddings=embeddings,
            documents=[c["text"] for c in chunks_to_embed],
            metadatas=[c["metadata"] for c in chunks_to_embed],
        )

    # --- Upsert ALL chunks to SQLite (including unchanged, for audit) ---
    inserted, updated, unchanged = sqlite_store.upsert_chunks(chunks)

    # --- Delete stale entries ---
    valid_ids = {c["metadata"]["chunk_id"] for c in chunks}
    stale_chroma = chroma_store.delete_stale(valid_ids)
    stale_sqlite = sqlite_store.delete_stale(valid_ids)

    # --- Load fund facts into SQLite ---
    facts_files = sorted(PROCESSED_DIR.glob("*_facts.json"))
    for facts_file in facts_files:
        with open(facts_file, "r", encoding="utf-8") as f:
            facts_data = json.load(f)
        sqlite_store.upsert_fund_facts(facts_data)
        logger.info(f"Loaded fund facts: {facts_data['scheme_name']}")

    # --- Summary ---
    duration = time.time() - start_time
    stats = {
        "status": "success",
        "total_chunks": len(chunks),
        "embedded": len(chunks_to_embed),
        "skipped_unchanged": skipped,
        "sqlite_inserted": inserted,
        "sqlite_updated": updated,
        "sqlite_unchanged": unchanged,
        "stale_deleted_chroma": stale_chroma,
        "stale_deleted_sqlite": stale_sqlite,
        "chroma_total": chroma_store.count(),
        "sqlite_total": sqlite_store.count(),
        "fund_facts_loaded": len(facts_files),
        "duration_seconds": round(duration, 2),
    }

    sqlite_store.close()

    logger.info(f"\n{'='*60}")
    logger.info(f"EMBEDDING PIPELINE — Complete ({duration:.1f}s)")
    logger.info(f"  ChromaDB: {stats['chroma_total']} vectors")
    logger.info(f"  SQLite:   {stats['sqlite_total']} chunks, {stats['fund_facts_loaded']} fund facts")
    logger.info(f"  Embedded: {stats['embedded']} new/changed, {stats['skipped_unchanged']} unchanged")
    logger.info(f"{'='*60}")

    return stats


# =============================================================================
# CLI Entry Point
# =============================================================================

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    parser = argparse.ArgumentParser(description="Embedder — Generate embeddings and store in ChromaDB + SQLite")
    parser.add_argument("--rebuild", action="store_true", help="Rebuild stores from scratch")
    args = parser.parse_args()

    stats = run_embedding_pipeline(rebuild=args.rebuild)
    print(json.dumps(stats, indent=2))
