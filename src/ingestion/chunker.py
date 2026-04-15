"""
Chunker Module — Two-Tier Chunking + Metadata Enrichment

Implements the chunking pipeline as specified in ChunkingEmbeddingArchitecture.md §3:
- Tier 1: Section-level splitting (each section = one chunk if ≤ 512 tokens)
- Tier 2: Token-level splitting (RecursiveCharacterTextSplitter for oversized sections)
- Metadata enrichment with scheme, section, source, and hash info

Usage:
    python src/ingestion/chunker.py                    # Process all raw HTML files
    python src/ingestion/chunker.py --file <path> --url <url>  # Process a single file
"""

import argparse
import hashlib
import json
import logging
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from config import (
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    MIN_CHUNK_SIZE,
    CHUNK_SEPARATORS,
    RAW_DIR,
    PROCESSED_DIR,
    URLS_FILE,
    AMC_NAME,
    ensure_directories,
    get_scheme_slug,
    get_scheme_info,
)
from src.ingestion.parser import parse_groww_page, parse_raw_html_file, ParsedSection, ParsedPage

logger = logging.getLogger(__name__)

# IST timezone
IST = timezone(timedelta(hours=5, minutes=30))


# =============================================================================
# Token Counter — using tiktoken for accurate OpenAI-compatible token counts
# =============================================================================

def _get_token_counter():
    """Get the token counting function. Falls back to word-based estimate."""
    try:
        import tiktoken
        encoding = tiktoken.get_encoding("cl100k_base")
        return lambda text: len(encoding.encode(text))
    except ImportError:
        logger.warning("tiktoken not installed, using word-based token estimation")
        return lambda text: len(text.split())


token_count_fn = _get_token_counter()


# =============================================================================
# Text Splitter — for Tier 2 sub-chunking
# =============================================================================

def _get_text_splitter():
    """Get the text splitter. Falls back to a simple split if langchain not installed."""
    try:
        from langchain_text_splitters import RecursiveCharacterTextSplitter

        return RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            length_function=token_count_fn,
            separators=CHUNK_SEPARATORS,
        )
    except ImportError:
        logger.warning("langchain-text-splitters not installed, using simple splitter")
        return None


text_splitter = _get_text_splitter()


def _simple_split(text: str, max_tokens: int = CHUNK_SIZE) -> list[str]:
    """Fallback simple splitter when langchain is not available."""
    words = text.split()
    chunks = []
    current_chunk = []
    current_count = 0

    for word in words:
        current_chunk.append(word)
        current_count += 1
        if current_count >= max_tokens:
            chunks.append(" ".join(current_chunk))
            # Keep overlap
            overlap_words = current_chunk[-CHUNK_OVERLAP:] if CHUNK_OVERLAP > 0 else []
            current_chunk = overlap_words
            current_count = len(overlap_words)

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks


# =============================================================================
# Metadata Builder
# =============================================================================

def _compute_chunk_hash(text: str) -> str:
    """Compute SHA-256 hash for a chunk's text content."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _build_chunk_id(scheme_slug: str, section_name: str, chunk_index: int) -> str:
    """Build a unique chunk ID: {scheme_slug}-{section}-{index}."""
    return f"{scheme_slug}-{section_name}-{chunk_index}"


def _build_metadata(
    section: ParsedSection,
    chunk_index: int,
    total_chunks: int,
    token_count: int,
    content_hash: str,
    scrape_date: str,
    category: str,
) -> dict:
    """Build the full metadata dict for a chunk."""
    chunk_id = _build_chunk_id(section.scheme_slug, section.section_name, chunk_index)

    return {
        "chunk_id": chunk_id,
        "scheme_name": section.scheme_name,
        "scheme_slug": section.scheme_slug,
        "amc": AMC_NAME,
        "category": category,
        "doc_type": "groww_page",
        "section": section.section_name,
        "source_url": section.source_url,
        "scrape_date": scrape_date,
        "chunk_index": chunk_index,
        "total_chunks": total_chunks,
        "token_count": token_count,
        "data_points": ",".join(section.data_points),  # ChromaDB requires str values
        "content_hash": content_hash,
    }


# =============================================================================
# Core Chunking Logic
# =============================================================================

def chunk_parsed_page(parsed_page: ParsedPage, scrape_date: str = None) -> list[dict]:
    """
    Apply two-tier chunking to all sections of a parsed Groww page.

    Tier 1: Each section is a chunk (if ≤ CHUNK_SIZE tokens)
    Tier 2: Split large sections into sub-chunks using RecursiveCharacterTextSplitter

    Returns a list of chunk dicts: {"text": str, "metadata": dict}
    """
    if scrape_date is None:
        scrape_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    chunks = []
    section_counters = {}

    for section in parsed_page.sections:
        text = section.raw_text.strip()
        if not text:
            continue

        current_idx = section_counters.get(section.section_name, 0)
        tokens = token_count_fn(text)

        if tokens <= CHUNK_SIZE:
            # Tier 1 — section fits in one chunk
            if tokens >= MIN_CHUNK_SIZE:
                content_hash = _compute_chunk_hash(text)
                metadata = _build_metadata(
                    section=section,
                    chunk_index=current_idx,
                    total_chunks=1,
                    token_count=tokens,
                    content_hash=content_hash,
                    scrape_date=scrape_date,
                    category=parsed_page.category,
                )
                chunks.append({"text": text, "metadata": metadata})
                logger.debug(
                    f"  Tier 1 chunk: {metadata['chunk_id']} ({tokens} tokens)"
                )
                current_idx += 1
            else:
                logger.debug(
                    f"  Skipped section '{section.section_name}' — "
                    f"too small ({tokens} tokens < {MIN_CHUNK_SIZE})"
                )
        else:
            # Tier 2 — section too large, needs sub-chunking
            if text_splitter:
                sub_texts = text_splitter.split_text(text)
            else:
                sub_texts = _simple_split(text)

            valid_sub_texts = [
                st for st in sub_texts if token_count_fn(st) >= MIN_CHUNK_SIZE
            ]

            for sub_text in valid_sub_texts:
                sub_tokens = token_count_fn(sub_text)
                content_hash = _compute_chunk_hash(sub_text)
                metadata = _build_metadata(
                    section=section,
                    chunk_index=current_idx,
                    total_chunks=len(valid_sub_texts),
                    token_count=sub_tokens,
                    content_hash=content_hash,
                    scrape_date=scrape_date,
                    category=parsed_page.category,
                )
                chunks.append({"text": sub_text, "metadata": metadata})
                logger.debug(
                    f"  Tier 2 chunk: {metadata['chunk_id']} ({sub_tokens} tokens)"
                )
                current_idx += 1

            logger.info(
                f"  Section '{section.section_name}' split into "
                f"{len(valid_sub_texts)} sub-chunks (was {tokens} tokens)"
            )
            
        section_counters[section.section_name] = current_idx

    logger.info(
        f"Chunked {parsed_page.scheme_name}: "
        f"{len(parsed_page.sections)} sections → {len(chunks)} chunks"
    )

    return chunks


# =============================================================================
# Batch Chunking — Process all scraped pages
# =============================================================================

def chunk_all_raw_files() -> list[dict]:
    """
    Process all raw HTML files in data/raw/ through the parse → chunk pipeline.
    Returns all chunks from all pages.
    """
    ensure_directories()

    # Load URL mapping
    with open(URLS_FILE, "r", encoding="utf-8") as f:
        url_entries = json.load(f)

    url_map = {get_scheme_slug(entry["url"]): entry["url"] for entry in url_entries}

    # Find all HTML files in raw/
    all_html_files = sorted(RAW_DIR.glob("*.html"))
    if not all_html_files:
        logger.warning(f"No HTML files found in {RAW_DIR}")
        return []

    # Filter to only keep the latest file per scheme
    latest_files = {}
    for filepath in all_html_files:
        filename_parts = filepath.stem.rsplit("_", 1)
        scheme_slug = filename_parts[0] if len(filename_parts) >= 2 else filepath.stem
        # Since files are sorted chronologically by name, the last seen is the latest
        latest_files[scheme_slug] = filepath

    html_files = list(latest_files.values())

    all_chunks = []
    scrape_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    logger.info(f"Processing {len(html_files)} raw HTML files...")

    for filepath in html_files:
        # Extract scheme slug from filename: {slug}_{date}.html
        filename_parts = filepath.stem.rsplit("_", 1)
        scheme_slug = filename_parts[0] if len(filename_parts) >= 2 else filepath.stem

        # Find the corresponding URL
        url = url_map.get(scheme_slug)
        if not url:
            logger.warning(f"No URL mapping found for {scheme_slug}, skipping {filepath.name}")
            continue

        # Extract scrape date from filename
        if len(filename_parts) >= 2:
            scrape_date = filename_parts[1]

        logger.info(f"\n--- Processing: {filepath.name} ---")

        # Parse HTML
        parsed_page = parse_raw_html_file(str(filepath), url)

        # Chunk
        chunks = chunk_parsed_page(parsed_page, scrape_date=scrape_date)
        all_chunks.extend(chunks)

        # Save parsed fund facts to processed/
        if parsed_page.fund_facts:
            facts_file = PROCESSED_DIR / f"{scheme_slug}_facts.json"
            facts_data = {
                "scheme_name": parsed_page.scheme_name,
                "scheme_slug": parsed_page.scheme_slug,
                "category": parsed_page.category,
                "source_url": url,
                "scrape_date": scrape_date,
                "fund_facts": parsed_page.fund_facts,
            }
            with open(facts_file, "w", encoding="utf-8") as f:
                json.dump(facts_data, f, indent=2, ensure_ascii=False)
            logger.info(f"  Saved fund facts: {facts_file.name}")

    # Save all chunks to processed/
    if all_chunks:
        chunks_file = PROCESSED_DIR / "all_chunks.json"
        # Convert for JSON serialization
        chunks_for_json = [
            {"text": c["text"], "metadata": c["metadata"]} for c in all_chunks
        ]
        with open(chunks_file, "w", encoding="utf-8") as f:
            json.dump(chunks_for_json, f, indent=2, ensure_ascii=False)
        logger.info(f"\nSaved {len(all_chunks)} total chunks to {chunks_file}")

    logger.info(f"\n{'='*60}")
    logger.info(f"CHUNKING COMPLETE: {len(all_chunks)} chunks from {len(html_files)} files")
    logger.info(f"{'='*60}")

    return all_chunks


# =============================================================================
# CLI Entry Point
# =============================================================================

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    parser = argparse.ArgumentParser(description="Chunker — Parse and chunk Groww HTML")
    parser.add_argument("--file", type=str, help="Path to a single raw HTML file")
    parser.add_argument("--url", type=str, help="Source URL for the file (required with --file)")
    args = parser.parse_args()

    if args.file:
        if not args.url:
            parser.error("--url is required when using --file")
        parsed = parse_raw_html_file(args.file, args.url)
        chunks = chunk_parsed_page(parsed)
        print(f"\nChunks generated: {len(chunks)}")
        for c in chunks:
            print(f"  [{c['metadata']['chunk_id']}] ({c['metadata']['token_count']} tokens) "
                  f"section={c['metadata']['section']}")
    else:
        all_chunks = chunk_all_raw_files()
