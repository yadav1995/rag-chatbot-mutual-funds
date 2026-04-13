"""
Tests for the Chunking & Embedding Pipeline (Phase 2).
Tests parser, chunker, and embedder modules.
"""

import json
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import (
    RAW_DIR, PROCESSED_DIR, VECTORSTORE_DIR, CHUNKS_DB_FILE,
    CHUNK_SIZE, MIN_CHUNK_SIZE, get_scheme_slug, get_scheme_info,
)
from src.ingestion.parser import (
    parse_groww_page, ParsedSection, ParsedPage, _clean_text,
)
from src.ingestion.chunker import (
    chunk_parsed_page, _compute_chunk_hash, _build_chunk_id,
    _build_metadata, token_count_fn,
)


# =============================================================================
# Parser Tests
# =============================================================================


class TestParser:
    """Tests for parser.py functions."""

    def test_clean_text_removes_react_comments(self):
        text = "Hello<!-- --> World<!-- -->"
        assert _clean_text(text) == "Hello World"

    def test_clean_text_normalizes_whitespace(self):
        text = "  Line 1  \n\n  \n  Line 2  \n  "
        cleaned = _clean_text(text)
        assert "Line 1" in cleaned
        assert "Line 2" in cleaned

    def test_clean_text_empty(self):
        assert _clean_text("") == ""
        assert _clean_text(None) == ""

    def test_parse_groww_page_returns_parsed_page(self):
        """Test parsing with a minimal HTML structure."""
        html = """
        <html><body>
        <div id="root">
            <h1 class="header_schemeName__test">Test Fund Direct Growth</h1>
            <div class="pills_container__test">
                <span>Equity</span><span>Mid Cap</span>
            </div>
        </div>
        </body></html>
        """
        result = parse_groww_page(html, "https://groww.in/mutual-funds/hdfc-mid-cap-fund-direct-growth")
        assert isinstance(result, ParsedPage)
        assert result.scheme_name == "HDFC Mid-Cap Fund"
        assert result.scheme_slug == "hdfc-mid-cap-fund-direct-growth"

    def test_parse_groww_page_extracts_fund_details(self):
        """Test extraction of the fund_details section (NAV, SIP, AUM, etc.)."""
        html = """
        <html><body>
        <div class="fundDetails_fundDetailsContainer__xyz">
            <div class="flex flex-column fundDetails_gap4__abc">
                <div class="valign-wrapper bodyLarge contentTertiary fundDetails_gap4__abc">NAV: 10 Apr '26</div>
                <div class="bodyXLargeHeavy contentPrimary valign-wrapper">₹215.55</div>
            </div>
            <div class="flex flex-column fundDetails_gap4__abc">
                <div class="valign-wrapper bodyLarge contentTertiary fundDetails_gap4__abc">Min. for SIP</div>
                <div class="bodyXLargeHeavy contentPrimary valign-wrapper">₹100</div>
            </div>
            <div class="flex flex-column fundDetails_gap4__abc">
                <div class="valign-wrapper bodyLarge contentTertiary fundDetails_gap4__abc">Fund size (AUM)</div>
                <div class="bodyXLargeHeavy contentPrimary valign-wrapper">₹85,357.92 Cr</div>
            </div>
            <div class="flex flex-column fundDetails_gap4__abc">
                <div class="valign-wrapper bodyLarge contentTertiary fundDetails_gap4__abc">Expense ratio</div>
                <div class="bodyXLargeHeavy contentPrimary valign-wrapper">0.77%</div>
            </div>
            <div class="flex flex-column fundDetails_gap4__abc">
                <div class="valign-wrapper bodyLarge contentTertiary fundDetails_gap4__abc">Rating</div>
                <div class="bodyXLargeHeavy contentPrimary valign-wrapper">5</div>
            </div>
        </div>
        </body></html>
        """
        result = parse_groww_page(html, "https://groww.in/mutual-funds/hdfc-mid-cap-fund-direct-growth")

        # Find the fund_details section
        fd_sections = [s for s in result.sections if s.section_name == "fund_details"]
        assert len(fd_sections) == 1

        fd = fd_sections[0]
        assert "nav" in fd.data_points
        assert "min_sip" in fd.data_points
        assert "fund_size" in fd.data_points
        assert "expense_ratio" in fd.data_points
        assert "rating" in fd.data_points

        # Check fund facts are extracted
        assert "nav" in result.fund_facts
        assert "min_sip" in result.fund_facts

    def test_parse_with_real_html_file(self):
        """Test parsing with the actual scraped HTML file (if it exists)."""
        html_file = RAW_DIR / "hdfc-mid-cap-fund-direct-growth_2026-04-13.html"
        if not html_file.exists():
            pytest.skip("Real HTML file not available")

        from src.ingestion.parser import parse_raw_html_file
        result = parse_raw_html_file(
            str(html_file),
            "https://groww.in/mutual-funds/hdfc-mid-cap-fund-direct-growth"
        )

        assert result.scheme_name == "HDFC Mid-Cap Fund"
        assert len(result.sections) >= 5  # Should extract most sections
        assert len(result.parse_warnings) < 5  # Most sections should be found

        # Verify key fund facts are extracted
        assert "nav" in result.fund_facts
        assert "expense_ratio" in result.fund_facts


# =============================================================================
# Chunker Tests
# =============================================================================


class TestChunker:
    """Tests for chunker.py functions."""

    def test_compute_chunk_hash_deterministic(self):
        hash1 = _compute_chunk_hash("test text")
        hash2 = _compute_chunk_hash("test text")
        assert hash1 == hash2
        assert len(hash1) == 64  # SHA-256

    def test_compute_chunk_hash_different_text(self):
        assert _compute_chunk_hash("text A") != _compute_chunk_hash("text B")

    def test_build_chunk_id(self):
        chunk_id = _build_chunk_id("hdfc-mid-cap", "fund_details", 0)
        assert chunk_id == "hdfc-mid-cap-fund_details-0"

    def test_build_chunk_id_with_index(self):
        chunk_id = _build_chunk_id("hdfc-mid-cap", "holdings", 2)
        assert chunk_id == "hdfc-mid-cap-holdings-2"

    def test_token_count_fn(self):
        """Token counter should return a positive integer."""
        count = token_count_fn("Hello world, this is a test sentence.")
        assert count > 0
        assert isinstance(count, int)

    def test_chunk_parsed_page_tier1(self):
        """Test Tier 1 chunking: short sections stay as single chunks."""
        # Text must be above MIN_CHUNK_SIZE (50 tokens) to be kept
        text = (
            "Test Fund Direct Growth — Fund Details\n"
            "NAV: ₹215.55 as of 10 Apr 2026\n"
            "Minimum SIP Investment: ₹100 per month\n"
            "Fund Size (AUM): ₹85,357.92 Crore\n"
            "Expense Ratio: 0.77% per annum\n"
            "Rating: 5 stars out of 5 by Groww\n"
            "Category: Equity Mid Cap\n"
            "Risk Level: Very High Risk\n"
            "Exit Load: 1% if redeemed within 1 year from the date of allotment\n"
            "Stamp Duty: 0.005% from July 1st 2020"
        )
        page = ParsedPage(
            url="https://groww.in/mutual-funds/test-fund",
            scheme_name="Test Fund",
            scheme_slug="test-fund",
            category="mid-cap",
            sections=[
                ParsedSection(
                    section_name="fund_details",
                    raw_text=text,
                    data_points=["nav", "min_sip", "fund_size", "expense_ratio", "rating"],
                    source_url="https://groww.in/mutual-funds/test-fund",
                    scheme_name="Test Fund",
                    scheme_slug="test-fund",
                ),
            ],
        )
        chunks = chunk_parsed_page(page, scrape_date="2026-04-13")

        assert len(chunks) == 1
        assert chunks[0]["metadata"]["section"] == "fund_details"
        assert chunks[0]["metadata"]["chunk_index"] == 0
        assert chunks[0]["metadata"]["total_chunks"] == 1
        assert chunks[0]["metadata"]["scrape_date"] == "2026-04-13"

    def test_chunk_parsed_page_skips_small_sections(self):
        """Test that chunks below MIN_CHUNK_SIZE are discarded."""
        page = ParsedPage(
            url="https://groww.in/mutual-funds/test-fund",
            scheme_name="Test Fund",
            scheme_slug="test-fund",
            category="mid-cap",
            sections=[
                ParsedSection(
                    section_name="tiny_section",
                    raw_text="Just a few words",  # Very short
                    data_points=[],
                    source_url="https://groww.in/mutual-funds/test-fund",
                    scheme_name="Test Fund",
                    scheme_slug="test-fund",
                ),
            ],
        )
        chunks = chunk_parsed_page(page)
        # Should be empty because the text is too short
        assert len(chunks) == 0

    def test_chunk_parsed_page_tier2(self):
        """Test Tier 2 chunking: large sections get split into sub-chunks."""
        # Create a long text that exceeds 512 tokens
        long_text = "Test Fund — Fund Manager Details\n" + "\n".join(
            [f"Manager {i} has extensive experience in portfolio management and has been with "
             f"the firm since 200{i%10}. Education: MBA from IIM. Experience: {10+i} years in "
             f"financial services, previously at Goldman Sachs. Specialization: mid-cap equity."
             for i in range(20)]
        )

        page = ParsedPage(
            url="https://groww.in/mutual-funds/test-fund",
            scheme_name="Test Fund",
            scheme_slug="test-fund",
            category="mid-cap",
            sections=[
                ParsedSection(
                    section_name="fund_manager",
                    raw_text=long_text,
                    data_points=["fund_manager_name"],
                    source_url="https://groww.in/mutual-funds/test-fund",
                    scheme_name="Test Fund",
                    scheme_slug="test-fund",
                ),
            ],
        )
        chunks = chunk_parsed_page(page)

        # Should be split into multiple chunks
        assert len(chunks) > 1
        # All chunks should reference the same section
        for c in chunks:
            assert c["metadata"]["section"] == "fund_manager"
        # Total_chunks should be consistent
        total = chunks[0]["metadata"]["total_chunks"]
        assert total == len(chunks)

    def test_chunk_metadata_fields(self):
        """Test that all required metadata fields are present."""
        page = ParsedPage(
            url="https://groww.in/mutual-funds/hdfc-mid-cap-fund-direct-growth",
            scheme_name="HDFC Mid-Cap Fund",
            scheme_slug="hdfc-mid-cap-fund-direct-growth",
            category="mid-cap",
            sections=[
                ParsedSection(
                    section_name="fund_details",
                    raw_text="HDFC Mid-Cap Fund — Fund Details\nNAV: ₹215.55 as of 10 Apr 2026\nMinimum SIP: ₹100\nFund Size AUM: ₹85,357.92 Cr\nExpense Ratio: 0.77%\nRating: 5 stars out of 5",
                    data_points=["nav", "min_sip"],
                    source_url="https://groww.in/mutual-funds/hdfc-mid-cap-fund-direct-growth",
                    scheme_name="HDFC Mid-Cap Fund",
                    scheme_slug="hdfc-mid-cap-fund-direct-growth",
                ),
            ],
        )
        chunks = chunk_parsed_page(page, scrape_date="2026-04-13")
        assert len(chunks) == 1

        meta = chunks[0]["metadata"]
        required_fields = [
            "chunk_id", "scheme_name", "scheme_slug", "amc", "category",
            "doc_type", "section", "source_url", "scrape_date",
            "chunk_index", "total_chunks", "token_count", "data_points",
            "content_hash",
        ]
        for field in required_fields:
            assert field in meta, f"Missing metadata field: {field}"

        assert meta["amc"] == "HDFC Mutual Fund"
        assert meta["doc_type"] == "groww_page"
        assert meta["category"] == "mid-cap"

    def test_chunk_all_raw_files_with_real_data(self):
        """Integration test: chunk all raw HTML files (if available)."""
        html_files = list(RAW_DIR.glob("*.html"))
        if not html_files:
            pytest.skip("No raw HTML files available")

        from src.ingestion.chunker import chunk_all_raw_files
        chunks = chunk_all_raw_files()

        assert len(chunks) > 0
        # Check all chunks have valid metadata
        for c in chunks:
            assert "text" in c
            assert "metadata" in c
            assert len(c["text"]) > 0
            assert c["metadata"]["token_count"] >= MIN_CHUNK_SIZE

        # Check processed files were created
        assert (PROCESSED_DIR / "all_chunks.json").exists()


# =============================================================================
# Embedder Tests (lightweight — no model loading)
# =============================================================================


class TestEmbedderStores:
    """Tests for SQLiteStore (does not require GPU or model download)."""

    def test_sqlite_store_create_and_insert(self):
        """Test SQLite store creation and chunk insertion."""
        from src.ingestion.embedder import SQLiteStore

        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            store = SQLiteStore(db_path=tmp_path)

            chunks = [{
                "text": "NAV: ₹215.55, Fund Size: ₹85,357 Cr",
                "metadata": {
                    "chunk_id": "test-fund-details-0",
                    "scheme_name": "Test Fund",
                    "scheme_slug": "test-fund",
                    "amc": "HDFC Mutual Fund",
                    "category": "mid-cap",
                    "section": "fund_details",
                    "source_url": "https://groww.in/mutual-funds/test-fund",
                    "scrape_date": "2026-04-13",
                    "token_count": 50,
                    "content_hash": "abc123",
                    "data_points": "nav,fund_size",
                },
            }]

            inserted, updated, skipped = store.upsert_chunks(chunks)
            assert inserted == 1
            assert updated == 0
            assert skipped == 0
            assert store.count() == 1

            # Re-insert same chunk — should skip
            inserted, updated, skipped = store.upsert_chunks(chunks)
            assert inserted == 0
            assert skipped == 1

            # Update content
            chunks[0]["metadata"]["content_hash"] = "def456"
            inserted, updated, skipped = store.upsert_chunks(chunks)
            assert updated == 1

            store.close()
        finally:
            os.unlink(tmp_path)

    def test_sqlite_store_fund_facts(self):
        """Test fund facts insertion."""
        from src.ingestion.embedder import SQLiteStore

        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            store = SQLiteStore(db_path=tmp_path)

            facts = {
                "scheme_slug": "test-fund",
                "scheme_name": "Test Fund",
                "category": "mid-cap",
                "source_url": "https://groww.in/mutual-funds/test-fund",
                "scrape_date": "2026-04-13",
                "fund_facts": {
                    "nav": "₹215.55",
                    "nav_date": "10 Apr '26",
                    "min_sip": "₹100",
                    "fund_size": "₹85,357.92 Cr",
                    "expense_ratio": "0.77%",
                    "rating": "5",
                },
            }
            store.upsert_fund_facts(facts)

            # Verify
            row = store.conn.execute(
                "SELECT nav, expense_ratio, rating FROM fund_facts WHERE scheme_slug = ?",
                ("test-fund",),
            ).fetchone()
            assert row is not None
            assert row[0] == "₹215.55"
            assert row[1] == "0.77%"
            assert row[2] == "5"

            store.close()
        finally:
            os.unlink(tmp_path)

    def test_sqlite_store_stale_deletion(self):
        """Test deletion of stale chunks."""
        from src.ingestion.embedder import SQLiteStore

        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            store = SQLiteStore(db_path=tmp_path)

            chunks = [
                {
                    "text": "Chunk A",
                    "metadata": {
                        "chunk_id": "fund-a-0",
                        "scheme_name": "Fund A", "scheme_slug": "fund-a",
                        "amc": "HDFC MF", "category": "mid-cap",
                        "section": "details", "source_url": "https://example.com",
                        "scrape_date": "2026-04-13", "token_count": 50,
                        "content_hash": "hash_a", "data_points": "nav",
                    },
                },
                {
                    "text": "Chunk B",
                    "metadata": {
                        "chunk_id": "fund-b-0",
                        "scheme_name": "Fund B", "scheme_slug": "fund-b",
                        "amc": "HDFC MF", "category": "large-cap",
                        "section": "details", "source_url": "https://example.com",
                        "scrape_date": "2026-04-13", "token_count": 50,
                        "content_hash": "hash_b", "data_points": "nav",
                    },
                },
            ]
            store.upsert_chunks(chunks)
            assert store.count() == 2

            # Delete fund-b (only fund-a is valid now)
            deleted = store.delete_stale({"fund-a-0"})
            assert deleted == 1
            assert store.count() == 1

            store.close()
        finally:
            os.unlink(tmp_path)
