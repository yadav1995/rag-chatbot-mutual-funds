"""
Tests for the Scraping Service and Scraper modules.
"""

import json
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import get_scheme_slug, get_scheme_info, CATEGORY_MAP
from src.ingestion.scraper import (
    compute_content_hash,
    clean_html,
    ScrapeResult,
)
from src.ingestion.scraping_service import (
    has_content_changed,
    load_urls,
)


# =============================================================================
# Config Tests
# =============================================================================


class TestConfig:
    """Tests for config.py helper functions."""

    def test_get_scheme_slug(self):
        url = "https://groww.in/mutual-funds/hdfc-mid-cap-fund-direct-growth"
        assert get_scheme_slug(url) == "hdfc-mid-cap-fund-direct-growth"

    def test_get_scheme_slug_trailing_slash(self):
        url = "https://groww.in/mutual-funds/hdfc-mid-cap-fund-direct-growth/"
        assert get_scheme_slug(url) == "hdfc-mid-cap-fund-direct-growth"

    def test_get_scheme_info_known(self):
        url = "https://groww.in/mutual-funds/hdfc-mid-cap-fund-direct-growth"
        info = get_scheme_info(url)
        assert info["scheme_name"] == "HDFC Mid-Cap Fund"
        assert info["category"] == "mid-cap"
        assert info["amc"] == "HDFC Mutual Fund"

    def test_get_scheme_info_unknown(self):
        url = "https://groww.in/mutual-funds/unknown-fund-direct-growth"
        info = get_scheme_info(url)
        assert info["category"] == "unknown"

    def test_all_15_schemes_mapped(self):
        assert len(CATEGORY_MAP) == 15


# =============================================================================
# Scraper Tests
# =============================================================================


class TestScraper:
    """Tests for scraper.py functions."""

    def test_compute_content_hash(self):
        text = "Hello, world!"
        hash1 = compute_content_hash(text)
        hash2 = compute_content_hash(text)
        assert hash1 == hash2  # Deterministic
        assert len(hash1) == 64  # SHA-256 hex digest

    def test_compute_content_hash_different_text(self):
        hash1 = compute_content_hash("text A")
        hash2 = compute_content_hash("text B")
        assert hash1 != hash2

    def test_clean_html_removes_scripts(self):
        html = """
        <html>
        <head><script>alert('hi')</script></head>
        <body>
            <nav>Navigation</nav>
            <div>Main content here</div>
            <footer>Footer stuff</footer>
        </body>
        </html>
        """
        cleaned = clean_html(html)
        assert "alert" not in cleaned
        assert "Navigation" not in cleaned
        assert "Footer stuff" not in cleaned
        assert "Main content here" in cleaned

    def test_clean_html_removes_cookie_banners(self):
        html = """
        <html><body>
            <div class="cookie-banner">Accept cookies</div>
            <div>Real content</div>
        </body></html>
        """
        cleaned = clean_html(html)
        assert "cookie" not in cleaned.lower()
        assert "Real content" in cleaned

    def test_scrape_result_dataclass(self):
        result = ScrapeResult(
            url="https://example.com",
            scheme_name="Test Fund",
            scheme_slug="test-fund",
            status="success",
        )
        assert result.status == "success"
        assert result.html_content is None
        assert result.error is None
        assert result.sections == {}


# =============================================================================
# Scraping Service Tests
# =============================================================================


class TestScrapingService:
    """Tests for scraping_service.py functions."""

    def test_has_content_changed_new_scheme(self):
        assert has_content_changed("abc123", "new-scheme", {}) is True

    def test_has_content_changed_same_hash(self):
        prev = {"test-scheme": "abc123"}
        assert has_content_changed("abc123", "test-scheme", prev) is False

    def test_has_content_changed_different_hash(self):
        prev = {"test-scheme": "abc123"}
        assert has_content_changed("def456", "test-scheme", prev) is True

    def test_load_urls(self):
        urls = load_urls()
        assert len(urls) == 15
        assert all("url" in entry for entry in urls)
        assert all("scheme_name" in entry for entry in urls)

    def test_urls_are_valid_groww_links(self):
        urls = load_urls()
        for entry in urls:
            assert entry["url"].startswith("https://groww.in/mutual-funds/")
            assert "hdfc" in entry["url"].lower()
