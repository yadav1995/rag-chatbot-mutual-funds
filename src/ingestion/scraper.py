"""
Scraper Module — Individual URL Fetch & HTML Parse

Handles fetching a single Groww mutual fund page, validating the response,
and falling back to Selenium if requests returns incomplete content.
"""

import hashlib
import logging
import time
from dataclasses import dataclass, field
from typing import Optional

import requests
from bs4 import BeautifulSoup

# Add project root to path for imports
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from config import (
    SCRAPE_HEADERS,
    SCRAPE_TIMEOUT,
    SCRAPE_RETRY_COUNT,
    SCRAPE_RETRY_DELAY,
    MIN_CONTENT_LENGTH,
    SELENIUM_HEADLESS,
    SELENIUM_PAGE_LOAD_TIMEOUT,
    SELENIUM_WAIT_AFTER_LOAD,
    get_scheme_info,
    get_scheme_slug,
)

logger = logging.getLogger(__name__)


@dataclass
class ScrapeResult:
    """Result of scraping a single URL."""

    url: str
    scheme_name: str
    scheme_slug: str
    status: str  # "success", "failed", "skipped"
    html_content: Optional[str] = None
    cleaned_text: Optional[str] = None
    content_hash: Optional[str] = None
    scrape_time_ms: int = 0
    error: Optional[str] = None
    retry_count: int = 0
    method: str = "requests"  # "requests" or "selenium"
    sections: dict = field(default_factory=dict)


def fetch_with_requests(url: str) -> Optional[str]:
    """
    Fetch a URL using requests library.
    Returns HTML content as string, or None if failed.
    """
    try:
        response = requests.get(
            url,
            headers=SCRAPE_HEADERS,
            timeout=SCRAPE_TIMEOUT,
        )
        response.raise_for_status()

        if len(response.text) < MIN_CONTENT_LENGTH:
            logger.warning(
                f"Content too short ({len(response.text)} chars) for {url}, "
                f"likely JS-rendered page"
            )
            return None

        return response.text

    except requests.exceptions.HTTPError as e:
        logger.error(f"HTTP error for {url}: {e}")
        raise
    except requests.exceptions.Timeout:
        logger.error(f"Timeout fetching {url}")
        raise
    except requests.exceptions.RequestException as e:
        logger.error(f"Request failed for {url}: {e}")
        raise


def fetch_with_selenium(url: str) -> Optional[str]:
    """
    Fetch a URL using Selenium with headless Chrome.
    Fallback for JS-rendered Groww pages.
    Returns HTML content as string, or None if failed.
    """
    try:
        from selenium import webdriver
        from selenium.webdriver.chrome.options import Options
        from selenium.webdriver.chrome.service import Service

        options = Options()
        if SELENIUM_HEADLESS:
            options.add_argument("--headless=new")
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")
        options.add_argument("--disable-gpu")
        options.add_argument(f"user-agent={SCRAPE_HEADERS['User-Agent']}")

        driver = webdriver.Chrome(options=options)
        driver.set_page_load_timeout(SELENIUM_PAGE_LOAD_TIMEOUT)

        try:
            driver.get(url)
            # Wait for JS rendering
            time.sleep(SELENIUM_WAIT_AFTER_LOAD)
            html = driver.page_source

            if len(html) < MIN_CONTENT_LENGTH:
                logger.warning(
                    f"Selenium also returned short content ({len(html)} chars) for {url}"
                )
                return None

            return html
        finally:
            driver.quit()

    except ImportError:
        logger.error("Selenium not installed. Cannot use fallback scraper.")
        return None
    except Exception as e:
        logger.error(f"Selenium failed for {url}: {e}")
        return None


def compute_content_hash(text: str) -> str:
    """Compute SHA-256 hash of content for change detection."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def clean_html(html: str) -> str:
    """
    Clean raw HTML: remove scripts, styles, nav, footer,
    and extract main content text.
    """
    soup = BeautifulSoup(html, "html.parser")

    # Remove unwanted elements
    for tag in soup.find_all(["script", "style", "nav", "footer", "header", "noscript"]):
        tag.decompose()

    # Remove cookie banners, ad containers, and other noise
    for selector in [
        "[class*='cookie']",
        "[class*='banner']",
        "[class*='popup']",
        "[class*='modal']",
        "[class*='advertisement']",
        "[id*='cookie']",
        "[id*='banner']",
    ]:
        for el in soup.select(selector):
            el.decompose()

    # Get text with newlines separating blocks
    text = soup.get_text(separator="\n", strip=True)

    # Normalize whitespace: collapse multiple blank lines
    lines = [line.strip() for line in text.splitlines()]
    lines = [line for line in lines if line]  # Remove empty lines
    cleaned = "\n".join(lines)

    return cleaned


def extract_sections(html: str, url: str) -> dict:
    """
    Extract structured sections from a Groww mutual fund page.
    Returns a dict mapping section names to their text content.
    """
    soup = BeautifulSoup(html, "html.parser")
    sections = {}

    # --- Strategy: Try multiple selector patterns per section ---
    # Groww uses React, so class names may be hashed. We use heuristic patterns.

    # 1. Try to get the full page text as a fallback
    main_content = soup.find("main") or soup.find("div", {"id": "root"}) or soup
    full_text = main_content.get_text(separator="\n", strip=True) if main_content else ""

    # 2. Look for common section patterns
    # Key Information / Fund Details sections
    for heading_text in [
        "Key Information",
        "Fund Details",
        "Scheme Details",
        "Fund Overview",
        "Fund Manager",
        "Asset Allocation",
        "Tax Implications",
        "Top Holdings",
        "Investment Objective",
        "Exit Load",
        "Riskometer",
    ]:
        heading = soup.find(
            lambda tag: tag.name in ["h1", "h2", "h3", "h4", "div", "span"]
            and heading_text.lower() in (tag.get_text() or "").lower()
        )
        if heading:
            # Get the parent container or the next sibling content
            parent = heading.find_parent(["div", "section"])
            if parent:
                section_text = parent.get_text(separator="\n", strip=True)
                section_key = heading_text.lower().replace(" ", "_")
                sections[section_key] = section_text

    # 3. If we found very few sections, store the full text as "full_page"
    if len(sections) < 3 and full_text:
        sections["full_page"] = full_text
        logger.info(
            f"Found only {len(sections)} structured sections for {url}, "
            f"storing full page text as fallback"
        )

    return sections


def scrape_url(url: str) -> ScrapeResult:
    """
    Scrape a single Groww mutual fund URL.

    Flow:
    1. Try requests first
    2. If content is too short (JS-rendered), fall back to Selenium
    3. Clean HTML and extract sections
    4. Compute content hash for change detection

    Returns a ScrapeResult with all details.
    """
    scheme_info = get_scheme_info(url)
    start_time = time.time()
    retry_count = 0
    html_content = None
    method = "requests"
    last_error = None

    # --- Attempt with requests (+ retries) ---
    for attempt in range(1, SCRAPE_RETRY_COUNT + 1):
        try:
            html_content = fetch_with_requests(url)
            if html_content:
                break
            # Content too short — will try Selenium below
            break
        except Exception as e:
            retry_count = attempt
            last_error = str(e)
            if attempt < SCRAPE_RETRY_COUNT:
                logger.info(
                    f"Retry {attempt}/{SCRAPE_RETRY_COUNT} for {url} "
                    f"in {SCRAPE_RETRY_DELAY}s..."
                )
                time.sleep(SCRAPE_RETRY_DELAY)

    # --- Fallback to Selenium if requests failed or returned short content ---
    if not html_content:
        logger.info(f"Falling back to Selenium for {url}")
        method = "selenium"
        html_content = fetch_with_selenium(url)

    # --- Calculate elapsed time ---
    elapsed_ms = int((time.time() - start_time) * 1000)

    # --- Handle complete failure ---
    if not html_content:
        return ScrapeResult(
            url=url,
            scheme_name=scheme_info["scheme_name"],
            scheme_slug=scheme_info["scheme_slug"],
            status="failed",
            scrape_time_ms=elapsed_ms,
            error=last_error or "No content retrieved (both requests and Selenium failed)",
            retry_count=retry_count,
            method=method,
        )

    # --- Clean and extract ---
    cleaned_text = clean_html(html_content)
    content_hash = compute_content_hash(cleaned_text)
    sections = extract_sections(html_content, url)

    return ScrapeResult(
        url=url,
        scheme_name=scheme_info["scheme_name"],
        scheme_slug=scheme_info["scheme_slug"],
        status="success",
        html_content=html_content,
        cleaned_text=cleaned_text,
        content_hash=content_hash,
        scrape_time_ms=elapsed_ms,
        retry_count=retry_count,
        method=method,
        sections=sections,
    )
