"""
Scraping Service — Orchestrates full scrape of all 15 Groww URLs.

This is the main entry point called by the GitHub Actions workflow.
It loads URLs, scrapes each one, performs diff checks, saves raw HTML,
and writes a structured scrape log.

Usage:
    python src/ingestion/scraping_service.py
    python src/ingestion/scraping_service.py --mode full
    python src/ingestion/scraping_service.py --mode single --url <URL>
"""

import argparse
import json
import logging
import sys
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from config import (
    URLS_FILE,
    RAW_DIR,
    SCRAPE_LOG_FILE,
    SCRAPE_DELAY_BETWEEN,
    ensure_directories,
    get_scheme_slug,
)
from src.ingestion.scraper import scrape_url, ScrapeResult

# IST timezone
IST = timezone(timedelta(hours=5, minutes=30))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def load_urls() -> list[dict]:
    """Load the curated list of 15 Groww scheme URLs from urls.json."""
    if not URLS_FILE.exists():
        logger.error(f"URLs file not found: {URLS_FILE}")
        sys.exit(1)

    with open(URLS_FILE, "r", encoding="utf-8") as f:
        urls = json.load(f)

    logger.info(f"Loaded {len(urls)} URLs from {URLS_FILE}")
    return urls


def load_previous_hashes() -> dict[str, str]:
    """
    Load content hashes from the previous scrape run.
    Returns a dict mapping scheme_slug -> content_hash.
    """
    if not SCRAPE_LOG_FILE.exists():
        return {}

    try:
        with open(SCRAPE_LOG_FILE, "r", encoding="utf-8") as f:
            log_data = json.load(f)

        hashes = {}
        for result in log_data.get("results", []):
            slug = get_scheme_slug(result["url"])
            if result.get("content_hash"):
                hashes[slug] = result["content_hash"]

        return hashes
    except (json.JSONDecodeError, KeyError) as e:
        logger.warning(f"Could not parse previous scrape log: {e}")
        return {}


def has_content_changed(new_hash: str, scheme_slug: str, previous_hashes: dict) -> bool:
    """Check if content has changed compared to the previous scrape."""
    previous_hash = previous_hashes.get(scheme_slug)
    if previous_hash is None:
        return True  # First scrape — always treat as changed
    return new_hash != previous_hash


def save_raw_html(result: ScrapeResult, date_str: str) -> Path:
    """
    Save raw HTML to data/raw/<scheme-slug>_<date>.html.
    Returns the path to the saved file.
    """
    filename = f"{result.scheme_slug}_{date_str}.html"
    filepath = RAW_DIR / filename

    with open(filepath, "w", encoding="utf-8") as f:
        f.write(result.html_content)

    logger.info(f"Saved raw HTML: {filepath.name} ({len(result.html_content)} chars)")
    return filepath


def save_scrape_log(
    results: list[dict],
    start_time: datetime,
    duration_seconds: float,
    triggered_by: str = "manual",
) -> None:
    """Write the structured scrape log to data/scrape_log.json."""
    summary = {
        "total": len(results),
        "updated": sum(1 for r in results if r["status"] == "updated"),
        "unchanged": sum(1 for r in results if r["status"] == "unchanged"),
        "failed": sum(1 for r in results if r["status"] == "failed"),
    }

    log_data = {
        "run_id": start_time.astimezone(IST).isoformat(),
        "triggered_by": triggered_by,
        "duration_seconds": round(duration_seconds, 2),
        "results": results,
        "summary": summary,
    }

    with open(SCRAPE_LOG_FILE, "w", encoding="utf-8") as f:
        json.dump(log_data, f, indent=2, ensure_ascii=False)

    logger.info(f"Scrape log saved: {SCRAPE_LOG_FILE}")
    logger.info(
        f"Summary: {summary['total']} total | "
        f"{summary['updated']} updated | "
        f"{summary['unchanged']} unchanged | "
        f"{summary['failed']} failed"
    )


def run_full_scrape(triggered_by: str = "manual") -> dict:
    """
    Main orchestration function.
    Scrapes all 15 Groww URLs, performs diff checks, saves raw HTML, and logs results.

    Returns the scrape log dict.
    """
    logger.info("=" * 60)
    logger.info("SCRAPING SERVICE — Starting full scrape")
    logger.info("=" * 60)

    # Ensure all directories exist
    ensure_directories()

    # Load URL list and previous hashes
    urls = load_urls()
    previous_hashes = load_previous_hashes()
    start_time = datetime.now(timezone.utc)
    date_str = start_time.strftime("%Y-%m-%d")

    results = []
    changed_count = 0

    for i, url_entry in enumerate(urls, 1):
        url = url_entry["url"]
        scheme_name = url_entry.get("scheme_name", "Unknown")
        scheme_slug = get_scheme_slug(url)

        logger.info(f"\n[{i}/{len(urls)}] Scraping: {scheme_name}")
        logger.info(f"  URL: {url}")

        # Scrape the URL
        scrape_result = scrape_url(url)

        if scrape_result.status == "failed":
            # Failed to fetch
            results.append({
                "url": url,
                "scheme": scheme_name,
                "scheme_slug": scheme_slug,
                "status": "failed",
                "error": scrape_result.error,
                "retry_count": scrape_result.retry_count,
                "scrape_time_ms": scrape_result.scrape_time_ms,
                "method": scrape_result.method,
                "content_hash": None,
            })
            logger.error(f"  ✗ FAILED: {scrape_result.error}")
            continue

        # Diff check — has content changed?
        content_changed = has_content_changed(
            scrape_result.content_hash, scheme_slug, previous_hashes
        )

        if content_changed:
            # Content changed — save raw HTML
            save_raw_html(scrape_result, date_str)
            status = "updated"
            changed_count += 1
            logger.info(
                f"  ✓ UPDATED (hash: {scrape_result.content_hash[:12]}...) "
                f"[{scrape_result.scrape_time_ms}ms via {scrape_result.method}]"
            )
            logger.info(
                f"  Sections found: {list(scrape_result.sections.keys())}"
            )
        else:
            status = "unchanged"
            logger.info(
                f"  — UNCHANGED (hash: {scrape_result.content_hash[:12]}...) "
                f"[{scrape_result.scrape_time_ms}ms]"
            )

        results.append({
            "url": url,
            "scheme": scheme_name,
            "scheme_slug": scheme_slug,
            "status": status,
            "content_hash": scrape_result.content_hash,
            "scrape_time_ms": scrape_result.scrape_time_ms,
            "method": scrape_result.method,
            "sections_found": list(scrape_result.sections.keys()),
            "cleaned_text_length": len(scrape_result.cleaned_text or ""),
        })

        # Polite delay between requests
        if i < len(urls):
            time.sleep(SCRAPE_DELAY_BETWEEN)

    # Calculate duration
    duration = (datetime.now(timezone.utc) - start_time).total_seconds()

    # Save scrape log
    save_scrape_log(results, start_time, duration, triggered_by)

    logger.info(f"\n{'=' * 60}")
    logger.info(f"SCRAPING SERVICE — Complete ({duration:.1f}s)")
    logger.info(f"  {changed_count} pages changed out of {len(urls)}")
    logger.info(f"{'=' * 60}")

    return {
        "changed_count": changed_count,
        "total": len(urls),
        "duration_seconds": duration,
    }


def run_single_scrape(url: str) -> ScrapeResult:
    """
    Scrape a single URL (for testing/debugging).
    """
    ensure_directories()
    logger.info(f"Single scrape: {url}")
    result = scrape_url(url)

    if result.status == "success":
        date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        save_raw_html(result, date_str)
        logger.info(f"  ✓ Success — {len(result.cleaned_text)} chars cleaned text")
        logger.info(f"  Sections: {list(result.sections.keys())}")
    else:
        logger.error(f"  ✗ Failed: {result.error}")

    return result


# =============================================================================
# CLI Entry Point
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Scraping Service — Fetch Groww mutual fund pages"
    )
    parser.add_argument(
        "--mode",
        choices=["full", "single"],
        default="full",
        help="Scrape mode: 'full' for all URLs, 'single' for one URL",
    )
    parser.add_argument(
        "--url",
        type=str,
        help="URL to scrape (required for --mode single)",
    )
    parser.add_argument(
        "--triggered-by",
        type=str,
        default="manual",
        choices=["manual", "scheduler", "github_actions"],
        help="Who/what triggered this run (for logging)",
    )

    args = parser.parse_args()

    if args.mode == "single":
        if not args.url:
            parser.error("--url is required when --mode is 'single'")
        run_single_scrape(args.url)
    else:
        # Check if running in GitHub Actions
        import os
        triggered_by = args.triggered_by
        if os.environ.get("GITHUB_ACTIONS") == "true":
            triggered_by = "github_actions"

        result = run_full_scrape(triggered_by=triggered_by)

        # Exit with error if all URLs failed
        if result["changed_count"] == 0 and result["total"] > 0:
            logger.warning("No pages were updated in this run")
