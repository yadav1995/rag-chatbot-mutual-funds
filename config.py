"""
Central Configuration — Mutual Fund FAQ Assistant
All constants, paths, and settings used across the project.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env file from project root (if it exists)
load_dotenv()

# =============================================================================
# Project Paths
# =============================================================================

# Root of the project (parent of 'src/')
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
VECTORSTORE_DIR = DATA_DIR / "vectorstore"
URLS_FILE = DATA_DIR / "urls.json"
SCRAPE_LOG_FILE = DATA_DIR / "scrape_log.json"
CHUNKS_DB_FILE = DATA_DIR / "chunks.db"

# =============================================================================
# Scraping Configuration
# =============================================================================

SCRAPE_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
}

SCRAPE_TIMEOUT = 30          # seconds per request
SCRAPE_RETRY_COUNT = 3       # max retries per URL
SCRAPE_RETRY_DELAY = 30      # seconds between retries
SCRAPE_DELAY_BETWEEN = 2     # seconds between URLs (polite scraping)
MIN_CONTENT_LENGTH = 500     # minimum HTML chars to consider valid

# =============================================================================
# Selenium / Headless Chrome Fallback
# =============================================================================

SELENIUM_HEADLESS = True
SELENIUM_PAGE_LOAD_TIMEOUT = 30  # seconds
SELENIUM_WAIT_AFTER_LOAD = 3     # seconds to wait for JS rendering

# =============================================================================
# Chunking Configuration
# =============================================================================

CHUNK_SIZE = 512           # max tokens per chunk
CHUNK_OVERLAP = 64         # token overlap between chunks
MIN_CHUNK_SIZE = 50        # discard chunks smaller than this
CHUNK_SEPARATORS = ["\n\n", "\n", ". ", " "]

# =============================================================================
# Embedding Configuration
# =============================================================================

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIMENSION = 384
EMBEDDING_BATCH_SIZE = 32

# Alternative: OpenAI embeddings (set via env var)
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
OPENAI_EMBEDDING_MODEL = "text-embedding-3-small"

# Groq API Key (for LLM generation)
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")

# =============================================================================
# Vector Store Configuration (ChromaDB)
# =============================================================================

CHROMA_COLLECTION_NAME = "groww_mf_chunks"
CHROMA_DISTANCE_METRIC = "cosine"

# =============================================================================
# LLM Configuration
# =============================================================================

LLM_PROVIDER = "groq"  # "groq" or "openai"
LLM_MODEL = "llama-3.3-70b-versatile"  # Groq model
LLM_TEMPERATURE = 0.0
LLM_MAX_TOKENS = 200
GROQ_BASE_URL = "https://api.groq.com/openai/v1"

# =============================================================================
# Scheme Metadata — Category Mapping
# =============================================================================

CATEGORY_MAP = {
    "hdfc-mid-cap-fund-direct-growth": {
        "scheme_name": "HDFC Mid-Cap Fund",
        "category": "mid-cap",
    },
    "hdfc-equity-fund-direct-growth": {
        "scheme_name": "HDFC Equity Fund",
        "category": "multi-cap",
    },
    "hdfc-focused-fund-direct-growth": {
        "scheme_name": "HDFC Focused Fund",
        "category": "focused",
    },
    "hdfc-elss-tax-saver-fund-direct-plan-growth": {
        "scheme_name": "HDFC ELSS Tax Saver Fund",
        "category": "elss",
    },
    "hdfc-balanced-advantage-fund-direct-growth": {
        "scheme_name": "HDFC Balanced Advantage Fund",
        "category": "balanced-advantage",
    },
    "hdfc-large-cap-fund-direct-growth": {
        "scheme_name": "HDFC Large Cap Fund",
        "category": "large-cap",
    },
    "hdfc-i-come-plus-arbitrage-active-fof-direct-growth": {
        "scheme_name": "HDFC Income Plus Arbitrage Active FoF",
        "category": "fof-arbitrage",
    },
    "hdfc-infrastructure-fund-direct-growth": {
        "scheme_name": "HDFC Infrastructure Fund",
        "category": "sectoral-thematic",
    },
    "hdfc-nifty-next-50-index-fund-direct-growth": {
        "scheme_name": "HDFC Nifty Next 50 Index Fund",
        "category": "index",
    },
    "hdfc-large-and-mid-cap-fund-direct-growth": {
        "scheme_name": "HDFC Large and Mid Cap Fund",
        "category": "large-mid-cap",
    },
    "hdfc-nifty-100-equal-weight-index-fund-direct-growth": {
        "scheme_name": "HDFC Nifty 100 Equal Weight Index Fund",
        "category": "index",
    },
    "hdfc-small-cap-fund-direct-growth": {
        "scheme_name": "HDFC Small Cap Fund",
        "category": "small-cap",
    },
    "hdfc-nifty50-equal-weight-index-fund-direct-growth": {
        "scheme_name": "HDFC Nifty 50 Equal Weight Index Fund",
        "category": "index",
    },
    "hdfc-multi-asset-active-fof-direct-growth": {
        "scheme_name": "HDFC Multi Asset Active FoF",
        "category": "fof-multi-asset",
    },
    "hdfc-retirement-savings-fund-equity-plan-direct-growth": {
        "scheme_name": "HDFC Retirement Savings Fund - Equity Plan",
        "category": "retirement-equity",
    },
}

AMC_NAME = "HDFC Mutual Fund"


def get_scheme_slug(url: str) -> str:
    """Extract the scheme slug from a Groww URL."""
    return url.rstrip("/").split("/")[-1]


def get_scheme_info(url: str) -> dict:
    """Get scheme name and category from a Groww URL."""
    slug = get_scheme_slug(url)
    info = CATEGORY_MAP.get(slug, {})
    return {
        "scheme_name": info.get("scheme_name", slug.replace("-", " ").title()),
        "scheme_slug": slug,
        "category": info.get("category", "unknown"),
        "amc": AMC_NAME,
    }


def ensure_directories():
    """Create all required data directories if they don't exist."""
    for dir_path in [DATA_DIR, RAW_DIR, PROCESSED_DIR, VECTORSTORE_DIR]:
        dir_path.mkdir(parents=True, exist_ok=True)
