"""
Query Classifier — Intent Detection & PII Guard

Implements multi-layer classification per RAGArchitecture.md §4.1 and §7:
- Rule-based PII detection (HARD BLOCK): PAN, Aadhaar, email, phone
- Rule-based advisory detection: "should", "recommend", "better"
- Keyword-based intent classification: FACTUAL vs OFF_TOPIC

Classification flow:
  Query → PII Check → Advisory Check → Off-Topic Check → FACTUAL
"""

import logging
import re
import sys
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

logger = logging.getLogger(__name__)


class QueryIntent(Enum):
    """Possible query classifications."""
    FACTUAL = "factual"
    ADVISORY = "advisory"
    COMPARATIVE = "comparative"
    OFF_TOPIC = "off_topic"
    PII_DETECTED = "pii_detected"


@dataclass
class ClassificationResult:
    """Result of query classification."""
    intent: QueryIntent
    confidence: float  # 0.0 - 1.0
    reason: str  # Human-readable explanation
    pii_type: str = ""  # Type of PII detected (if any)


# =============================================================================
# PII Detection Patterns
# =============================================================================

PII_PATTERNS = {
    "pan": re.compile(r"\b[A-Z]{5}\d{4}[A-Z]\b"),
    "aadhaar": re.compile(r"\b\d{4}[\s\-]?\d{4}[\s\-]?\d{4}\b"),
    "email": re.compile(r"\b[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}\b"),
    "phone": re.compile(r"\b(?:\+91[\s\-]?)?[6-9]\d{9}\b"),
    "account_number": re.compile(r"\b\d{9,18}\b"),
    "otp": re.compile(r"\bOTP\b.*\b\d{4,6}\b", re.IGNORECASE),
}

# =============================================================================
# Advisory / Comparative Keywords
# =============================================================================

ADVISORY_KEYWORDS = [
    "should i invest",
    "should i buy",
    "is it good",
    "is it safe",
    "is it worth",
    "recommend",
    "suggestion",
    "suggest",
    "advice",
    "advise",
    "best fund",
    "best mutual fund",
    "good investment",
    "bad investment",
    "safe to invest",
    "risky fund",
    "worth investing",
    "potential returns",
    "future performance",
    "will it grow",
    "good time to invest",
    "should i switch",
    "should i redeem",
    "where should",
    "what should",
]

COMPARATIVE_KEYWORDS = [
    "which is better",
    "which fund is best",
    "which fund gives",
    "gives better",
    "compare",
    "comparison",
    "versus",
    " vs ",
    "better than",
    "worse than",
    "outperform",
    "underperform",
    "better returns",
]

# =============================================================================
# Mutual Fund Topic Keywords (for on-topic detection)
# =============================================================================

MF_TOPIC_KEYWORDS = [
    "mutual fund", "fund", "sip", "nav", "aum", "expense ratio",
    "exit load", "lock-in", "lock in", "elss", "hdfc", "nifty",
    "equity", "debt", "balanced", "index fund", "mid cap", "midcap",
    "large cap", "largecap", "small cap", "smallcap", "multi cap",
    "flexi cap", "flexicap", "tax saver", "lump sum", "lumpsum",
    "scheme", "portfolio", "asset allocation", "benchmark",
    "fund manager", "returns", "growth", "dividend", "direct plan",
    "regular plan", "folio", "redemption", "stamp duty", "stcg", "ltcg",
    "capital gains", "unit", "folio", "arbitrage", "fof",
    "retirement", "infrastructure", "thematic", "sectoral",
    "minimum investment", "risk", "riskometer", "category",
]


# =============================================================================
# Classifier
# =============================================================================

def classify_query(query: str) -> ClassificationResult:
    """
    Classify a user query through the multi-layer classification pipeline.

    Pipeline:
    1. PII Check → HARD BLOCK
    2. Advisory Check → Polite refusal
    3. Comparative Check → Polite refusal
    4. Off-Topic Check → Generic refusal
    5. Default → FACTUAL (proceed to retrieval)
    """
    query_lower = query.lower().strip()

    # ── Layer 1: PII Detection (HARD BLOCK) ──────────────────────────────
    for pii_type, pattern in PII_PATTERNS.items():
        if pattern.search(query):
            logger.warning(f"PII DETECTED ({pii_type}): query blocked")
            return ClassificationResult(
                intent=QueryIntent.PII_DETECTED,
                confidence=1.0,
                reason=f"PII detected: {pii_type}",
                pii_type=pii_type,
            )

    # ── Layer 2: Advisory Intent ─────────────────────────────────────────
    for keyword in ADVISORY_KEYWORDS:
        if keyword in query_lower:
            logger.info(f"Advisory intent detected: '{keyword}'")
            return ClassificationResult(
                intent=QueryIntent.ADVISORY,
                confidence=0.9,
                reason=f"Advisory keyword detected: '{keyword}'",
            )

    # ── Layer 3: Comparative Intent ──────────────────────────────────────
    for keyword in COMPARATIVE_KEYWORDS:
        if keyword in query_lower:
            logger.info(f"Comparative intent detected: '{keyword}'")
            return ClassificationResult(
                intent=QueryIntent.COMPARATIVE,
                confidence=0.9,
                reason=f"Comparative keyword detected: '{keyword}'",
            )

    # ── Layer 4: On-Topic Check ──────────────────────────────────────────
    is_on_topic = any(kw in query_lower for kw in MF_TOPIC_KEYWORDS)

    if not is_on_topic and len(query_lower.split()) > 2:
        # Short queries (1-2 words) get a pass — could be abbreviations
        logger.info(f"Off-topic query: '{query[:50]}...'")
        return ClassificationResult(
            intent=QueryIntent.OFF_TOPIC,
            confidence=0.7,
            reason="Query does not appear to be about mutual funds",
        )

    # ── Default: FACTUAL ─────────────────────────────────────────────────
    return ClassificationResult(
        intent=QueryIntent.FACTUAL,
        confidence=0.8,
        reason="Query classified as factual",
    )
