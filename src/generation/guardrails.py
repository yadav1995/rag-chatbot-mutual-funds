"""
Guardrails — Post-Generation Validation

Implements the guardrail validator per RAGArchitecture.md §4.7:
- Advisory language detection in LLM output
- Citation presence check
- Response length enforcement (≤ 3 sentences)
- PII leakage detection in output
- Hallucination check (key facts appear in retrieved chunks)
"""

import logging
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

logger = logging.getLogger(__name__)


@dataclass
class GuardrailResult:
    """Result of guardrail validation."""
    passed: bool
    violations: list[str] = field(default_factory=list)
    corrected_response: str = ""  # If auto-correctable


# =============================================================================
# Advisory Language Patterns (in LLM output)
# =============================================================================

ADVISORY_OUTPUT_PATTERNS = [
    r"\bi (?:would )?(?:recommend|suggest|advise)\b",
    r"\byou should (?:invest|buy|consider|switch|redeem)\b",
    r"\b(?:great|excellent|poor|bad|risky) (?:investment|fund|choice)\b",
    r"\bi think\b",
    r"\bin my opinion\b",
    r"\bmy recommendation\b",
    r"\bbest option\b",
    r"\bworst option\b",
    r"\bsafe bet\b",
    r"\bgood time to (?:invest|buy)\b",
]

ADVISORY_OUTPUT_RE = re.compile(
    "|".join(ADVISORY_OUTPUT_PATTERNS), re.IGNORECASE
)


# =============================================================================
# PII Patterns (in LLM output)
# =============================================================================

PII_OUTPUT_PATTERNS = {
    "pan": re.compile(r"\b[A-Z]{5}\d{4}[A-Z]\b"),
    "aadhaar": re.compile(r"\b\d{4}[\s\-]?\d{4}[\s\-]?\d{4}\b"),
    "email": re.compile(r"\b[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}\b"),
    "phone": re.compile(r"\b(?:\+91[\s\-]?)?[6-9]\d{9}\b"),
}

# Exceptions: emails that are legitimate (support, official)
PII_EMAIL_EXCEPTIONS = [
    "noreply@", "support@", "info@", "contact@",
    "@amfiindia.com", "@sebi.gov.in", "@groww.in",
]


# =============================================================================
# Guardrail Checks
# =============================================================================

def check_advisory_language(response: str) -> list[str]:
    """Check for advisory/recommendation language in the response."""
    violations = []
    matches = ADVISORY_OUTPUT_RE.findall(response)
    for match in matches:
        violations.append(f"Advisory language detected: '{match}'")
    return violations


def check_citation_present(response: str) -> list[str]:
    """Check that the response includes at least one URL citation."""
    violations = []
    url_pattern = re.compile(r"https?://\S+")
    if not url_pattern.search(response):
        violations.append("No source citation URL found in response")
    return violations


def check_response_length(response: str, max_sentences: int = 3) -> list[str]:
    """
    Check that the response is within the sentence limit.
    Excludes the 'Last updated' footer and citation lines.
    """
    violations = []

    # Remove the footer line and URL-only lines for counting
    lines = response.strip().split("\n")
    content_lines = [
        line for line in lines
        if line.strip()
        and not line.strip().startswith("Last updated")
        and not line.strip().startswith("Source:")
        and not re.match(r"^https?://\S+$", line.strip())
    ]

    content = " ".join(content_lines)

    # Count sentences using a smarter split that avoids decimals/URLs
    # Replace decimal numbers and URLs temporarily to avoid false splits
    import re as _re
    temp = _re.sub(r'\d+\.\d+', 'NUM', content)   # 0.77 → NUM
    temp = _re.sub(r'https?://\S+', 'URL', temp)    # URLs → URL

    sentences = re.split(r'(?<=[.!?])\s+', temp)
    sentences = [s.strip() for s in sentences if s.strip() and len(s.strip()) > 10]

    if len(sentences) > max_sentences:
        violations.append(
            f"Response too long: {len(sentences)} sentences (max {max_sentences})"
        )

    return violations


def check_pii_leakage(response: str) -> list[str]:
    """Check for PII data accidentally included in the response."""
    violations = []

    for pii_type, pattern in PII_OUTPUT_PATTERNS.items():
        matches = pattern.findall(response)
        for match in matches:
            # Check email exceptions
            if pii_type == "email":
                if any(exc in match.lower() for exc in PII_EMAIL_EXCEPTIONS):
                    continue
            violations.append(f"PII leakage ({pii_type}): '{match}'")

    return violations


def check_hallucination(response: str, context_chunks: list[dict]) -> list[str]:
    """
    Basic hallucination check: verify that key numeric values in the response
    appear in the retrieved context chunks.

    This is a lightweight check — full hallucination detection would require
    NLI models or more sophisticated approaches.
    """
    violations = []

    # Extract numbers from the response (e.g., ₹215.55, 0.77%, 85357)
    # Use non-capturing group to ensure trailing dots (end of sentence) aren't included
    response_numbers = set(re.findall(r"\d+(?:\.\d+)?", response))

    if not response_numbers or not context_chunks:
        return violations

    # Build a set of all numbers in the context
    context_text = " ".join(c["text"] for c in context_chunks)
    context_numbers = set(re.findall(r"\d+(?:\.\d+)?", context_text))

    # Check for numbers in response that don't appear in context
    suspicious = response_numbers - context_numbers
    # Filter out common numbers (years, small integers, dates)
    common_years = {str(y) for y in range(2010, 2031)}
    suspicious = {
        n for n in suspicious
        if float(n) > 31  # Skip small numbers including day-of-month (1-31)
        and n not in common_years
    }

    if suspicious:
        violations.append(
            f"Potential hallucination: numbers {suspicious} not found in context"
        )

    return violations


# =============================================================================
# Main Guardrail Validator
# =============================================================================

def validate_response(
    response: str,
    context_chunks: list[dict] = None,
    scrape_date: str = "N/A",
) -> GuardrailResult:
    """
    Run all guardrail checks on an LLM response.

    Returns a GuardrailResult with pass/fail status and any violations.
    """
    violations = []

    # Run all checks
    violations.extend(check_advisory_language(response))
    violations.extend(check_citation_present(response))
    violations.extend(check_response_length(response))
    violations.extend(check_pii_leakage(response))

    if context_chunks:
        violations.extend(check_hallucination(response, context_chunks))

    passed = len(violations) == 0

    if not passed:
        logger.warning(f"Guardrail violations: {violations}")

    result = GuardrailResult(
        passed=passed,
        violations=violations,
        corrected_response=response if passed else "",
    )

    return result
