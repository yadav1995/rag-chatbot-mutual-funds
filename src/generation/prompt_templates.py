"""
Prompt Templates — System, User, and Refusal Prompts

Implements all prompt templates per RAGArchitecture.md §5.
"""

# =============================================================================
# System Prompt — sent as the system message to the LLM
# =============================================================================

SYSTEM_PROMPT = """You are a facts-only mutual fund FAQ assistant. You answer using ONLY the provided \
context from official AMC, AMFI, and SEBI sources.

STRICT RULES:
1. Answer in a maximum of 3 sentences.
2. Include EXACTLY ONE source citation URL from the context.
3. End every response with: "Last updated from sources: {scrape_date}"
4. NEVER provide investment advice, opinions, or recommendations.
5. NEVER compare fund performance or calculate returns.
6. If the context does not contain the answer, say:
   "I don't have this information in my current sources. Please check [relevant official URL]."
7. NEVER ask for or acknowledge PAN, Aadhaar, account numbers, OTPs, emails, or phone numbers.
8. For performance-related queries, respond ONLY with a link to the official factsheet.

You are NOT a financial advisor. You are a factual information retrieval system."""


# =============================================================================
# User Prompt Template — wraps the assembled context + query
# =============================================================================

USER_PROMPT_TEMPLATE = """Context:
{context}

Question: {query}

Answer (max 3 sentences, 1 citation, include last updated date):"""


# =============================================================================
# Context Assembly Template — formats each retrieved chunk
# =============================================================================

CONTEXT_CHUNK_TEMPLATE = """[SOURCE {index}] (scheme: {scheme_name} | section: {section} | url: {source_url})
{text}"""


# =============================================================================
# Refusal Templates
# =============================================================================

REFUSAL_ADVISORY = """I can only provide factual information about mutual fund schemes, such as expense \
ratios, exit loads, SIP amounts, and lock-in periods. I'm unable to offer investment \
advice or recommendations.

For investment guidance, please visit: https://www.amfiindia.com/investor-corner

Last updated from sources: {scrape_date}"""


REFUSAL_COMPARATIVE = """I can only provide factual information about individual mutual fund schemes. \
I'm unable to compare funds or make relative performance judgments.

For fund comparisons, please visit: https://www.amfiindia.com/mutual-fund-tools

Last updated from sources: {scrape_date}"""


REFUSAL_PII = """I cannot process personal information. Please never share PAN numbers, \
Aadhaar numbers, bank account details, OTPs, email addresses, or phone numbers \
with this assistant.

Your privacy is important. No personal data has been stored.

Last updated from sources: {scrape_date}"""


REFUSAL_OFF_TOPIC = """I can only answer factual questions about mutual fund schemes — such as \
expense ratios, exit loads, minimum SIP amounts, fund managers, and asset allocation.

Please try a question like: "What is the expense ratio of HDFC Mid-Cap Fund?"

Last updated from sources: {scrape_date}"""


REFUSAL_NO_CONTEXT = """I don't have this information in my current sources. \
Please check the official Groww page or AMFI website for the latest details.

Visit: https://www.amfiindia.com

Last updated from sources: {scrape_date}"""


# =============================================================================
# Helper Functions
# =============================================================================

def build_system_prompt(scrape_date: str = "N/A") -> str:
    """Build the system prompt with the scrape date."""
    return SYSTEM_PROMPT.replace("{scrape_date}", scrape_date)


def build_user_prompt(query: str, context_chunks: list[dict]) -> str:
    """
    Assemble the user prompt from query and retrieved chunks.

    Args:
        query: The user's question
        context_chunks: List of chunk dicts with 'text' and 'metadata'

    Returns:
        Formatted user prompt string
    """
    # Build context from chunks
    context_parts = []
    for i, chunk in enumerate(context_chunks, start=1):
        meta = chunk.get("metadata", {})
        context_parts.append(CONTEXT_CHUNK_TEMPLATE.format(
            index=i,
            scheme_name=meta.get("scheme_name", "Unknown"),
            section=meta.get("section", "Unknown"),
            source_url=meta.get("source_url", "N/A"),
            text=chunk["text"],
        ))

    context = "\n\n".join(context_parts)

    return USER_PROMPT_TEMPLATE.format(
        context=context,
        query=query,
    )


def get_refusal_response(intent: str, scrape_date: str = "N/A") -> str:
    """Get the appropriate refusal response for a given intent."""
    templates = {
        "advisory": REFUSAL_ADVISORY,
        "comparative": REFUSAL_COMPARATIVE,
        "pii_detected": REFUSAL_PII,
        "off_topic": REFUSAL_OFF_TOPIC,
        "no_context": REFUSAL_NO_CONTEXT,
    }
    template = templates.get(intent, REFUSAL_OFF_TOPIC)
    return template.format(scrape_date=scrape_date)
