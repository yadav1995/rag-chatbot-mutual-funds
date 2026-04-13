"""
LLM Generator — Response Generation via Groq API (OpenAI-compatible)

Supports both Groq and OpenAI as LLM providers:
- Groq: llama-3.3-70b-versatile (fast, free tier available)
- OpenAI: gpt-4o-mini (fallback)

Both use the OpenAI Python client since Groq's API is OpenAI-compatible.
"""

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from config import (
    LLM_PROVIDER,
    LLM_MODEL,
    LLM_TEMPERATURE,
    LLM_MAX_TOKENS,
    GROQ_API_KEY,
    GROQ_BASE_URL,
    OPENAI_API_KEY,
)

from src.generation.prompt_templates import build_system_prompt, build_user_prompt

logger = logging.getLogger(__name__)


class LLMGenerator:
    """
    LLM response generator using Groq or OpenAI API.
    Both use the OpenAI Python client since Groq is OpenAI-compatible.
    """

    def __init__(
        self,
        provider: str = LLM_PROVIDER,
        model: str = LLM_MODEL,
        temperature: float = LLM_TEMPERATURE,
        max_tokens: int = LLM_MAX_TOKENS,
    ):
        self.provider = provider
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._client = None

    def _get_client(self):
        """Lazy-load the LLM client (Groq or OpenAI)."""
        if self._client is None:
            from openai import OpenAI

            if self.provider == "groq":
                self._client = OpenAI(
                    api_key=GROQ_API_KEY,
                    base_url=GROQ_BASE_URL,
                )
                logger.info(f"Groq client initialized (model={self.model})")
            else:
                self._client = OpenAI(api_key=OPENAI_API_KEY)
                logger.info(f"OpenAI client initialized (model={self.model})")

        return self._client

    def generate(
        self,
        query: str,
        context_chunks: list[dict],
        scrape_date: str = "N/A",
        conversation_history: list[dict] = None,
    ) -> str:
        """
        Generate a response using the LLM.

        Args:
            query: The user's question
            context_chunks: Retrieved chunks with text and metadata
            scrape_date: Date of last data scrape (for footer)
            conversation_history: Optional previous messages for context

        Returns:
            The LLM-generated response string
        """
        client = self._get_client()

        # Build messages
        messages = []

        # System prompt
        system_prompt = build_system_prompt(scrape_date)
        messages.append({"role": "system", "content": system_prompt})

        # Add conversation history (last 3 exchanges max)
        if conversation_history:
            recent = conversation_history[-6:]  # Last 3 user+assistant pairs
            messages.extend(recent)

        # User prompt with context
        user_prompt = build_user_prompt(query, context_chunks)
        messages.append({"role": "user", "content": user_prompt})

        logger.info(
            f"LLM call: provider={self.provider}, model={self.model}, "
            f"context_chunks={len(context_chunks)}, "
            f"messages={len(messages)}"
        )

        try:
            response = client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )

            answer = response.choices[0].message.content.strip()

            logger.info(
                f"LLM response: {len(answer)} chars, "
                f"tokens_used={response.usage.total_tokens}"
            )

            return answer

        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            return (
                "I'm sorry, I'm unable to generate a response at the moment. "
                "Please try again later.\n\n"
                f"Last updated from sources: {scrape_date}"
            )


# Singleton
_generator_instance = None


def get_generator() -> LLMGenerator:
    """Get or create the singleton LLM generator."""
    global _generator_instance
    if _generator_instance is None:
        _generator_instance = LLMGenerator()
    return _generator_instance
