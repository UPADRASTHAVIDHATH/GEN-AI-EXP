"""Week 2: Reusable Gemini LLM client wrapper."""

from __future__ import annotations

import os
from dataclasses import dataclass
from dotenv import load_dotenv
from google import genai
from google.genai import types


@dataclass
class LLMResponse:
    """Structured response returned by LLMClient."""

    text: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class LLMClient:
    """Clean wrapper around the Google GenAI SDK."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        system_instruction: str | None = None,
    ) -> None:
        load_dotenv()
        resolved_key = api_key or os.getenv("GEMINI_API_KEY")
        if not resolved_key:
            raise ValueError(
                "GEMINI_API_KEY is required. Pass api_key or configure .env."
            )

        self.model = model or os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
        self.system_instruction = system_instruction
        self._client = genai.Client(api_key=resolved_key)

    def generate(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_output_tokens: int | None = None,
    ) -> LLMResponse:
        """Generate text from a prompt and return structured output with token counts."""
        if not prompt or not prompt.strip():
            raise ValueError("Prompt must be a non-empty string.")

        config_kwargs: dict[str, object] = {"temperature": temperature}
        if max_output_tokens is not None:
            config_kwargs["max_output_tokens"] = max_output_tokens
        if self.system_instruction:
            config_kwargs["system_instruction"] = self.system_instruction

        config = types.GenerateContentConfig(**config_kwargs)
        raw = self._client.models.generate_content(
            model=self.model,
            contents=prompt,
            config=config,
        )

        usage = getattr(raw, "usage_metadata", None)
        prompt_tokens = int(getattr(usage, "prompt_token_count", 0) or 0)
        completion_tokens = int(getattr(usage, "candidates_token_count", 0) or 0)
        total_tokens = int(
            getattr(usage, "total_token_count", prompt_tokens + completion_tokens) or 0
        )

        return LLMResponse(
            text=raw.text or "",
            model=self.model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
        )
