"""Week 3: Instrumented Gemini LLM client with cost estimation and MongoDB logging."""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from dotenv import load_dotenv
from google import genai
from google.genai import types

from cost_calculator import CostBreakdown, calculate_cost
from mongo_logger import MongoLogger


@dataclass
class TrackedLLMResponse:
    """Response object containing text, token metrics, cost breakdown, latency, and log ID."""

    text: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    latency_ms: float
    cost: CostBreakdown
    log_id: str | None = None


class LLMClient:
    """Gemini client wrapper that measures latency, estimates cost, and logs to MongoDB."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        logger: MongoLogger | None = None,
        system_instruction: str | None = None,
    ) -> None:
        load_dotenv()
        resolved_key = api_key or os.getenv("GEMINI_API_KEY")
        if not resolved_key:
            raise ValueError(
                "GEMINI_API_KEY is missing. Copy .env.example to .env and configure it."
            )

        self.model = model or os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
        self.system_instruction = system_instruction
        self.logger = logger
        self._client = genai.Client(api_key=resolved_key)

    def generate(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_output_tokens: int | None = None,
        metadata: dict[str, object] | None = None,
    ) -> TrackedLLMResponse:
        """Generate a completion, calculate its estimated cost, and optionally log it."""
        if not prompt or not prompt.strip():
            raise ValueError("Prompt must be a non-empty string.")

        config_kwargs: dict[str, object] = {"temperature": temperature}
        if max_output_tokens is not None:
            config_kwargs["max_output_tokens"] = max_output_tokens
        if self.system_instruction:
            config_kwargs["system_instruction"] = self.system_instruction

        config = types.GenerateContentConfig(**config_kwargs)

        start = time.perf_counter()
        raw = self._client.models.generate_content(
            model=self.model,
            contents=prompt,
            config=config,
        )
        latency_ms = (time.perf_counter() - start) * 1000.0

        usage = getattr(raw, "usage_metadata", None)
        prompt_tokens = int(getattr(usage, "prompt_token_count", 0) or 0)
        completion_tokens = int(getattr(usage, "candidates_token_count", 0) or 0)
        total_tokens = int(
            getattr(usage, "total_token_count", prompt_tokens + completion_tokens) or 0
        )

        cost = calculate_cost(
            model=self.model,
            input_tokens=prompt_tokens,
            output_tokens=completion_tokens,
        )

        text = raw.text or ""
        log_id: str | None = None
        if self.logger is not None:
            log_id = self.logger.log_interaction(
                prompt=prompt,
                response_text=text,
                model=self.model,
                input_tokens=prompt_tokens,
                output_tokens=completion_tokens,
                total_tokens=total_tokens,
                total_cost_usd=cost.total_cost_usd,
                latency_ms=latency_ms,
                metadata=metadata,
            )

        return TrackedLLMResponse(
            text=text,
            model=self.model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            latency_ms=round(latency_ms, 2),
            cost=cost,
            log_id=log_id,
        )
