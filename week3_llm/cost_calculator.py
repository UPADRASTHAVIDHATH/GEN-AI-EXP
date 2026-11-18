"""Week 3: Token usage cost estimator for Gemini models."""

from __future__ import annotations

from dataclasses import dataclass

# Approximate USD rates per 1,000,000 tokens (verify against current Google AI pricing)
MODEL_PRICING_PER_MILLION: dict[str, tuple[float, float]] = {
    "gemini-2.5-flash": (0.15, 0.60),
    "gemini-2.5-pro": (1.25, 10.00),
    "gemini-2.0-flash": (0.10, 0.40),
    "gemini-1.5-flash": (0.075, 0.30),
    "gemini-1.5-pro": (1.25, 5.00),
}

DEFAULT_PRICING = (0.15, 0.60)


@dataclass(frozen=True)
class CostBreakdown:
    """Computed USD cost breakdown for a single LLM call."""

    model: str
    input_tokens: int
    output_tokens: int
    input_cost_usd: float
    output_cost_usd: float
    total_cost_usd: float


def calculate_cost(
    model: str,
    input_tokens: int,
    output_tokens: int,
    pricing_table: dict[str, tuple[float, float]] | None = None,
) -> CostBreakdown:
    """Calculate estimated USD cost given model name and token counts."""
    if input_tokens < 0 or output_tokens < 0:
        raise ValueError("Token counts must be non-negative.")

    table = pricing_table if pricing_table is not None else MODEL_PRICING_PER_MILLION
    input_rate, output_rate = table.get(model, DEFAULT_PRICING)

    input_cost = (input_tokens / 1_000_000) * input_rate
    output_cost = (output_tokens / 1_000_000) * output_rate
    total_cost = input_cost + output_cost

    return CostBreakdown(
        model=model,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        input_cost_usd=round(input_cost, 8),
        output_cost_usd=round(output_cost, 8),
        total_cost_usd=round(total_cost, 8),
    )
