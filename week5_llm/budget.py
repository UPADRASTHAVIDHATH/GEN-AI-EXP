"""Week 5: Token Economics and Budget Management for LLM Applications."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

# Approximate USD rates per 1,000,000 tokens (input_rate, output_rate)
DEFAULT_MODEL_PRICING: dict[str, tuple[float, float]] = {
    "gemini-2.5-flash": (0.15, 0.60),
    "gemini-2.5-pro": (1.25, 10.00),
    "gemini-2.0-flash": (0.10, 0.40),
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4o": (2.50, 10.00),
}


class BudgetExceededError(RuntimeError):
    """Raised when an LLM request would exceed the configured USD or token budget."""


@dataclass(frozen=True)
class UsageRecord:
    """Single logged LLM usage entry."""

    timestamp_utc: str
    model: str
    input_tokens: int
    output_tokens: int
    cost_usd: float
    tag: str = "default"

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


def estimate_tokens(text: str, model: str = "gpt-4o-mini") -> int:
    """Estimate token count using tiktoken when available, otherwise a char-based heuristic."""
    if not text:
        return 0
    try:
        import tiktoken

        try:
            enc = tiktoken.encoding_for_model(model)
        except KeyError:
            enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))
    except ImportError:
        return max(1, math.ceil(len(text) / 4))


@dataclass
class TokenBudgetManager:
    """Tracks token consumption, calculates cost, and enforces budget guardrails."""

    max_budget_usd: float
    max_total_tokens: int | None = None
    warning_threshold: float = 0.80
    pricing_table: dict[str, tuple[float, float]] = field(
        default_factory=lambda: dict(DEFAULT_MODEL_PRICING)
    )
    records: list[UsageRecord] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.max_budget_usd <= 0:
            raise ValueError("max_budget_usd must be positive.")
        if self.max_total_tokens is not None and self.max_total_tokens <= 0:
            raise ValueError("max_total_tokens must be positive when provided.")
        if not (0 < self.warning_threshold <= 1.0):
            raise ValueError("warning_threshold must be between 0 and 1.")

    def estimate_cost(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int,
    ) -> float:
        """Compute estimated USD cost for a request."""
        if input_tokens < 0 or output_tokens < 0:
            raise ValueError("Token counts must be non-negative.")
        if model not in self.pricing_table:
            raise KeyError(f"Unknown model pricing for '{model}'.")

        in_rate, out_rate = self.pricing_table[model]
        cost = (input_tokens / 1_000_000) * in_rate + (output_tokens / 1_000_000) * out_rate
        return round(cost, 8)

    @property
    def spent_usd(self) -> float:
        return round(sum(r.cost_usd for r in self.records), 8)

    @property
    def remaining_usd(self) -> float:
        return round(max(0.0, self.max_budget_usd - self.spent_usd), 8)

    @property
    def total_input_tokens(self) -> int:
        return sum(r.input_tokens for r in self.records)

    @property
    def total_output_tokens(self) -> int:
        return sum(r.output_tokens for r in self.records)

    @property
    def total_tokens(self) -> int:
        return self.total_input_tokens + self.total_output_tokens

    @property
    def utilization_ratio(self) -> float:
        return self.spent_usd / self.max_budget_usd

    def can_afford(
        self,
        model: str,
        estimated_input_tokens: int,
        estimated_output_tokens: int,
    ) -> bool:
        """Return True if the projected request stays within USD and token budgets."""
        projected_cost = self.estimate_cost(
            model, estimated_input_tokens, estimated_output_tokens
        )
        if self.spent_usd + projected_cost > self.max_budget_usd:
            return False

        if self.max_total_tokens is not None:
            projected_tokens = (
                self.total_tokens + estimated_input_tokens + estimated_output_tokens
            )
            if projected_tokens > self.max_total_tokens:
                return False

        return True

    def enforce_budget(
        self,
        model: str,
        estimated_input_tokens: int,
        estimated_output_tokens: int,
    ) -> float:
        """Validate that a projected call fits inside budget; raise BudgetExceededError otherwise."""
        projected_cost = self.estimate_cost(
            model, estimated_input_tokens, estimated_output_tokens
        )
        if self.spent_usd + projected_cost > self.max_budget_usd:
            raise BudgetExceededError(
                f"USD budget exceeded: projected ${projected_cost:.6f} "
                f"exceeds remaining ${self.remaining_usd:.6f}."
            )

        if self.max_total_tokens is not None:
            projected_tokens = (
                self.total_tokens + estimated_input_tokens + estimated_output_tokens
            )
            if projected_tokens > self.max_total_tokens:
                raise BudgetExceededError(
                    f"Token budget exceeded: projected {projected_tokens} tokens "
                    f"exceeds limit of {self.max_total_tokens}."
                )

        return projected_cost

    def record_usage(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int,
        tag: str = "default",
        enforce: bool = True,
    ) -> UsageRecord:
        """Record a completed LLM request and optionally enforce budget limits."""
        if enforce:
            cost = self.enforce_budget(model, input_tokens, output_tokens)
        else:
            cost = self.estimate_cost(model, input_tokens, output_tokens)

        record = UsageRecord(
            timestamp_utc=datetime.now(timezone.utc).isoformat(),
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost,
            tag=tag,
        )
        self.records.append(record)
        return record

    def status(self) -> str:
        """Return budget health state: OK, WARNING, or EXCEEDED."""
        if self.spent_usd >= self.max_budget_usd:
            return "EXCEEDED"
        if self.max_total_tokens is not None and self.total_tokens >= self.max_total_tokens:
            return "EXCEEDED"
        if self.utilization_ratio >= self.warning_threshold:
            return "WARNING"
        return "OK"

    def summary(self) -> dict[str, Any]:
        """Return a structured snapshot of current token and cost metrics."""
        return {
            "status": self.status(),
            "requests": len(self.records),
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_tokens": self.total_tokens,
            "spent_usd": self.spent_usd,
            "remaining_usd": self.remaining_usd,
            "max_budget_usd": self.max_budget_usd,
            "utilization_pct": round(self.utilization_ratio * 100.0, 2),
        }

    def reset(self) -> None:
        """Clear all tracked usage records."""
        self.records.clear()


def main() -> None:
    """Demonstrate token economics tracking and budget enforcement."""
    manager = TokenBudgetManager(
        max_budget_usd=0.05,
        max_total_tokens=100_000,
        warning_threshold=0.75,
    )

    sample_calls = [
        ("gemini-2.5-flash", 4_000, 1_500, "summarization"),
        ("gemini-2.5-pro", 8_000, 2_500, "code-review"),
        ("gemini-2.5-pro", 10_000, 3_500, "deep-analysis"),
    ]

    print("=== Week 5: Token Economics & Budget Management Demo ===")
    for model, in_tok, out_tok, tag in sample_calls:
        try:
            rec = manager.record_usage(model, in_tok, out_tok, tag=tag)
            print(
                f"[ALLOWED] tag={rec.tag:14s} model={rec.model:17s} "
                f"tokens={rec.total_tokens:6d} cost=${rec.cost_usd:.6f} "
                f"status={manager.status()}"
            )
        except BudgetExceededError as exc:
            print(f"[BLOCKED] tag={tag:14s} model={model:17s} reason={exc}")

    print("\nFinal Budget Summary:")
    for key, value in manager.summary().items():
        print(f"  {key:20s}: {value}")


if __name__ == "__main__":
    main()
