"""Week 3: Run a Gemini prompt with cost calculation and MongoDB logging."""

import sys
from llm_client import LLMClient
from mongo_logger import MongoLogger


def main() -> None:
    prompt = (
        " ".join(sys.argv[1:])
        if len(sys.argv) > 1
        else "Explain why logging token usage and cost is essential for production LLM systems."
    )

    mongo_logger: MongoLogger | None = None
    try:
        mongo_logger = MongoLogger()
    except Exception as exc:
        print(f"Warning: MongoDB unavailable ({exc}). Continuing without DB logging.")

    client = LLMClient(logger=mongo_logger)
    try:
        result = client.generate(
            prompt=prompt,
            temperature=0.5,
            metadata={"student": "Vidhath", "roll_number": "2520030597"},
        )
    except Exception as exc:
        print(f"Error executing LLM call: {exc}", file=sys.stderr)
        sys.exit(1)
    finally:
        if mongo_logger is not None:
            mongo_logger.close()

    print(f"Model      : {result.model}")
    print(f"Latency    : {result.latency_ms:.2f} ms")
    print(
        f"Tokens     : input={result.prompt_tokens}, output={result.completion_tokens}, total={result.total_tokens}"
    )
    print(f"Est. Cost  : ${result.cost.total_cost_usd:.8f} USD")
    print(f"MongoDB ID : {result.log_id or 'Not logged'}")
    print("\nResponse:\n" + result.text)


if __name__ == "__main__":
    main()
