"""Week 2: CLI entry point for testing the reusable LLMClient."""

import sys
from llm_client import LLMClient


def main() -> None:
    prompt = (
        " ".join(sys.argv[1:])
        if len(sys.argv) > 1
        else "Summarize why modular software design matters when building AI applications."
    )

    client = LLMClient(
        system_instruction="You are a concise computer science tutor."
    )
    result = client.generate(prompt=prompt, temperature=0.4)

    print(f"Model: {result.model}")
    print(f"Tokens -> prompt: {result.prompt_tokens}, output: {result.completion_tokens}, total: {result.total_tokens}")
    print("\nResponse:\n" + result.text)


if __name__ == "__main__":
    main()
