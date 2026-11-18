"""Week 1: Basic text generation using the Google GenAI SDK."""

import os
import sys
from dotenv import load_dotenv
from google import genai


def generate_response(prompt: str, model_name: str | None = None) -> str:
    """Send a prompt to Gemini and return the generated text."""
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError(
            "GEMINI_API_KEY is missing. Copy .env.example to .env and add your API key."
        )

    selected_model = model_name or os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model=selected_model,
        contents=prompt,
    )
    return response.text or ""


def main() -> None:
    prompt = (
        " ".join(sys.argv[1:])
        if len(sys.argv) > 1
        else "Explain Generative AI in three concise bullet points."
    )
    print(f"Prompt: {prompt}\n")
    try:
        output = generate_response(prompt)
        print("Response:")
        print(output)
    except Exception as exc:
        print(f"Error calling Gemini API: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
