"""Week 4: Subword Tokenizer Comparison Lab (BPE, WordPiece, SentencePiece).

Compares tokenization behavior of:
  - BPE (gpt2)
  - WordPiece (bert-base-multilingual-cased)
  - SentencePiece (xlm-roberta-base)
across English, Telugu, and Hindi sentences.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from tabulate import tabulate
from transformers import AutoTokenizer


TOKENIZER_CONFIGS: list[tuple[str, str]] = [
    ("BPE", "gpt2"),
    ("WordPiece", "bert-base-multilingual-cased"),
    ("SentencePiece", "xlm-roberta-base"),
]

DEFAULT_SAMPLES: dict[str, str] = {
    "English": "Generative AI models learn patterns from massive text datasets.",
    "Telugu": "జనరేటివ్ ఏఐ నమూనాలు పెద్ద టెక్స్ట్ డేటాసెట్ల నుండి నమూనాలను నేర్చుకుంటాయి.",
    "Hindi": "जनरेटिव एआई मॉडल बड़े टेक्स्ट डेटासेट से पैटर्न सीखते हैं।",
}


@dataclass
class TokenizationStats:
    """Metrics describing how a tokenizer segments an input sentence."""

    language: str
    algorithm: str
    model_name: str
    word_count: int
    token_count: int
    fertility: float
    chars_per_token: float
    tokens: list[str]
    token_ids: list[int]


def load_tokenizers() -> list[tuple[str, str, AutoTokenizer]]:
    """Download/load the three configured pretrained tokenizers."""
    loaded: list[tuple[str, str, AutoTokenizer]] = []
    for algorithm, model_name in TOKENIZER_CONFIGS:
        print(f"Loading {algorithm} tokenizer ({model_name})...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        loaded.append((algorithm, model_name, tokenizer))
    return loaded


def analyze_text(
    language: str,
    text: str,
    algorithm: str,
    model_name: str,
    tokenizer: AutoTokenizer,
) -> TokenizationStats:
    """Tokenize a single sentence and compute segmentation efficiency metrics."""
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    tokens = tokenizer.convert_ids_to_tokens(token_ids)
    word_count = max(len(text.split()), 1)
    token_count = len(token_ids)
    fertility = round(token_count / word_count, 2)
    chars_per_token = round(len(text) / max(token_count, 1), 2)

    return TokenizationStats(
        language=language,
        algorithm=algorithm,
        model_name=model_name,
        word_count=word_count,
        token_count=token_count,
        fertility=fertility,
        chars_per_token=chars_per_token,
        tokens=tokens,
        token_ids=token_ids,
    )


def run_comparison(samples: dict[str, str], show_tokens: bool = True) -> None:
    """Run all tokenizers across all language samples and print a summary report."""
    tokenizers = load_tokenizers()
    all_stats: list[TokenizationStats] = []

    print("\n" + "=" * 78)
    print("WEEK 4: TOKENIZER COMPARISON (BPE vs WordPiece vs SentencePiece)")
    print("=" * 78)

    for language, sentence in samples.items():
        print(f"\n[{language}] Input: {sentence}")
        print("-" * 78)
        for algorithm, model_name, tokenizer in tokenizers:
            stats = analyze_text(language, sentence, algorithm, model_name, tokenizer)
            all_stats.append(stats)
            if show_tokens:
                preview = stats.tokens[:18]
                suffix = " ..." if len(stats.tokens) > 18 else ""
                print(
                    f"  {algorithm:14s} ({model_name}): "
                    f"{stats.token_count} tokens | fertility={stats.fertility:.2f}"
                )
                print(f"    Tokens: {preview}{suffix}")

    rows = [
        [
            s.language,
            s.algorithm,
            s.model_name,
            s.word_count,
            s.token_count,
            f"{s.fertility:.2f}",
            f"{s.chars_per_token:.2f}",
        ]
        for s in all_stats
    ]

    print("\n" + "=" * 78)
    print("SUMMARY METRICS TABLE")
    print("=" * 78)
    print(
        tabulate(
            rows,
            headers=[
                "Language",
                "Algorithm",
                "Model",
                "Words",
                "Tokens",
                "Tokens/Word",
                "Chars/Token",
            ],
            tablefmt="github",
        )
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare BPE, WordPiece, and SentencePiece tokenizers."
    )
    parser.add_argument(
        "--text",
        type=str,
        default=None,
        help="Optional custom text to evaluate instead of the built-in multilingual samples.",
    )
    parser.add_argument(
        "--no-tokens",
        action="store_true",
        help="Hide individual token lists and show only the summary table.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    samples = {"Custom": args.text} if args.text else DEFAULT_SAMPLES
    run_comparison(samples=samples, show_tokens=not args.no_tokens)


if __name__ == "__main__":
    main()
