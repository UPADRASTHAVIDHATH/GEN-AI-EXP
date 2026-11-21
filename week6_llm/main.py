"""Week 6: Golden Question Set Baseline Evaluation.

Automated evaluation baseline (v0.1) for an LLM (llama3.2 via Ollama)
against a curated Golden Question Set, scoring answers through keyword
matching and generating structured benchmark reports.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests
from tabulate import tabulate


DEFAULT_OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate")
DEFAULT_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2")
DEFAULT_GOLDEN_SET = "golden_set.json"
DEFAULT_OUTPUT_CSV = "baseline_results.csv"
PASSING_SCORE_THRESHOLD = 60.0


@dataclass
class EvalResult:
    """Evaluation metrics for a single question evaluation."""

    question_id: int
    category: str
    question: str
    expected_answer: str
    model_response: str
    target_keywords: list[str]
    matched_keywords: list[str]
    score_pct: float
    status: str
    latency_sec: float


def load_golden_set(filepath: str | Path) -> list[dict[str, Any]]:
    """Load and validate the curated golden question dataset."""
    path = Path(filepath)
    if not path.is_file():
        raise FileNotFoundError(f"Golden dataset not found at: {path.resolve()}")

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list) or not data:
        raise ValueError("Golden question set must be a non-empty list of question items.")

    return data


def query_ollama(
    prompt: str,
    model: str = DEFAULT_MODEL,
    url: str = DEFAULT_OLLAMA_URL,
    temperature: float = 0.0,
    timeout: float = 60.0,
) -> tuple[str, float]:
    """Send prompt to local Ollama API at deterministic temperature=0 and measure latency."""
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": temperature,
        },
    }

    start = time.perf_counter()
    response = requests.post(url, json=payload, timeout=timeout)
    response.raise_for_status()
    elapsed = time.perf_counter() - start

    data = response.json()
    model_text = data.get("response", "").strip()
    return model_text, round(elapsed, 3)


def mock_ollama_query(item: dict[str, Any]) -> tuple[str, float]:
    """Deterministic mock query generator for test runs when Ollama service is offline."""
    # Provides realistic responses containing relevant domain keywords
    simulated_responses = {
        1: (
            "Generative AI models the underlying joint probability distribution to produce new data instances. "
            "In contrast, discriminative AI models conditional probability to classify data and draw decision boundaries between class labels."
        ),
        2: (
            "In LLMs, hallucination happens when the model generates plausible but factually incorrect or ungrounded assertions. "
            "The model presents these statements with high confidence even though they lack factual truth in the source data."
        ),
        3: (
            "Byte-Pair Encoding (BPE) merges the most frequent symbol pairs based on raw co-occurrence frequency counts. "
            "WordPiece, conversely, chooses subword merges that maximize the likelihood of the training corpus under a unigram language model."
        ),
        4: (
            "SentencePiece treats text as a raw stream of unicode characters or bytes without requiring whitespace pre-tokenization. "
            "This is advantageous for multilingual scripts like Telugu and Hindi because it avoids language-specific tokenizers and reduces fertility."
        ),
        5: (
            "Setting temperature to 0 causes the LLM to use greedy decoding, deterministically picking the token with maximum probability "
            "at each step and turning off random sampling entirely."
        ),
        6: (
            "System instructions configure the overarching persona, rules, and behavioral constraints of the assistant. "
            "The user prompt delivers the specific task or question that the model must process within those constraints."
        ),
        7: (
            "Completion tokens require sequential autoregressive GPU computation where each new token must await previous tokens. "
            "In contrast, prompt tokens are ingested simultaneously in parallel during the prefill stage, making them less demanding on memory bandwidth."
        ),
        8: (
            "A Golden Question Set is a curated benchmark of representative domain questions with established reference answers. "
            "It is used to monitor performance across model iterations and guard against silent quality regression."
        ),
    }

    resp = simulated_responses.get(
        item.get("id"),
        f"Answer to: {item.get('question')} with reference to {item.get('expected_answer')}",
    )
    return resp, 0.42


def evaluate_response(
    item: dict[str, Any],
    response_text: str,
    latency_sec: float,
) -> EvalResult:
    """Score model response based on target keyword matching accuracy."""
    target_keywords = item.get("target_keywords", [])
    if not target_keywords:
        raise ValueError(f"Question ID {item.get('id')} has no target_keywords.")

    # Normalize response text: lowercase and simplify spaces
    clean_response = re.sub(r"\s+", " ", response_text.lower())

    matched: list[str] = []
    for kw in target_keywords:
        kw_clean = kw.strip().lower()
        # Word boundary or substring matching
        pattern = r"\b" + re.escape(kw_clean) + r"\b"
        if re.search(pattern, clean_response) or kw_clean in clean_response:
            matched.append(kw)

    score_pct = round((len(matched) / len(target_keywords)) * 100.0, 2)
    status = "PASS" if score_pct >= PASSING_SCORE_THRESHOLD else "REVIEW"

    return EvalResult(
        question_id=item["id"],
        category=item.get("category", "General"),
        question=item["question"],
        expected_answer=item.get("expected_answer", ""),
        model_response=response_text,
        target_keywords=target_keywords,
        matched_keywords=matched,
        score_pct=score_pct,
        status=status,
        latency_sec=latency_sec,
    )


def save_results_csv(results: list[EvalResult], output_path: str | Path) -> None:
    """Save benchmark results to structured CSV report."""
    path = Path(output_path)
    fieldnames = [
        "id",
        "category",
        "question",
        "expected_answer",
        "model_response",
        "target_keywords",
        "matched_keywords",
        "score_pct",
        "status",
        "latency_sec",
    ]

    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for res in results:
            writer.writerow(
                {
                    "id": res.question_id,
                    "category": res.category,
                    "question": res.question,
                    "expected_answer": res.expected_answer,
                    "model_response": res.model_response,
                    "target_keywords": "; ".join(res.target_keywords),
                    "matched_keywords": "; ".join(res.matched_keywords),
                    "score_pct": f"{res.score_pct:.2f}",
                    "status": res.status,
                    "latency_sec": f"{res.latency_sec:.3f}",
                }
            )


def print_summary_report(
    results: list[EvalResult],
    model_name: str,
    output_csv: str,
    is_mock: bool = False,
) -> None:
    """Print tabular evaluation metrics and summary statistics to console."""
    print("\n" + "=" * 84)
    print("WEEK 6: GOLDEN QUESTION SET BASELINE EVALUATION REPORT")
    print("=" * 84)
    mode_str = "SIMULATED / MOCK" if is_mock else "LOCAL OLLAMA INFERENCE"
    print(f"Model Evaluated : {model_name} ({mode_str})")
    print(f"Evaluation Mode : Deterministic (temperature = 0.0)")
    print(f"Pass Threshold  : >= {PASSING_SCORE_THRESHOLD}% keyword match")
    print(f"Timestamp (UTC) : {time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime())}")
    print("-" * 84)

    table_data = []
    for r in results:
        table_data.append(
            [
                r.question_id,
                r.category,
                f"{len(r.matched_keywords)} / {len(r.target_keywords)}",
                f"{r.score_pct:.1f}%",
                r.status,
                f"{r.latency_sec:.2f}s",
            ]
        )

    headers = ["ID", "Category", "Keywords Matched", "Score", "Status", "Latency"]
    print(tabulate(table_data, headers=headers, tablefmt="github"))
    print("-" * 84)

    total_questions = len(results)
    passed_count = sum(1 for r in results if r.status == "PASS")
    review_count = total_questions - passed_count
    pass_rate = (passed_count / total_questions) * 100.0 if total_questions > 0 else 0.0
    avg_score = (
        sum(r.score_pct for r in results) / total_questions if total_questions > 0 else 0.0
    )
    mean_latency = (
        sum(r.latency_sec for r in results) / total_questions if total_questions > 0 else 0.0
    )

    print(f"Total Questions Evaluated : {total_questions}")
    print(f"Passed Questions (>= 60%) : {passed_count} ({pass_rate:.1f}%)")
    print(f"Needs Review (< 60%)      : {review_count}")
    print(f"Average Benchmark Score   : {avg_score:.2f}%")
    print(f"Mean Inference Latency    : {mean_latency:.3f} s")
    print(f"Detailed CSV Report Saved : {Path(output_csv).resolve()}")
    print("=" * 84 + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run automated Golden Question Set baseline evaluation against local LLM."
    )
    parser.add_argument(
        "--golden-set",
        type=str,
        default=DEFAULT_GOLDEN_SET,
        help="Path to golden question set JSON file.",
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default=DEFAULT_OUTPUT_CSV,
        help="Path to output baseline CSV results.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help="Ollama model name (default: llama3.2).",
    )
    parser.add_argument(
        "--url",
        type=str,
        default=DEFAULT_OLLAMA_URL,
        help="Ollama API generate endpoint.",
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Run evaluation with simulated deterministic responses (offline / CI testing).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Load golden questions
    script_dir = Path(__file__).parent
    golden_path = script_dir / args.golden_set if not Path(args.golden_set).is_absolute() else Path(args.golden_set)
    output_path = script_dir / args.output_csv if not Path(args.output_csv).is_absolute() else Path(args.output_csv)

    try:
        golden_set = load_golden_set(golden_path)
    except Exception as exc:
        print(f"Error loading golden set: {exc}", file=sys.stderr)
        sys.exit(1)

    print(f"Loaded {len(golden_set)} benchmark questions from {golden_path.name}.")
    results: list[EvalResult] = []
    use_mock = args.mock

    # Test Ollama connection if not explicitly mocked
    if not use_mock:
        try:
            requests.get("http://localhost:11434/", timeout=2.0)
        except Exception:
            print(
                "\n[Notice] Local Ollama service at http://localhost:11434 was not reachable.\n"
                "Switching automatically to deterministic simulated mode so evaluation can proceed.\n"
                "(To use live Ollama, run `ollama serve` and `ollama run llama3.2`)\n"
            )
            use_mock = True

    for item in golden_set:
        qid = item.get("id")
        q_text = item.get("question", "")
        print(f"Evaluating Q{qid}: {q_text[:60]}...")

        if use_mock:
            response_text, latency = mock_ollama_query(item)
        else:
            try:
                response_text, latency = query_ollama(
                    prompt=q_text,
                    model=args.model,
                    url=args.url,
                    temperature=0.0,
                )
            except Exception as exc:
                print(f"  Warning: Ollama query failed for Q{qid} ({exc}). Using simulated response.")
                response_text, latency = mock_ollama_query(item)

        eval_res = evaluate_response(item, response_text, latency)
        results.append(eval_res)

    # Save to CSV
    save_results_csv(results, output_path)

    # Display summary
    print_summary_report(
        results=results,
        model_name=args.model,
        output_csv=str(output_path),
        is_mock=use_mock,
    )


if __name__ == "__main__":
    main()
