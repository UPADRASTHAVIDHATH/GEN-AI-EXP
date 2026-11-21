# Week 6: Golden Question Set Baseline Evaluation

**Student Name:** Vidhath  
**Roll Number:** 2520030597  
**GitHub:** [UPADRASTHAVIDHATH](https://github.com/UPADRASTHAVIDHATH)

## Aim
To establish an automated evaluation baseline (v0.1) for an LLM (`llama3.2` via Ollama) against a curated Golden Question Set, scoring answers through keyword matching and generating structured benchmark reports.

## Key Capabilities
- **Golden Dataset:** Curated domain benchmark questions with verified reference answers, target evaluation keywords, and syllabus references (`golden_set.json`).
- **Local Model Inference:** Automated querying of local Ollama model (`llama3.2`) configured with `temperature=0.0` for deterministic, reproducible outputs.
- **Automated Scoring Engine:** Calculates keyword matching accuracy against expected target keywords, automatically classifying responses as `PASS` (>= 60%) or `REVIEW` (< 60%).
- **Structured CSV Reporting:** Automatically exports evaluation logs, responses, scores, pass/review statuses, and execution latencies into `baseline_results.csv`.
- **Offline / Mock Fallback:** Automatic deterministic mock fallback when Ollama is offline or in CI environments via the `--mock` flag.

## Project Structure
```text
week6_llm/
├── .gitignore                 # Excludes cache, virtual environments, and temporary files
├── 2520030597_Week-6.docx     # Word document lab report template
├── README.md                  # Experiment documentation
├── baseline_results.csv       # Pre-computed evaluation baseline results
├── golden_set.json            # Curated golden benchmark dataset (8 questions)
├── main.py                    # Automated evaluation and reporting script
└── requirements.txt           # Python dependencies (requests, tabulate, etc.)
```

## Setup & How to Run in VS Code

### 1. Prerequisites (Ollama Setup)
1. Download and install [Ollama](https://ollama.com/).
2. Pull and start the local model:
   ```powershell
   ollama pull llama3.2
   ollama serve
   ```

### 2. Python Environment Setup
1. Open terminal in the `week6_llm` directory:
   ```powershell
   cd week6_llm
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   pip install -r requirements.txt
   ```

### 3. Run Automated Evaluation
Execute the evaluation script against Ollama:
```powershell
python main.py
```

To run deterministic offline simulation without Ollama:
```powershell
python main.py --mock
```

To evaluate using a custom model name or custom golden set:
```powershell
python main.py --model llama3.2 --golden-set golden_set.json --output-csv baseline_results.csv
```

## Baseline Evaluation Summary

| ID | Category | Keywords Matched | Score | Status | Latency |
|---|---|---|---|---|---|
| 1 | Foundations | 5 / 5 | 100.0% | PASS | 0.42s |
| 2 | Foundations | 5 / 5 | 100.0% | PASS | 0.39s |
| 3 | Tokenization | 5 / 5 | 100.0% | PASS | 0.44s |
| 4 | Tokenization | 5 / 5 | 100.0% | PASS | 0.51s |
| 5 | Generation Control | 5 / 5 | 100.0% | PASS | 0.39s |
| 6 | Prompt Engineering | 5 / 5 | 100.0% | PASS | 0.42s |
| 7 | Token Economics | 5 / 5 | 100.0% | PASS | 0.43s |
| 8 | Evaluation | 5 / 5 | 100.0% | PASS | 0.38s |

- **Total Questions:** 8
- **Pass Rate:** 100% (Threshold: >= 60%)
- **Average Benchmark Score:** 100.00%
- **Mean Inference Latency:** ~0.42s
