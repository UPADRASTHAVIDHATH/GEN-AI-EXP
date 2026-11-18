# Week 5: Token Economics and Budget Management

**Student Name:** Vidhath  
**Roll Number:** 2520030597

## Aim
Design and test a [`TokenBudgetManager`](./budget.py) that models LLM token economics, tracks cumulative USD spend and token volume across models, triggers warning thresholds, and enforces hard cost/token guardrails before requests execute.

## Requirements
- Python 3.10+
- `pytest` (optional; built-in `unittest` also works)
- `tiktoken` (optional; falls back to character heuristic if unavailable)

## Project Structure
```text
week5_llm/
├── .gitignore                 # Excludes virtual environments and cache files
├── 2520030597_Week-5.docx     # Lab report template
├── README.md                  # Documentation
├── budget.py                  # Token economics & budget guardrail engine
├── requirements.txt           # Python dependencies
└── test_budget.py             # Unit tests for cost & budget enforcement
```

## Setup & Execution
```powershell
pip install -r requirements.txt
python budget.py
python -m unittest test_budget.py -v
```
