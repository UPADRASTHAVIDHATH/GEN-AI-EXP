# Week 3: Token Usage Cost Estimation & MongoDB Request Logging

**Student Name:** Vidhath  
**Roll Number:** 2520030597

## Aim
Extend the reusable Gemini LLM client with per-request latency measurement, token cost estimation ([`cost_calculator.py`](./cost_calculator.py)), and persistent telemetry logging in MongoDB ([`mongo_logger.py`](./mongo_logger.py)).

## Requirements
- Python 3.10+
- `google-genai`
- `python-dotenv`
- `pymongo`
- Running local MongoDB instance or MongoDB Atlas connection string

## Project Structure
```text
week3_llm/
├── .env.example              # Gemini API key and MongoDB URI template
├── .gitignore                # Excludes .env and cache directories
├── 2520030597_Week-3.docx    # Lab report template
├── README.md                 # Documentation
├── cost_calculator.py        # Per-token USD cost calculator
├── llm_client.py             # Instrumented Gemini LLM client
├── main.py                   # CLI script with DB logging
├── mongo_logger.py           # MongoDB persistence wrapper
├── requirements.txt          # Dependencies
└── test_client.py            # Mocked unit tests
```

## Setup & Execution
```bash
pip install -r requirements.txt
copy .env.example .env
# Edit .env with your GEMINI_API_KEY and MONGO_URI
python main.py "Why is observability important for LLM apps?"
python -m unittest test_client.py -v
```
