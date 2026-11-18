# Week 2: Modular LLM Client & Automated Unit Testing

**Student Name:** Vidhath  
**Roll Number:** 2520030597

## Aim
Encapsulate Gemini API interactions in a reusable [`LLMClient`](./llm_client.py) class with configurable generation parameters, structured token-usage metadata, and unit tests using mocks.

## Requirements
- Python 3.10+
- `google-genai`
- `python-dotenv`
- `pytest` (optional, works with built-in `unittest` too)

## Project Structure
```text
week2_llm/
├── .env.example              # Environment variable template
├── .gitignore                # Ignores .env and __pycache__
├── 2520030597_Week-2.docx    # Lab report template
├── README.md                 # Documentation
├── llm_client.py             # Reusable LLMClient class
├── main.py                   # CLI entry point
├── requirements.txt          # Dependencies
└── test_client.py            # Mocked unit tests
```

## Setup & Execution
```bash
pip install -r requirements.txt
copy .env.example .env
# Edit .env with your GEMINI_API_KEY
python main.py "Explain object-oriented encapsulation briefly."
python -m unittest test_client.py -v
```
