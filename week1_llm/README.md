# Week 1: Introduction to Generative AI with Google Gemini API

**Student Name:** Vidhath  
**Roll Number:** 2520030597

## Aim
Set up a Python development environment, load API credentials securely from a `.env` file, and generate text responses using the official Google GenAI SDK (`google-genai`).

## Requirements
- Python 3.10+
- `google-genai`
- `python-dotenv`

## Project Structure
```text
week1_llm/
├── .env.example              # Template for GEMINI_API_KEY
├── .gitignore                # Excludes .env and Python cache files
├── 2520030597_Week-1.docx    # Lab report template
├── README.md                 # Documentation
├── call.py                   # Gemini API text generation script
└── requirements.txt          # Python dependencies
```

## Setup & Execution
```bash
pip install -r requirements.txt
copy .env.example .env
# Edit .env and insert your GEMINI_API_KEY
python call.py "What are large language models?"
```
