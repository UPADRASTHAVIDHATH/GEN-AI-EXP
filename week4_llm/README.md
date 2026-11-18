# Week 4: Tokenizer Lab (BPE, WordPiece, SentencePiece)

**Student Name:** Vidhath  
**Roll Number:** 2520030597

## Aim
Compare subword tokenization algorithms across **English**, **Telugu**, and **Hindi** sentences:
- **BPE (Byte-Pair Encoding):** `gpt2`
- **WordPiece:** `bert-base-multilingual-cased`
- **SentencePiece (Unigram):** `xlm-roberta-base`

## Requirements
- Python 3.10+
- `transformers`
- `sentencepiece`
- `protobuf`
- `tabulate`

## Project Structure
```text
week4_llm/
├── .gitignore                 # Excludes virtual environments and cache files
├── 2520030597_Week-4.docx     # Lab report template
├── README.md                  # Documentation
├── requirements.txt           # Python dependencies
└── tokenizer_comparison.py    # Multilingual tokenizer benchmark script
```

## Setup & Execution
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python tokenizer_comparison.py
```

You can also evaluate a custom sentence:
```powershell
python tokenizer_comparison.py --text "Large language models tokenize text into subwords."
```
