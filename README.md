# BS-Summarizer

A document summarization tool built with LangChain and LLMs.

## Features

- Summarizes both small and large documents
- Supports PDF, TXT, MD, and HTML files
- Uses different strategies based on document size:
  - Stuff method for small documents
  - Map-reduce method for large documents
- Allows using different LLM models for chunk summarization and final summarization
- Provides verbose output tracking the summarization process
- Returns intermediate summaries for analysis

## Requirements

- Python 3.8+
- Ollama (for running local LLMs)

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage

```python
from document_summarizer import DocumentSummarizer

# Initialize the summarizer
summarizer = DocumentSummarizer(
    model_name="qwen3",           # Model for final summary
    chunk_llm_name="llama3.2",    # Model for chunk summarization
    chunk_size=4000,
    chunk_overlap=500,
    verbose=True,
    language="English"
)

# Load a document
documents = summarizer.load_document("./files/your_document.pdf")

# Summarize the document
result = summarizer.summarize(documents)

# Access the final summary
print(result["summary"])
```

### Command Line Usage

```bash
# Summarize a document using default settings
python document_summarizer.py --file ./files/your_document.pdf

# Specify LLM models and language
python document_summarizer.py --file ./files/your_document.pdf --model qwen3 --chunk-model llama3.2 --language English
```

## Example

See the included `document_summarizer_example.py` file for a complete example with detailed usage patterns.

```bash
python document_summarizer_example.py
```
