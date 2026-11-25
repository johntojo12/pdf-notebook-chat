# 📓 NotebookLM-style Document Chat (Groq + Local RAG)

Chat with your PDFs, including **tables, diagrams and image-only pages**, with page-level citations.
Built with Streamlit, ChromaDB, BM25, local OCR and the Groq API.

## Features
- **Notebooks**: each notebook holds many PDFs, searched together
- **Hybrid search**: vector (meaning) + BM25 (exact words/numbers), merged with Reciprocal Rank Fusion
- **Smart routing**: names a file → searches that file; "compare…" → balanced across files
- **Complex PDFs**: real tables → Markdown, image-only pages → local OCR, charts/scans → optional vision model
- **Citations** on every answer, plus an inspector showing exactly what was extracted
- Per-file summaries, Briefing / FAQ / Study guide generators, follow-up question rewriting

## Architecture
```
PDF → text + tables (PyMuPDF) → local OCR for image pages (RapidOCR) → chunks + metadata
    → embeddings (Chroma, local CPU) → vector DB on disk
Question → rewrite → route sources → vector + BM25 → top chunks → Groq LLM → cited answer
```
Embeddings and OCR run **locally**. Groq is used for answers, summaries, query rewriting
and (optionally) image captions.

## Setup
```bash
git clone <your-repo-url>
cd <repo-folder>
python -m venv venv
venv\Scripts\activate            # Windows  (macOS/Linux: source venv/bin/activate)
pip install -r requirements.txt
copy .env.example .env           # macOS/Linux: cp .env.example .env
# edit .env and paste your GROQ_API_KEY
streamlit run app.py
```
Python 3.11 or 3.12 recommended. Model names change over time: check
https://console.groq.com/docs/models and edit `.env` if needed.

## Project layout
| File | Purpose |
|---|---|
| `app.py` | Streamlit UI |
| `ingest.py` | PDF → chunks (text, tables, OCR, figures) |
| `ocr.py` | Local OCR with table-row reconstruction |
| `rag.py` | Vector DB, hybrid search, routing, answering, summaries |
| `config.py` | Settings and model names |
| `inspect_db.py` | Inspect what was extracted and stored |

## Limitations
- Free-tier Groq limits make vision captioning slow; OCR avoids this for most catalogues
- OCR can misread small digits: verify critical numbers
- Two-column pages may interleave lines in OCR output
