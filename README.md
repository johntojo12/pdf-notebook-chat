# 📓 Notebook Chat: talk to your PDFs

**Upload PDFs, ask questions in plain English, get answers with page citations.**
It understands **tables, diagrams, scanned pages and image-only catalogues**, and remembers your files between sessions.

> A small, open-source "notebook-style" document assistant built with Python, Streamlit, ChromaDB, local OCR and the [Groq](https://groq.com) API.
> Not affiliated with or endorsed by Google or any other company.

---

## Table of contents
1. [What does it do?](#-what-does-it-do)
2. [How it works (simple version)](#-how-it-works-simple-version)
3. [Quick start](#-quick-start)
4. [Using the app](#-using-the-app)
5. [Project structure](#-project-structure)
6. [Module guide](#-module-guide-what-each-file-does)
7. [Settings you can change](#-settings-you-can-change)
8. [Privacy: what goes where](#-privacy-what-goes-where)
9. [Troubleshooting](#-troubleshooting)
10. [Glossary for beginners](#-glossary-for-beginners)
11. [Limitations and ideas](#-limitations-and-ideas)

---

## ✨ What does it do?

| Feature | What it means for you |
|---|---|
| **Notebooks** | Group related PDFs together. Each notebook has its own memory and chat. |
| **Many files, one search** | Ask a question and it searches *all* PDFs in the notebook at once. |
| **Page citations** | Every answer says where it came from, like `(report.pdf p.7)`. |
| **Table understanding** | Real tables are converted to Markdown so rows and columns stay aligned. |
| **Image-only PDFs** | Catalogues where text is baked into pictures are read with **local OCR** (free, on your computer). |
| **Diagrams and scans** | Optional: an AI vision model describes charts and diagrams. |
| **Hybrid search** | Finds text by *meaning* and by *exact words or numbers* (like part code `F 90 CP 020 C`). |
| **Smart routing** | Name a file in your question and only that file is searched. Say "compare…" and each file gets a fair share. |
| **Follow-up questions** | "What about the second one?" is rewritten into a full question behind the scenes. |
| **Summaries and study tools** | Per-file summaries, plus one-click **Briefing**, **FAQ** and **Study guide**. |
| **Inspector** | See exactly what text was extracted and saved, page by page. |
| **Persistent memory** | Close the app and come back later; your notebooks are still there. |

---

## 🧠 How it works (simple version)

An AI model **cannot read your whole PDF every time**. It is too long, and the model doesn't remember files. So the app works like a smart librarian:

1. **Reading (once per file):** it extracts text, tables and OCR text, then cuts everything into small pieces called **chunks**, each tagged with its file and page.
2. **Filing:** each chunk is turned into numbers that capture its meaning (an **embedding**) and saved in a local database. This is the app's "memory".
3. **Asking (every question):** the app finds the few chunks most relevant to your question.
4. **Answering:** it gives *only those chunks* to the AI model and asks it to answer from them, with citations.

```
 ┌────────────── UPLOAD (once per file) ──────────────┐
 │ PDF                                                │
 │  ├─ normal text ───────────┐                       │
 │  ├─ real tables → Markdown ├─► chunks + labels ──► embeddings ──► 💾 database
 │  ├─ image-only page → OCR ─┤   (file, page, type)  (local CPU)   (data/chroma)
 │  └─ charts/scans → vision ─┘
 └────────────────────────────────────────────────────┘

 ┌────────────── ASK (every question) ────────────────┐
 │ your question                                      │
 │  → rewrite follow-ups into a full question         │
 │  → decide which files to search                    │
 │  → vector search + keyword search, merged          │
 │  → top 8 chunks                                    │
 │  → Groq AI writes the answer with citations ✅     │
 └────────────────────────────────────────────────────┘
```
This technique is called **RAG** (Retrieval-Augmented Generation).

---

## 🚀 Quick start

### What you need
- **Python 3.11 or 3.12** (check with `python --version`)
- **Git** (to download the code)
- A **free Groq API key** from [console.groq.com](https://console.groq.com) → *API Keys*

### Step 1: Download the code
```bash
git clone <your-repo-url>
cd <repo-folder-name>
```

### Step 2: Create a virtual environment (keeps packages tidy)
```bash
python -m venv venv
```
Activate it:
- **Windows (PowerShell):** `venv\Scripts\activate`
- **macOS / Linux:** `source venv/bin/activate`

You should now see `(venv)` at the start of your terminal line.

### Step 3: Install the packages
```bash
pip install -r requirements.txt
```
This can take a few minutes. If it seems stuck on "looking at multiple versions", just wait or see [Troubleshooting](#-troubleshooting).

### Step 4: Add your API key
Copy the example settings file, then open the new `.env` file and paste your key:
- **Windows:** `copy .env.example .env`
- **macOS / Linux:** `cp .env.example .env`

```
GROQ_API_KEY=gsk_your_key_here
CHAT_MODEL=openai/gpt-oss-120b
FAST_MODEL=openai/gpt-oss-20b
VISION_MODEL=qwen/qwen3.8-27b
```
> ⚠️ **Never share or commit your `.env` file.** It holds your secret key. It is already listed in `.gitignore`.

> Model names change over time and depend on your Groq plan. See the list at [console.groq.com/docs/models](https://console.groq.com/docs/models) and edit `.env` if a name stops working.

### Step 5: Run it
```bash
streamlit run app.py
```
Your browser opens at **http://localhost:8501**. The first run downloads a small embedding model (about 80 MB), so give it a minute.

---

## 🖱 Using the app

1. **Pick or create a notebook** in the left sidebar.
2. **Upload PDFs** and click **Process files**. A progress bar shows each page, what was found (`ocr`, `text`, `table`…) and an estimated time left.
3. **Ask questions** in the chat box at the bottom.
4. **Check the sources.** Under each answer, open **📎 Sources used** to see the exact passages.
5. **Inspect the data.** Open **🔍 Inspect what was extracted and saved** to read the text the app stored for any page, or download it all as a `.txt`.
6. **Generate extras** with the **✨ Briefing doc / FAQ / Study guide** buttons.
7. **Choose sources.** Untick a file in the sidebar to exclude it from searches.

Sidebar options:
- **Local OCR for image-only pages:** reads text inside pictures. Keep this on for catalogues and scans.
- **Caption diagrams/scans with AI:** uses the vision model. Slow on free plans, so turn it off if OCR is enough.
- **Max images to caption per file:** caps vision calls to protect your rate limit.

**Example questions**
- *"What sizes of reducing bush are available?"*
- *"Compare the pressure ratings in report.pdf and budget.pdf."*
- *"What is part code F 65 CP 114 L?"*

---

## 📂 Project structure

```
.
├── app.py            # the web interface (what you see)
├── ingest.py         # reads PDFs and turns them into chunks
├── ocr.py            # local OCR for image-only pages
├── rag.py            # memory, search, routing and answering
├── config.py         # settings and model names
├── inspect_db.py     # command-line tool to view stored data
├── requirements.txt  # list of Python packages
├── .env.example      # template for your secret settings
├── .gitignore        # files Git should never upload
└── data/             # created automatically: your saved notebooks
    ├── chroma/       #   the vector database (the "memory")
    └── summaries.json#   saved per-file summaries
```

---

## 🧩 Module guide: what each file does

### `config.py`: settings in one place
Loads your `.env`, creates the shared Groq client, and holds every tunable number (chunk size, limits, model names). Other files import from here so you only change settings in one spot. It also has `reasoning_args()`, which keeps "thinking" models fast by lowering their reasoning effort.

### `ingest.py`: turns a PDF into searchable chunks
The reading step. For each page it does the following:

| Function | Job |
|---|---|
| `ingest_pdf()` | Main entry. Loops over pages, reports progress, returns all chunks. |
| `process_page()` | Handles one page: tables, text, OCR, then vision fallback. |
| `split_text()` | Cuts normal text into ~1000-character chunks on paragraph breaks, with a small overlap so sentences aren't lost at the edges. |
| `split_table()` | Splits very large tables by rows and repeats the header in each part. |
| `split_lines()` | Cuts OCR text on line boundaries so a table row is never split in half. |
| `extract_page_images()` | Pulls out the largest meaningful images on a page, skipping tiny logos. |
| `describe_image()` | Sends an image to the Groq vision model and returns a text description. It paces itself to respect rate limits and retries on errors. |

Page logic in plain words: *real tables → Markdown; normal text → chunks; page with almost no text → local OCR; if OCR isn't available → vision model.*

### `ocr.py`: reads text that is baked into images
Many catalogues are exported as pictures, so normal extraction finds nothing. This module renders the page, uses **RapidOCR** to detect every piece of text, then **rebuilds the rows**: boxes at the same height are joined left to right. A table row becomes:
```
F 90 CP 020 C | 20 | 27.5 | 16.5 | 12.0
```
It runs on your CPU, with no API calls and no limits.

### `rag.py`: memory, search and answers
The brain of the app.

| Area | Functions | What they do |
|---|---|---|
| **Notebooks** | `get_notebook`-style helpers (`_get`, `list_notebooks`, `delete_notebook`, `safe_name`) | Each notebook is its own collection in ChromaDB, so notebooks never mix. |
| **Saving** | `add_chunks`, `delete_source`, `list_sources` | Stores chunks (with embeddings made locally) and removes files. |
| **Search** | `hybrid_search` | Runs **vector search** (meaning) and **BM25 keyword search** (exact words and numbers), then merges both rankings with *Reciprocal Rank Fusion*. |
| **Routing** | `plan_sources`, `retrieve` | If your question names a file, search only that file. If it says "compare/difference/both", give each file an equal share of results. |
| **Follow-ups** | `rewrite_question` | Rewrites "what about the second one?" into a standalone question before searching. |
| **Answering** | `stream_answer` | Sends the best chunks to Groq with strict rules: answer only from sources, cite pages, say so if not found. |
| **Summaries** | `summarize_source`, `resummarize`, `stream_overview` | Writes a faithful summary per file, then builds Briefing / FAQ / Study guide from those summaries. |
| **Inspecting** | `get_chunks` | Returns stored chunks for a file or page. |

### `app.py`: the interface
A [Streamlit](https://streamlit.io) page with three parts: the **sidebar** (notebooks, upload, source checkboxes, options), the **Inspector** panel, and the **chat**. It wires user actions to `ingest.py` and `rag.py`, and keeps a separate chat history per notebook.

### `inspect_db.py`: look inside the memory from the terminal
```bash
python inspect_db.py                              # list notebooks and files
python inspect_db.py my_notebook file.pdf         # print and save all extracted text
python inspect_db.py my_notebook file.pdf 5       # only page 5
```

### Chunk types you'll see
| Type | Meaning |
|---|---|
| `text` | Normal text found inside the PDF |
| `table` | A real table converted to Markdown |
| `ocr` | Text read from an image-only page by local OCR |
| `figure` | AI description of a chart, diagram or photo |
| `scanned_page` | AI transcription of a scanned page (vision fallback) |

---

## ⚙️ Settings you can change

Edit `config.py` (or `.env` for models and the vision limit):

| Setting | Default | What it does |
|---|---|---|
| `CHAT_MODEL` | `openai/gpt-oss-120b` | Writes the answers |
| `FAST_MODEL` | `openai/gpt-oss-20b` | Cheap helper for rewriting questions |
| `VISION_MODEL` | `qwen/qwen3.8-27b` | Describes images |
| `TOP_K` | `8` | How many chunks are sent to the AI per question. Raise for deeper answers. |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | `1000` / `150` | Size of text pieces and how much they overlap |
| `OCR_MIN_CHARS` | `300` | Pages with less real text than this get OCR |
| `VISION_OTPM` | `1000` | Your plan's output-tokens-per-minute limit. Raise it after upgrading. |
| `DEFAULT_MAX_IMAGES` | `6` | Images captioned per file by the vision model |

---

## 🔒 Privacy: what goes where

| Step | Where it runs |
|---|---|
| PDF reading, tables, chunking | 💻 Your computer |
| **OCR** | 💻 Your computer |
| **Embeddings** and search | 💻 Your computer |
| Saved notebooks (`data/`) | 💻 Your computer only |
| Answers, summaries, question rewriting | ☁️ Groq, receiving only the **top retrieved chunks**, never the whole PDF |
| Image captions (optional) | ☁️ Groq, receiving just the images it describes |

Turn off the vision option and OCR still works fully offline for extraction. Only the chat step needs the internet.

---

## 🛠 Troubleshooting

| Problem | Fix |
|---|---|
| `pip install` seems stuck ("looking at multiple versions…") | It is working out compatible versions. Wait, or use a fresh venv with Python 3.11/3.12, or `pip install uv` then `uv pip install -r requirements.txt`. |
| `model_not_found` or 404 error | That model isn't available on your plan. List yours at `https://api.groq.com/openai/v1/models` and update `.env`. |
| `429 rate limit` or "Request too large" during upload | Vision calls hit your tokens-per-minute limit. Turn off **Caption diagrams/scans**, lower the image slider, or rely on **Local OCR**. |
| Answer says *"I couldn't find that in your sources"* | Open the **Inspector** and check that page has extracted text. If it's empty, the PDF is image-only, so enable **Local OCR**, delete the file and re-upload. |
| Warning: *OCR not installed* | Run `pip install rapidocr onnxruntime`, then restart. |
| Summary says *"summary failed"* | Click **🔄 Regenerate summary** in the file's sidebar expander. |
| `ModuleNotFoundError` | Your virtual environment isn't active. Run the activate command from Step 2. |
| `streamlit` not recognised | Activate the venv, or run `python -m streamlit run app.py`. |
| Wrong or stale data | Delete the `data/` folder to wipe all notebooks and start fresh. |
| Wrong numbers read from a scanned table | OCR can misread small digits. Check the Inspector and verify critical values against the PDF. |

---

## 📖 Glossary for beginners

- **RAG (Retrieval-Augmented Generation):** find relevant passages first, then let an AI answer using them.
- **Chunk:** a small piece of a document (a paragraph, one table, one OCR block).
- **Embedding:** a list of numbers that represents the *meaning* of text, so similar meanings sit close together.
- **Vector database (ChromaDB):** a database that stores embeddings and quickly finds the closest ones.
- **BM25:** a classic keyword search that matches exact words and numbers.
- **Hybrid search:** combining meaning-based and keyword search for better results.
- **OCR:** software that reads text from images.
- **LLM:** the large language model (here, models served by Groq) that writes answers.
- **Token:** a small piece of text (roughly ¾ of a word) that models count for pricing and limits.
- **Rate limit:** a cap on how much you can use an API per minute.
- **Virtual environment (venv):** a private folder of Python packages for one project.

---

## 🚧 Limitations and ideas

**Known limits**
- OCR can misread tiny digits, and two-column pages may interleave lines.
- Free-tier vision limits make image captioning slow.
- Summaries use a sample of up to ~9,000 characters, so very long files get a high-level summary only.
- Answers can still contain mistakes, especially arithmetic, so verify important facts against the citations.

**Ideas for the future**
- A reranker to sharpen search results
- Word, PowerPoint, web page and YouTube sources
- Audio overviews with text-to-speech
- Export chat and notes
- Better handling of multi-column layouts

---

## 🤝 Contributing

Issues and pull requests are welcome. Please do **not** commit `.env`, API keys or private documents.

## 📄 License

Add a `LICENSE` file (for example MIT) to tell others how they may use this code.
