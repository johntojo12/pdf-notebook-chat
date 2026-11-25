"""
MEMORY + RETRIEVAL + ANSWERING

Retrieval pipeline for every question:
  1. rewrite_question()  -> make follow-ups standalone
  2. plan_sources()      -> decide: search everything, one file, or balanced across files
  3. hybrid_search()     -> vector (meaning) + BM25 (keywords), merged with RRF
  4. stream_answer()     -> Groq answers from the chunks, with citations
"""
import json
import math
import os
import re
import chromadb
from rank_bm25 import BM25Okapi

from config import (groq_client, CHAT_MODEL, FAST_MODEL, DATA_DIR,
                    SUMMARY_FILE, TOP_K, reasoning_args)

db = chromadb.PersistentClient(path=DATA_DIR)

SYSTEM_PROMPT = """You are a research assistant that answers ONLY from the provided SOURCES.
Rules:
1. Use only the sources. If the answer is not in them, say: "I couldn't find that in your sources."
2. After each claim, cite it like (filename p.PAGE).
3. Tables appear in Markdown; read rows and columns carefully and do arithmetic if asked.
4. Text marked 'Figure' is an AI-generated description of an image; mention that when relying on it.
5. When sources disagree, say so and cite both.
6. Be clear and well organised."""


# =====================  NOTEBOOKS (Chroma collections)  =====================
def safe_name(name: str) -> str:
    n = re.sub(r"[^a-zA-Z0-9_-]", "_", name.strip()).strip("_-")
    return (n + "nb1")[:60] if len(n) < 3 else n[:60]


def _get(nb: str, create: bool = False):
    """Return the collection, or None if it doesn't exist (unless create=True)."""
    if create:
        return db.get_or_create_collection(safe_name(nb))
    try:
        return db.get_collection(safe_name(nb))
    except Exception:
        return None


def list_notebooks() -> list[str]:
    return sorted(c.name for c in db.list_collections())


def delete_notebook(nb: str):
    try:
        db.delete_collection(safe_name(nb))
    except Exception:
        pass
    _bm25_cache.pop(nb, None)
    sums = _load_summaries()
    sums.pop(nb, None)
    _save_summaries(sums)


# =====================  WRITING TO MEMORY  =====================
def add_chunks(nb: str, chunks: list[dict]):
    import hashlib
    col = _get(nb, create=True)
    for i in range(0, len(chunks), 100):
        batch = chunks[i:i + 100]
        ids = [hashlib.md5(f"{c['meta']['file']}|{c['meta']['page']}|{c['text']}"
                           .encode()).hexdigest() for c in batch]
        col.upsert(ids=ids, documents=[c["text"] for c in batch],
                   metadatas=[c["meta"] for c in batch])
    _bm25_cache.pop(nb, None)


def list_sources(nb: str) -> list[str]:
    col = _get(nb)
    if not col or col.count() == 0:
        return []
    return sorted({m["file"] for m in col.get(include=["metadatas"])["metadatas"]})


def delete_source(nb: str, filename: str):
    col = _get(nb)
    if col:
        col.delete(where={"file": filename})
    _bm25_cache.pop(nb, None)
    sums = _load_summaries()
    sums.get(nb, {}).pop(filename, None)
    _save_summaries(sums)


# =====================  PER-FILE SUMMARIES  =====================
def _load_summaries() -> dict:
    if os.path.exists(SUMMARY_FILE):
        with open(SUMMARY_FILE, encoding="utf-8") as f:
            return json.load(f)
    return {}


def _save_summaries(d: dict):
    with open(SUMMARY_FILE, "w", encoding="utf-8") as f:
        json.dump(d, f, ensure_ascii=False, indent=2)


def get_summaries(nb: str) -> dict:
    return _load_summaries().get(nb, {})


def summarize_source(nb: str, filename: str, chunks: list[dict]):
    """Summarise one file right after upload and store the summary."""
    ordered = sorted(chunks, key=lambda c: c["meta"]["page"])
    total = sum(len(c["text"]) for c in ordered)
    stride = max(1, total // 8000)            # sample evenly across the WHOLE file
    text = ""
    for c in ordered[::stride]:
        if len(text) + len(c["text"]) > 9000:
            break
        text += c["text"] + "\n\n"

    if len(text.strip()) < 300:
        summary = ("(Not enough readable text to summarise. If this PDF is made of "
                   "images, install OCR (pip install rapidocr onnxruntime), tick "
                   "'Local OCR' in the sidebar, delete the file and upload it again.)")
    else:
        try:
            resp = groq_client.chat.completions.create(
                model=CHAT_MODEL, temperature=0, max_tokens=2000,
                extra_body=reasoning_args(CHAT_MODEL),
                messages=[{"role": "user", "content":
                           "Summarise the document text below in 150-200 words: topic, key "
                           "points, notable tables/figures. Use ONLY facts found in the text. "
                           "Do NOT add outside knowledge, applications, materials, numbers or "
                           "units that are not written there. If the text is fragmentary, say so.\n\n"
                           + text}])
            summary = resp.choices[0].message.content.strip()
        except Exception as e:
            summary = f"(summary failed: {e})"
    sums = _load_summaries()
    sums.setdefault(nb, {})[filename] = summary
    _save_summaries(sums)


def get_chunks(nb: str, filename: str, page: int | None = None):
    """Return stored (text, meta) pairs for a file (optionally one page), in page order."""
    col = _get(nb)
    if not col:
        return []
    where = ({"file": filename} if page is None
             else {"$and": [{"file": filename}, {"page": page}]})
    data = col.get(where=where, include=["documents", "metadatas"])
    pairs = list(zip(data["documents"], data["metadatas"]))
    order = {"table": 0, "text": 1, "ocr": 2, "figure": 3, "scanned_page": 4}
    pairs.sort(key=lambda p: (p[1]["page"], order.get(p[1]["type"], 9)))
    return pairs


def resummarize(nb: str, filename: str):
    """Rebuild a file's summary from the chunks already stored (no re-upload needed)."""
    col = _get(nb)
    data = col.get(where={"file": filename}, include=["documents", "metadatas"])
    chunks = [{"text": d, "meta": m} for d, m in zip(data["documents"], data["metadatas"])]
    summarize_source(nb, filename, chunks)


# =====================  HYBRID SEARCH  =====================
_bm25_cache: dict = {}


def _tok(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


def _bm25_index(nb: str, col):
    """Keyword index over ALL chunks in the notebook. Cached until data changes."""
    c = _bm25_cache.get(nb)
    if c and c["count"] == col.count():
        return c
    data = col.get(include=["documents", "metadatas"])
    tokens = [_tok(d) for d in data["documents"]]
    c = {"count": col.count(), "ids": data["ids"], "docs": data["documents"],
         "metas": data["metadatas"], "bm25": BM25Okapi(tokens) if tokens else None}
    _bm25_cache[nb] = c
    return c


def hybrid_search(nb: str, query: str, sources: list[str] | None, k: int):
    """
    Two searches, one merged ranking:
      - vector search: finds chunks with similar MEANING
      - BM25 search:   finds chunks with the same exact WORDS / numbers
    Merged with Reciprocal Rank Fusion: score = sum(1 / (60 + rank)) per list.
    """
    col = _get(nb)
    if not col or col.count() == 0:
        return []
    pool = k * 3
    where = None
    if sources:
        where = {"file": sources[0]} if len(sources) == 1 else {"file": {"$in": sources}}

    # --- vector ---
    vres = col.query(query_texts=[query], n_results=min(pool, col.count()), where=where)
    vec_ids = vres["ids"][0]
    info = {i: (d, m) for i, d, m in
            zip(vres["ids"][0], vres["documents"][0], vres["metadatas"][0])}

    # --- keyword ---
    kw_ids = []
    idx = _bm25_index(nb, col)
    if idx["bm25"]:
        scores = idx["bm25"].get_scores(_tok(query))
        for i in sorted(range(len(scores)), key=lambda j: -scores[j]):
            if scores[i] <= 0 or len(kw_ids) >= pool:
                break
            if sources and idx["metas"][i]["file"] not in sources:
                continue
            kw_ids.append(idx["ids"][i])
            info[idx["ids"][i]] = (idx["docs"][i], idx["metas"][i])

    # --- merge ---
    fused: dict = {}
    for ranked in (vec_ids, kw_ids):
        for rank, cid in enumerate(ranked):
            fused[cid] = fused.get(cid, 0) + 1 / (60 + rank)
    top = sorted(fused, key=fused.get, reverse=True)[:k]
    return [info[i] for i in top]


# =====================  ROUTING ACROSS SOURCES  =====================
COMPARE_WORDS = ("compare", "comparison", "difference", "differ", " vs ", "versus",
                 "both", "each", "all documents", "all files", "across", "contrast")


def plan_sources(question: str, selected: list[str]):
    """Decide which files to search and whether to give each file a fair share."""
    q = question.lower()
    mentioned = [s for s in selected if os.path.splitext(s)[0].lower() in q]
    if mentioned:
        return mentioned, len(mentioned) > 1, f"Focused on: {', '.join(mentioned)}"
    if len(selected) > 1 and any(w in q for w in COMPARE_WORDS):
        return selected, True, f"Balanced across {len(selected)} sources"
    return selected, False, f"Searched all {len(selected)} selected sources"


def retrieve(nb: str, question: str, selected: list[str], k: int = TOP_K):
    targets, balanced, note = plan_sources(question, selected)
    if balanced:
        per = max(2, math.ceil(k / len(targets)))
        hits = []
        for src in targets:                       # each file gets its own quota
            hits += hybrid_search(nb, question, [src], per)
    else:
        hits = hybrid_search(nb, question, targets, k)
    return hits, note


# =====================  QUESTION REWRITING  =====================
def rewrite_question(question: str, history: list[dict]) -> str:
    """'what about the second one?' -> a standalone question good for searching."""
    if not history:
        return question
    convo = "\n".join(f"{m['role']}: {m['content'][:400]}" for m in history[-4:])
    try:
        resp = groq_client.chat.completions.create(
            model=FAST_MODEL, temperature=0, max_tokens=800,
            extra_body=reasoning_args(FAST_MODEL),
            messages=[{"role": "user", "content":
                       "Rewrite the last user question so it is fully standalone, using the "
                       "conversation for context. Output ONLY the rewritten question.\n\n"
                       f"Conversation:\n{convo}\n\nLast question: {question}"}])
        return resp.choices[0].message.content.strip() or question
    except Exception:
        return question


# =====================  ANSWERING  =====================
def _clean(history: list[dict]) -> list[dict]:
    return [{"role": m["role"], "content": m["content"]} for m in history]


def stream_answer(question: str, hits: list, history: list[dict]):
    if not hits:
        yield "No sources found. Upload a PDF first."
        return
    context = "\n\n---\n\n".join(
        f"[{m['file']} | p.{m['page']} | {m['type']}]\n{d}" for d, m in hits)
    messages = ([{"role": "system", "content": SYSTEM_PROMPT}]
                + _clean(history)[-6:]
                + [{"role": "user", "content": f"SOURCES:\n{context}\n\nQUESTION: {question}"}])
    stream = groq_client.chat.completions.create(
        model=CHAT_MODEL, messages=messages, temperature=0.2, stream=True,
        extra_body=reasoning_args(CHAT_MODEL))
    for ev in stream:
        tok = ev.choices[0].delta.content
        if tok:
            yield tok


OVERVIEW_PROMPTS = {
    "Briefing doc": "Write a briefing document: overview, key themes, important facts and numbers, and conclusions.",
    "FAQ": "Write 8-10 frequently asked questions with concise answers.",
    "Study guide": "Write a study guide: key concepts with definitions, 8 short-answer quiz questions, then an answer key.",
}


def stream_overview(nb: str, selected: list[str], kind: str):
    """Generate briefing/FAQ/study guide from the per-file summaries (no chunk search)."""
    sums = get_summaries(nb)
    material = "\n\n".join(f"## {f}\n{sums[f]}" for f in selected if f in sums)
    if not material:
        yield "No summaries available yet. Upload and process a PDF first."
        return
    stream = groq_client.chat.completions.create(
        model=CHAT_MODEL, temperature=0.3, stream=True,
        extra_body=reasoning_args(CHAT_MODEL),
        messages=[{"role": "system", "content":
                   "Use ONLY the document summaries below. Mention which file each point comes from."},
                  {"role": "user", "content":
                   f"{OVERVIEW_PROMPTS[kind]}\n\nSUMMARIES:\n{material}"}])
    for ev in stream:
        tok = ev.choices[0].delta.content
        if tok:
            yield tok
