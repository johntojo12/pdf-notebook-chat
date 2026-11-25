"""
INGESTION: PDF  ->  list of text chunks with metadata.

Every kind of content (text, tables, diagrams, scans) is turned into TEXT,
because only text can be embedded and searched.
"""
import base64
import re
import time
from collections import Counter
import pymupdf as fitz  # PyMuPDF

import ocr
from config import (groq_client, OCR_MIN_CHARS, VISION_MODEL, CHUNK_SIZE, CHUNK_OVERLAP,
                    MAX_IMAGES_PER_PAGE, MIN_IMAGE_PX, VISION_OTPM,
                    VISION_MAX_TOKENS, SCAN_MAX_TOKENS, DEFAULT_MAX_IMAGES)

VISION_PROMPT = (
    "You are helping index a document for search. Describe this image concisely but "
    "completely. Chart/graph: title, axes, units, series, key numbers and trends. "
    "Diagram: every label and how they connect. Table: reproduce as a Markdown table. "
    "Page of text: transcribe it. Be factual; do not guess."
)


# ---------- 1. Vision: image -> text ---------------------------------------
_next_ok = 0.0   # earliest time the next vision call may start


def _throttle(max_tokens: int):
    """Stay under the output-tokens-per-minute limit by spacing calls out."""
    global _next_ok
    wait = _next_ok - time.time()
    if wait > 0:
        time.sleep(wait)
    _next_ok = time.time() + max_tokens / VISION_OTPM * 60


def describe_image(png_bytes: bytes, max_tokens: int = VISION_MAX_TOKENS) -> str:
    """Ask the Groq vision model to describe an image (rate-limit aware)."""
    b64 = base64.b64encode(png_bytes).decode()
    variants = [{"reasoning_effort": "none"}, {}]   # try thinking-off first
    v = 0
    for attempt in range(5):
        _throttle(max_tokens)
        try:
            resp = groq_client.chat.completions.create(
                model=VISION_MODEL,
                messages=[{"role": "user", "content": [
                    {"type": "text", "text": VISION_PROMPT},
                    {"type": "image_url",
                     "image_url": {"url": f"data:image/png;base64,{b64}"}},
                ]}],
                temperature=0.1,
                max_tokens=max_tokens,
                extra_body=variants[v],
            )
            text = resp.choices[0].message.content or ""
            return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
        except Exception as e:
            msg = str(e).lower()
            print(f"Vision error (attempt {attempt + 1}):", str(e)[:200])
            if v == 0 and "reasoning" in msg:        # model rejects that setting
                v = 1
                continue
            if any(w in msg for w in ("model_not_found", "does not exist",
                                      "decommissioned", "not support", "invalid image")):
                print("-> Check VISION_MODEL in .env (must be an image-capable model).")
                return ""
            if "request too large" in msg and max_tokens > 200:
                max_tokens = int(max_tokens * 0.6)   # ask for a shorter caption
                continue
            if "429" in msg or "rate_limit" in msg:
                time.sleep(30)                        # let the minute window reset
                continue
            time.sleep(2 ** (attempt + 1))
    return ""


# ---------- 2. Chunking helpers --------------------------------------------
def split_text(text: str) -> list[str]:
    """Split text into ~CHUNK_SIZE pieces on paragraph boundaries, with overlap."""
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks, current = [], ""
    for para in paragraphs:
        while len(para) > CHUNK_SIZE:            # one giant paragraph: hard-cut it
            chunks.append(para[:CHUNK_SIZE])
            para = para[CHUNK_SIZE - CHUNK_OVERLAP:]
        if len(current) + len(para) + 2 <= CHUNK_SIZE:
            current = f"{current}\n\n{para}" if current else para
        else:
            chunks.append(current)
            current = f"{current[-CHUNK_OVERLAP:]}\n\n{para}"   # overlap
    if current.strip():
        chunks.append(current)
    return [c for c in chunks if c.strip()]


def split_table(md: str, max_chars: int = 3000) -> list[str]:
    """Split a huge Markdown table by rows, repeating the header in each part."""
    if len(md) <= max_chars:
        return [md]
    lines = md.strip().split("\n")
    header, rows = lines[:2], lines[2:]
    parts, cur = [], []
    for row in rows:
        if sum(len(r) for r in cur) + len(row) > max_chars and cur:
            parts.append("\n".join(header + cur))
            cur = []
        cur.append(row)
    if cur:
        parts.append("\n".join(header + cur))
    return parts


def split_lines(text: str, size: int = 1500, overlap_lines: int = 2) -> list[str]:
    """Chunk OCR text on LINE boundaries so a table row is never cut in half."""
    lines = text.split("\n")
    chunks, cur, n = [], [], 0
    for ln in lines:
        if n + len(ln) > size and cur:
            chunks.append("\n".join(cur))
            cur = cur[-overlap_lines:]
            n = sum(len(x) for x in cur)
        cur.append(ln)
        n += len(ln) + 1
    if cur:
        chunks.append("\n".join(cur))
    return chunks


# ---------- 3. Page-level extraction ---------------------------------------
def extract_page_images(doc, page) -> list[bytes]:
    """PNG bytes of the meaningful embedded images on a page, largest first."""
    found = []
    for img in page.get_images(full=True):
        pix = fitz.Pixmap(doc, img[0])
        if pix.width < MIN_IMAGE_PX or pix.height < MIN_IMAGE_PX:
            continue                               # icon / logo / line
        if pix.n - pix.alpha >= 4:                 # CMYK -> RGB
            pix = fitz.Pixmap(fitz.csRGB, pix)
        png = pix.tobytes("png")
        if len(png) < 3_500_000:
            found.append((pix.width * pix.height, png))
    found.sort(key=lambda x: -x[0])                # biggest = most informative
    return [p for _, p in found[:MAX_IMAGES_PER_PAGE]]


def process_page(doc, page, page_no: int, filename: str, img_budget: list,
                 use_ocr: bool = True) -> list[dict]:
    """Turn ONE page into chunks: tables + text + OCR + diagram captions."""
    def make(text, kind):
        return {"text": text,
                "meta": {"file": filename, "page": page_no, "type": kind}}

    chunks = []

    # (a) Real (vector) tables -> Markdown
    tables = page.find_tables().tables
    table_boxes = []
    for t in tables:
        table_boxes.append(fitz.Rect(t.bbox))
        md = t.to_markdown()
        if md.strip():
            for part in split_table(md):
                chunks.append(make(f"Table from page {page_no}:\n{part}", "table"))

    # (b) Normal text, skipping blocks that sit inside a table
    kept = []
    for b in page.get_text("blocks", sort=True):
        x0, y0, x1, y1, txt, _, btype = b[:7]
        if btype != 0:
            continue
        if any(fitz.Rect(x0, y0, x1, y1).intersects(tb) for tb in table_boxes):
            continue
        kept.append(txt.strip())
    page_text = "\n\n".join(kept)
    if len(page_text) >= 20:
        for piece in split_text(page_text):
            chunks.append(make(piece, "text"))

    # (c) Image-only page (text is baked into pixels)? -> LOCAL OCR, no API needed
    ocr_done = False
    if len(page_text) < OCR_MIN_CHARS and not tables and use_ocr and ocr.available():
        text = ocr.ocr_page_text(page)
        if len(text.strip()) > 20:
            head = text.split("\n", 1)[0][:80]
            for piece in split_lines(text):
                chunks.append(make(f"[Page {page_no} OCR | starts: {head}]\n{piece}", "ocr"))
            ocr_done = True

    # (d) No OCR result: fall back to the vision model (rate-limited, uses image budget)
    if not ocr_done:
        if len(page_text) < 50 and not tables:            # scanned page
            if img_budget[0] > 0:
                img_budget[0] -= 1
                png = page.get_pixmap(dpi=130).tobytes("png")
                scan = describe_image(png, SCAN_MAX_TOKENS)
                if scan:
                    chunks.append(make(scan, "scanned_page"))
            else:
                print(f"Page {page_no} has no text and the image budget is used up; skipped.")
        elif img_budget[0] > 0:                           # diagrams / charts
            for png in extract_page_images(doc, page):
                if img_budget[0] <= 0:
                    break
                img_budget[0] -= 1
                caption = describe_image(png)
                if caption:
                    chunks.append(make(f"Figure on page {page_no}: {caption}", "figure"))
    return chunks


def ingest_pdf(file_bytes: bytes, filename: str, progress=None,
               max_images: int = DEFAULT_MAX_IMAGES, use_ocr: bool = True) -> list[dict]:
    """
    Whole PDF -> all chunks.
    progress(done_pages, total_pages, message) is called before AND after every page.
    """
    doc = fitz.open(stream=file_bytes, filetype="pdf")
    total = len(doc)
    img_budget = [max_images]
    say = progress or (lambda d, t, m: None)

    if use_ocr and ocr.available():
        say(0, total, "loading OCR engine (a few seconds the first time)…")
        ocr.warmup()

    all_chunks, started = [], time.time()
    for i, page in enumerate(doc, start=1):
        say(i - 1, total, f"page {i}/{total}: working…")
        t0 = time.time()
        new = process_page(doc, page, i, filename, img_budget, use_ocr)
        all_chunks.extend(new)

        kinds = Counter(c["meta"]["type"] for c in new)
        found = ", ".join(f"{k}:{v}" for k, v in kinds.items()) or "nothing found"
        elapsed = time.time() - started
        eta = elapsed / i * (total - i)
        line = f"page {i}/{total} | {found} | {time.time() - t0:.1f}s | ETA ~{eta:.0f}s"
        print(f"[{filename}] {line}")          # also visible in the terminal
        say(i, total, line)
    return all_chunks
