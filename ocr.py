"""
LOCAL OCR (runs on your CPU, no API, no rate limits).

Many catalogues are exported as page-sized pictures: the tables are pixels, not text.
We render the page, detect every text box, then rebuild ROWS by grouping boxes that
share a vertical position. Table rows come out like:
    F 90 CP 020 C | 20 | 27.5 | 16.5 | 12.0
"""
import numpy as np

try:                                    # current package: pip install rapidocr onnxruntime
    from rapidocr import RapidOCR
    _NEW_API = True
except ImportError:
    try:                                # legacy package: pip install rapidocr-onnxruntime
        from rapidocr_onnxruntime import RapidOCR
        _NEW_API = False
    except ImportError:
        RapidOCR = None
        _NEW_API = False

_engine = None


def available() -> bool:
    return RapidOCR is not None


def _get_engine():
    global _engine
    if _engine is None:
        _engine = RapidOCR()
    return _engine


def warmup():
    """Load the OCR models now, so the first page isn't mysteriously slow."""
    try:
        _get_engine()
    except Exception as e:
        print("OCR warmup error:", e)


def _detect(img_bgr) -> list[tuple]:
    """Return [(x0, y0, x1, y1, text), ...] for every detected text box."""
    eng = _get_engine()
    if _NEW_API:
        out = eng(img_bgr)
        boxes, txts = getattr(out, "boxes", None), getattr(out, "txts", None)
        if boxes is None or txts is None:
            return []
        pairs = zip(boxes, txts)
    else:
        res, _ = eng(img_bgr)
        if not res:
            return []
        pairs = ((b, t) for b, t, _score in res)
    items = []
    for box, text in pairs:
        pts = np.array(box)
        items.append((pts[:, 0].min(), pts[:, 1].min(),
                      pts[:, 0].max(), pts[:, 1].max(), str(text).strip()))
    return [i for i in items if i[4]]


def _to_rows(items: list[tuple]) -> str:
    """Group boxes into visual rows (same height), order left->right, join with ' | '."""
    if not items:
        return ""
    heights = sorted(i[3] - i[1] for i in items)
    med = heights[len(heights) // 2] or 10
    items = sorted(items, key=lambda i: (i[1] + i[3]) / 2)
    rows, cur = [], [items[0]]
    cy = (items[0][1] + items[0][3]) / 2
    for it in items[1:]:
        yc = (it[1] + it[3]) / 2
        if abs(yc - cy) <= 0.6 * med:
            cur.append(it)
            cy = sum((i[1] + i[3]) / 2 for i in cur) / len(cur)
        else:
            rows.append(cur)
            cur, cy = [it], yc
    rows.append(cur)
    return "\n".join(" | ".join(i[4] for i in sorted(r, key=lambda i: i[0])) for r in rows)


def ocr_page_text(page, dpi: int = 200) -> str:
    """Render a PyMuPDF page and return its text as table-aware rows."""
    try:
        pix = page.get_pixmap(dpi=dpi, alpha=False)
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 3)
        img = np.ascontiguousarray(img[:, :, ::-1])          # RGB -> BGR for OpenCV
        return _to_rows(_detect(img))
    except Exception as e:
        print("OCR error:", e)
        return ""
