"""Streamlit UI. Run with:  streamlit run app.py"""
import streamlit as st
from ingest import ingest_pdf
import rag
import ocr
from config import DEFAULT_MAX_IMAGES

st.set_page_config(page_title="My NotebookLM", page_icon="📓", layout="wide")
st.title("📓 My NotebookLM Clone")

# ======================= SIDEBAR =======================
with st.sidebar:
    st.header("Notebook")
    existing = rag.list_notebooks()
    choice = st.selectbox("Open", ["➕ New notebook"] + existing)
    if choice.startswith("➕"):
        nb = rag.safe_name(st.text_input("New notebook name", "my_notebook"))
    else:
        nb = choice
        if st.button("🗑 Delete this notebook"):
            rag.delete_notebook(nb)
            st.rerun()

    st.header("Add sources")
    files = st.file_uploader("PDFs", type="pdf", accept_multiple_files=True)
    use_ocr = st.checkbox("Local OCR for image-only pages", value=True,
                          help="Reads tables/text baked into images. Runs on your CPU, no API.")
    if use_ocr and not ocr.available():
        st.warning("OCR not installed. Run: pip install rapidocr onnxruntime")
    use_vision = st.checkbox("Caption diagrams/scans with AI", value=True,
                             help="Slow on the free tier: ~30 seconds per image.")
    max_imgs = st.slider("Max images to caption per file", 0, 30, DEFAULT_MAX_IMAGES,
                         disabled=not use_vision)
    if st.button("Process files", disabled=not files):
        already = rag.list_sources(nb)
        for f in files:
            if f.name in already:
                st.info(f"{f.name} already indexed, skipped.")
                continue
            bar = st.progress(0.0, text=f"{f.name} — starting…")
            chunks = ingest_pdf(
                f.read(), f.name, max_images=(max_imgs if use_vision else 0), use_ocr=use_ocr,
                progress=lambda d, t, msg, n=f.name: bar.progress(min(d / t, 1.0), text=f"{n} — {msg}"))
            bar.progress(1.0, text=f"{f.name}: saving + summarising…")
            rag.add_chunks(nb, chunks)
            rag.summarize_source(nb, f.name, chunks)
            bar.empty()
            from collections import Counter
            mix = ", ".join(f"{k} {v}" for k, v in Counter(c["meta"]["type"] for c in chunks).items())
            st.success(f"{f.name}: {len(chunks)} chunks ({mix})")
        st.rerun()

    st.header("Sources")
    sources = rag.list_sources(nb)
    summaries = rag.get_summaries(nb)
    selected = []
    for s in sources:
        if st.checkbox(s, value=True, key=f"sel_{nb}_{s}"):
            selected.append(s)
        with st.expander("summary / delete"):
            st.caption(summaries.get(s, "No summary."))
            if st.button("🔄 Regenerate summary", key=f"re_{nb}_{s}"):
                with st.spinner("Summarising…"):
                    rag.resummarize(nb, s)
                st.rerun()
            if st.button("🗑 Delete file", key=f"del_{nb}_{s}"):
                rag.delete_source(nb, s)
                st.rerun()

# ======================= INSPECTOR =======================
with st.expander("🔍 Inspect what was extracted and saved (OCR, text, tables)"):
    if not sources:
        st.caption("Nothing indexed yet.")
    else:
        from collections import Counter
        f = st.selectbox("File", sources, key="insp_file")
        all_items = rag.get_chunks(nb, f)
        pages = sorted({m["page"] for _, m in all_items})
        p = st.selectbox("Page", ["All"] + pages, key="insp_page")
        items = all_items if p == "All" else [x for x in all_items if x[1]["page"] == p]
        kinds = Counter(m["type"] for _, m in items)
        st.caption(" · ".join(f"{k}: {v}" for k, v in kinds.items()) + f"  —  {len(items)} chunks")
        for d, m in items[:60]:
            st.markdown(f"**p.{m['page']} · {m['type']}**")
            st.code(d, language=None)
        if len(items) > 60:
            st.caption(f"Showing first 60 of {len(items)}. Pick a page or download everything.")
        full = "\n\n".join(f"=== p.{m['page']} | {m['type']} ===\n{d}" for d, m in all_items)
        st.download_button("⬇ Download all extracted text", full,
                           file_name=f"{f}.extracted.txt")

# ======================= CHAT =======================
key = f"chat_{nb}"
st.session_state.setdefault(key, [])
history = st.session_state[key]


def show_sources(hits, note=None):
    with st.expander("📎 Sources used" + (f" — {note}" if note else "")):
        for doc, m in hits:
            st.markdown(f"**{m['file']} · p.{m['page']} · {m['type']}**")
            st.caption(doc[:400] + ("…" if len(doc) > 400 else ""))


for msg in history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("hits"):
            show_sources(msg["hits"], msg.get("note"))

# Generator buttons (Briefing / FAQ / Study guide)
action = None
cols = st.columns(3)
for col, kind in zip(cols, rag.OVERVIEW_PROMPTS):
    if col.button(f"✨ {kind}", disabled=not selected, use_container_width=True):
        action = kind

question = st.chat_input("Ask about your documents…", disabled=not selected)

if action:
    with st.chat_message("user"):
        st.markdown(f"Generate: **{action}**")
    with st.chat_message("assistant"):
        answer = st.write_stream(rag.stream_overview(nb, selected, action))
    history += [{"role": "user", "content": f"Generate: {action}"},
                {"role": "assistant", "content": answer}]

elif question:
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        standalone = rag.rewrite_question(question, history)      # step 1
        hits, note = rag.retrieve(nb, standalone, selected)       # steps 2+3
        answer = st.write_stream(rag.stream_answer(question, hits, history))  # step 4
        show_sources(hits, note)
    history += [{"role": "user", "content": question},
                {"role": "assistant", "content": answer, "hits": hits, "note": note}]
