"""
See everything stored in Chroma.

  python view_chroma.py                         summary + export ALL chunks to chroma_export.csv
  python view_chroma.py --file CPVC-Catalogue.pdf --page 5      print that page's chunks
  python view_chroma.py --search "reducing bush sizes"          see what a search returns
  python view_chroma.py --notebook my_notebook                  limit to one notebook

Open chroma_export.csv in Excel to browse, filter and search all stored text.
"""
import argparse
import csv
import html
import webbrowser
from pathlib import Path
import chromadb


# ---- HTML_WRITER_START ----
def write_html(rows, path="chroma_export.html"):
    """Readable, searchable page: one card per chunk, full text visible."""
    cards = []
    for r in rows:
        cards.append(
            f'<div class="card" data-type="{html.escape(r["type"])}">'
            f'<div class="meta"><b>{html.escape(r["file"])}</b> &middot; page {r["page"]} '
            f'&middot; <span class="tag">{html.escape(r["type"])}</span> '
            f'&middot; {r["chars"]} chars &middot; notebook: {html.escape(r["notebook"])}</div>'
            f'<pre>{html.escape(r["text"])}</pre></div>')
    types = sorted({r["type"] for r in rows})
    options = "".join(f'<option value="{t}">{t}</option>' for t in types)
    page = f"""<!doctype html><html><head><meta charset="utf-8"><title>Chroma export</title>
<style>
body{{font-family:Segoe UI,Arial,sans-serif;margin:0;background:#f4f5f7;color:#1c1e21}}
header{{position:sticky;top:0;background:#fff;padding:12px 20px;border-bottom:1px solid #ddd;display:flex;gap:12px;align-items:center}}
input,select{{padding:8px;font-size:15px}} input{{flex:1}}
.card{{background:#fff;margin:12px 20px;border-radius:8px;padding:12px 16px;box-shadow:0 1px 3px rgba(0,0,0,.1)}}
.meta{{font-size:13px;color:#555;margin-bottom:8px}}
.tag{{background:#e3edff;color:#1a4fb8;padding:1px 8px;border-radius:10px}}
pre{{white-space:pre-wrap;word-break:break-word;font-family:Consolas,monospace;font-size:13px;margin:0}}
</style></head><body>
<header><input id="q" placeholder="Search all {len(rows)} chunks (e.g. 'reducing bush', 'F 90 CP 020 C')..." oninput="f()">
<select id="t" onchange="f()"><option value="">all types</option>{options}</select>
<span id="n"></span></header>
{''.join(cards)}
<script>
function f(){{var q=document.getElementById('q').value.toLowerCase(),t=document.getElementById('t').value,c=document.querySelectorAll('.card'),k=0;
c.forEach(function(e){{var ok=e.textContent.toLowerCase().includes(q)&&(!t||e.dataset.type===t);e.style.display=ok?'':'none';if(ok)k++;}});
document.getElementById('n').textContent=k+' shown';}} f();
</script></body></html>"""
    Path(path).write_text(page, encoding="utf-8")
    return Path(path).resolve()
# ---- HTML_WRITER_END ----

ap = argparse.ArgumentParser()
ap.add_argument("--notebook")
ap.add_argument("--file")
ap.add_argument("--page", type=int)
ap.add_argument("--search")
ap.add_argument("--path", default="data/chroma")
args = ap.parse_args()

client = chromadb.PersistentClient(path=args.path)
names = [c if isinstance(c, str) else c.name for c in client.list_collections()]
if args.notebook:
    names = [n for n in names if n == args.notebook]
if not names:
    raise SystemExit(f"No notebooks found in '{args.path}'. Run this from your project folder.")

rows = []
for nb in names:
    col = client.get_collection(nb)

    # --- semantic search mode ---
    if args.search:
        res = col.query(query_texts=[args.search], n_results=5)
        print(f"\n=== Top 5 results in '{nb}' for: {args.search} ===")
        for d, m in zip(res["documents"][0], res["metadatas"][0]):
            print(f"\n[{m['file']} | p.{m['page']} | {m['type']}]\n{d[:600]}")
        continue

    data = col.get(include=["documents", "metadatas"])
    for d, m in zip(data["documents"], data["metadatas"]):
        rows.append({"notebook": nb, "file": m["file"], "page": m["page"],
                     "type": m["type"], "chars": len(d), "text": d})

if args.search:
    raise SystemExit

rows.sort(key=lambda r: (r["notebook"], r["file"], r["page"], r["type"]))

# --- summary table ---
print(f"{'NOTEBOOK':<18}{'FILE':<30}{'CHUNKS':>7}  TYPES")
seen = {}
for r in rows:
    seen.setdefault((r["notebook"], r["file"]), []).append(r["type"])
for (nb, f), types in seen.items():
    mix = ", ".join(f"{t}:{types.count(t)}" for t in sorted(set(types)))
    print(f"{nb:<18}{f:<30}{len(types):>7}  {mix}")

# --- filtered print mode ---
if args.file:
    print()
    for r in rows:
        if r["file"] == args.file and (args.page is None or r["page"] == args.page):
            print(f"=== p.{r['page']} | {r['type']} | {r['chars']} chars ===\n{r['text']}\n")
else:
    with open("chroma_export.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=["notebook", "file", "page", "type", "chars", "text"])
        w.writeheader()
        w.writerows(rows)
    print(f"\nExported {len(rows)} chunks to chroma_export.csv (Excel).")
    out = write_html(rows)
    print(f"Readable page: {out}  (opening in your browser...)")
    webbrowser.open(out.as_uri())
