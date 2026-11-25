"""
Look inside the saved memory from the terminal.
  python inspect_db.py                          -> list notebooks and files
  python inspect_db.py my_notebook file.pdf     -> print + save everything extracted
  python inspect_db.py my_notebook file.pdf 5   -> only page 5
"""
import sys
import rag

if len(sys.argv) < 3:
    for nb in rag.list_notebooks():
        print(f"Notebook: {nb}")
        for f in rag.list_sources(nb):
            print(f"   - {f}")
    sys.exit()

nb, fname = sys.argv[1], sys.argv[2]
page = int(sys.argv[3]) if len(sys.argv) > 3 else None
items = rag.get_chunks(nb, fname, page)
text = "\n\n".join(f"=== p.{m['page']} | {m['type']} ===\n{d}" for d, m in items)
print(text)
out = f"{fname}.extracted.txt"
open(out, "w", encoding="utf-8").write(text)
print(f"\n[{len(items)} chunks saved to {out}]")
