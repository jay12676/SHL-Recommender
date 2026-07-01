"""Render APPROACH.md to a compact, 2-page-friendly PDF (APPROACH.pdf)."""
from __future__ import annotations

from pathlib import Path

import markdown
from xhtml2pdf import pisa

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "APPROACH.md"
OUT = ROOT / "APPROACH.pdf"

CSS = """
@page { size: A4; margin: 1.1cm 1.3cm; }
body { font-family: Helvetica, Arial, sans-serif; font-size: 9pt; line-height: 1.28; color: #1a1a1a; }
h1 { font-size: 15pt; margin: 0 0 4pt 0; color: #0b6b3a; }
h2 { font-size: 10.5pt; margin: 8pt 0 2pt 0; color: #0b6b3a; border-bottom: 0.5pt solid #cfd8dc; padding-bottom: 1pt; }
p { margin: 2pt 0; }
ul { margin: 2pt 0 2pt 0; padding-left: 14pt; }
li { margin: 1pt 0; }
strong { color: #111; }
code { font-family: Consolas, monospace; font-size: 8pt; background: #f2f4f5; }
table { border-collapse: collapse; width: 100%; font-size: 8.5pt; margin: 3pt 0; }
th, td { border: 0.5pt solid #b0bec5; padding: 2pt 4pt; text-align: left; }
th { background: #eef2f3; }
"""


def main() -> int:
    md = SRC.read_text(encoding="utf-8")
    body = markdown.markdown(md, extensions=["tables", "fenced_code", "sane_lists"])
    html = f"<html><head><style>{CSS}</style></head><body>{body}</body></html>"
    with open(OUT, "wb") as f:
        result = pisa.CreatePDF(html, dest=f)
    if result.err:
        print("PDF generation had errors")
        return 1
    print(f"Wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
