"""Normalize the raw SHL catalog into the shape the app consumes.

Reads ``data/catalog_raw.json`` (the scraped catalog, which contains unescaped
control characters, so it is parsed with ``strict=False``) and writes
``data/catalog.json``: a list of clean records with a stable ``id``, the public
``url``, a derived ``test_type`` code string, and a precomputed ``search_text``
used by retrieval.

Run:  python -m scripts.build_catalog
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "catalog_raw.json"
OUT = ROOT / "data" / "catalog.json"

# SHL's eight test-type codes, keyed by the catalog's full "keys" labels.
KEY_TO_CODE = {
    "Ability & Aptitude": "A",
    "Biodata & Situational Judgment": "B",
    "Competencies": "C",
    "Development & 360": "D",
    "Assessment Exercises": "E",
    "Knowledge & Skills": "K",
    "Personality & Behavior": "P",
    "Simulations": "S",
}
# Stable ordering so multi-key test_type strings are deterministic ("K,S" not "S,K").
CODE_ORDER = ["A", "B", "C", "D", "E", "K", "P", "S"]


def load_raw() -> list[dict]:
    text = RAW.read_text(encoding="utf-8")
    return json.loads(text, strict=False)


def derive_test_type(keys: list[str]) -> str:
    """Map the catalog's ``keys`` labels to a comma-joined SHL code string."""
    codes = {KEY_TO_CODE[k] for k in keys if k in KEY_TO_CODE}
    ordered = [c for c in CODE_ORDER if c in codes]
    return ",".join(ordered)


def clean(text: str) -> str:
    return " ".join((text or "").split())


def build_search_text(item: dict) -> str:
    parts = [
        item.get("name", ""),
        item.get("description", ""),
        " ".join(item.get("keys", [])),
        " ".join(item.get("job_levels", [])),
    ]
    return clean(" ".join(parts))


def normalize(raw: list[dict]) -> list[dict]:
    out: list[dict] = []
    seen_ids: set[str] = set()
    for item in raw:
        entity_id = str(item.get("entity_id", "")).strip()
        name = clean(item.get("name", ""))
        url = clean(item.get("link", ""))
        if not entity_id or not name or not url:
            # Records without an id, name, or URL cannot be safely recommended.
            continue
        if entity_id in seen_ids:
            continue
        seen_ids.add(entity_id)
        out.append(
            {
                "id": entity_id,
                "name": name,
                "url": url,
                "test_type": derive_test_type(item.get("keys", [])),
                "keys": item.get("keys", []),
                "description": clean(item.get("description", "")),
                "job_levels": item.get("job_levels", []),
                "languages": item.get("languages", []),
                "duration": clean(item.get("duration", "")),
                "remote": item.get("remote", ""),
                "adaptive": item.get("adaptive", ""),
                "search_text": build_search_text(item),
            }
        )
    return out


def main() -> int:
    if not RAW.exists():
        print(f"ERROR: {RAW} not found. Download the catalog first.", file=sys.stderr)
        return 1
    raw = load_raw()
    records = normalize(raw)
    OUT.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {len(records)} records to {OUT} (from {len(raw)} raw entries).")
    # Quick distribution sanity print.
    from collections import Counter

    tt = Counter(r["test_type"] for r in records)
    print("test_type distribution:", dict(tt.most_common(10)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
