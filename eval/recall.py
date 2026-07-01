"""Name resolution + Recall@K used by the eval harness.

Expected shortlist names from the traces are resolved to catalog ids by
normalized exact match, then a high-cutoff fuzzy fallback. We resolve to ids so a
recommendation and its labeled target are compared on identity, not on string
formatting differences.
"""
from __future__ import annotations

import difflib
import re
from typing import List

from app.catalog import get_catalog


def normalize(s: str) -> str:
    s = s.lower().replace("&", "and")
    s = re.sub(r"[^a-z0-9]+", " ", s)
    return " ".join(s.split())


def _index():
    catalog = get_catalog()
    by_norm = {normalize(a.name): a.id for a in catalog.items}
    return by_norm, list(by_norm)


def resolve_name(name: str) -> str | None:
    by_norm, norms = _index()
    n = normalize(name)
    if n in by_norm:
        return by_norm[n]
    match = difflib.get_close_matches(n, norms, n=1, cutoff=0.85)
    return by_norm[match[0]] if match else None


def resolve_expected(names: List[str]) -> List[str]:
    ids = []
    for nm in names:
        rid = resolve_name(nm)
        if rid is None:
            raise ValueError(f"Unresolved expected name: {nm!r}")
        ids.append(rid)
    return ids


def recall_at_k(recommended_ids: List[str], relevant_ids: List[str], k: int = 10) -> float:
    if not relevant_ids:
        return 0.0
    topk = set(recommended_ids[:k])
    hits = sum(1 for rid in relevant_ids if rid in topk)
    return hits / len(relevant_ids)
