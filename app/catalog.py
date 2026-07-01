"""In-memory catalog: load the normalized records once and index them by id.

The catalog is the single source of truth for ``name``, ``url``, and
``test_type``. The agent only ever references catalog *ids*; this module turns an
id back into a grounded record, which is what makes hallucinated URLs impossible.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Dict, List

from app.config import CATALOG_PATH


@dataclass(frozen=True)
class Assessment:
    id: str
    name: str
    url: str
    test_type: str
    description: str
    keys: tuple
    job_levels: tuple
    languages: tuple
    duration: str
    remote: str
    adaptive: str
    search_text: str


class Catalog:
    def __init__(self, records: List[dict]):
        self._items: List[Assessment] = [self._to_assessment(r) for r in records]
        self._by_id: Dict[str, Assessment] = {a.id: a for a in self._items}

    @staticmethod
    def _to_assessment(r: dict) -> Assessment:
        return Assessment(
            id=str(r["id"]),
            name=r["name"],
            url=r["url"],
            test_type=r.get("test_type", ""),
            description=r.get("description", ""),
            keys=tuple(r.get("keys", [])),
            job_levels=tuple(r.get("job_levels", [])),
            languages=tuple(r.get("languages", [])),
            duration=r.get("duration", ""),
            remote=r.get("remote", ""),
            adaptive=r.get("adaptive", ""),
            search_text=r.get("search_text", r.get("name", "")),
        )

    def __len__(self) -> int:
        return len(self._items)

    @property
    def items(self) -> List[Assessment]:
        return self._items

    def get(self, item_id: str) -> Assessment | None:
        return self._by_id.get(str(item_id))

    def exists_url(self, url: str) -> bool:
        return any(a.url == url for a in self._items)


@lru_cache(maxsize=1)
def get_catalog(path: str | None = None) -> Catalog:
    p = Path(path) if path else CATALOG_PATH
    if not p.exists():
        raise FileNotFoundError(
            f"Catalog not found at {p}. Run `python -m scripts.build_catalog` first."
        )
    records = json.loads(p.read_text(encoding="utf-8"))
    return Catalog(records)
