"""Hybrid retrieval over the catalog.

Core path is BM25 (lexical) which is strong on exact skills ("Java", "SQL") and
needs no model download. If ``USE_EMBEDDINGS`` is on and sentence-transformers is
importable, semantic scores are fused in via Reciprocal Rank Fusion to catch
matches like "works with stakeholders" -> "communication / influencing". If the
embedding model can't load, we silently fall back to BM25-only.

Retrieval is intentionally high-recall: test-type and job-level signals are *soft
boosts*, never hard filters, so a relevant assessment is never excluded outright.
The LLM compose step does the final precise selection from this candidate pool.
"""
from __future__ import annotations

import re
from functools import lru_cache
from typing import List, Tuple

from rank_bm25 import BM25Okapi

from app.catalog import Assessment, Catalog, get_catalog
from app.config import N_CANDIDATES, USE_EMBEDDINGS, EMBED_MODEL

_TOKEN_RE = re.compile(r"[a-z0-9+#.]+")

# Assessments that recur across personas as default components (the traces show
# OPQ32r / Verify G+ / GSA / Graduate Scenarios used as the standard personality /
# cognitive / SJT layers). We always make them *available* to the selector; we do
# not auto-recommend them.
CORE_SET_NAMES = (
    "Occupational Personality Questionnaire OPQ32r",
    "SHL Verify Interactive G+",
    "Global Skills Assessment",
    "Graduate Scenarios",
)

# Maps test-type codes to query terms used for the soft semantic/lexical boost.
TEST_TYPE_TERMS = {
    "A": "ability aptitude cognitive reasoning",
    "B": "biodata situational judgment scenarios",
    "C": "competencies skills",
    "D": "development 360 feedback report",
    "E": "assessment exercise",
    "K": "knowledge skills technical",
    "P": "personality behavior",
    "S": "simulation simulated",
}


def tokenize(text: str) -> List[str]:
    return _TOKEN_RE.findall(text.lower())


class Retriever:
    def __init__(self, catalog: Catalog):
        self.catalog = catalog
        self.items: List[Assessment] = catalog.items
        self._corpus_tokens = [tokenize(a.search_text) for a in self.items]
        self.bm25 = BM25Okapi(self._corpus_tokens)
        self._core_ids = [
            a.id for a in self.items if a.name in CORE_SET_NAMES
        ]
        self._embedder = None
        self._embeddings = None
        if USE_EMBEDDINGS:
            self._try_init_embeddings()

    def _try_init_embeddings(self) -> None:
        try:
            from sentence_transformers import SentenceTransformer

            self._embedder = SentenceTransformer(EMBED_MODEL)
            self._embeddings = self._embedder.encode(
                [a.search_text for a in self.items],
                normalize_embeddings=True,
                show_progress_bar=False,
            )
        except Exception:
            # Degrade gracefully to BM25-only.
            self._embedder = None
            self._embeddings = None

    @property
    def embeddings_active(self) -> bool:
        return self._embeddings is not None

    # --- scoring ---------------------------------------------------------

    def _bm25_ranking(self, query: str) -> List[int]:
        scores = self.bm25.get_scores(tokenize(query))
        order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        # Keep only positive-score hits to avoid padding with irrelevant noise.
        return [i for i in order if scores[i] > 0]

    def _embed_ranking(self, query: str) -> List[int]:
        if not self.embeddings_active:
            return []
        import numpy as np

        q = self._embedder.encode([query], normalize_embeddings=True)[0]
        sims = self._embeddings @ q
        order = list(np.argsort(-sims))
        return [int(i) for i in order]

    @staticmethod
    def _rrf(rankings: List[List[int]], k: int = 60) -> dict:
        """Reciprocal Rank Fusion over multiple ranked id-index lists."""
        fused: dict = {}
        for ranking in rankings:
            for rank, idx in enumerate(ranking):
                fused[idx] = fused.get(idx, 0.0) + 1.0 / (k + rank + 1)
        return fused

    def search(
        self,
        query: str,
        test_type_hints: List[str] | None = None,
        job_level_hint: str | None = None,
        top_n: int = N_CANDIDATES,
    ) -> List[Assessment]:
        query = (query or "").strip()
        if not query:
            # Nothing to search on: hand back the core set so compose still has
            # something grounded to work with.
            return [self.catalog.get(i) for i in self._core_ids]

        rankings = [self._bm25_ranking(query)]
        emb = self._embed_ranking(query)
        if emb:
            rankings.append(emb[: max(top_n * 3, 60)])

        fused = self._rrf(rankings)

        # Soft boosts: nudge items whose test_type or job_level matches the hint.
        hints = set(test_type_hints or [])
        jl = (job_level_hint or "").lower()
        for idx, score in list(fused.items()):
            item = self.items[idx]
            if hints and any(code in item.test_type.split(",") for code in hints):
                fused[idx] = score + 0.01
            if jl and any(jl in lvl.lower() for lvl in item.job_levels):
                fused[idx] = fused[idx] + 0.005

        ranked_idx = sorted(fused, key=lambda i: fused[i], reverse=True)
        result_ids = [self.items[i].id for i in ranked_idx[:top_n]]

        # Always make the curated core set available to the selector.
        for cid in self._core_ids:
            if cid not in result_ids:
                result_ids.append(cid)

        return [self.catalog.get(i) for i in result_ids if self.catalog.get(i)]


@lru_cache(maxsize=1)
def get_retriever() -> Retriever:
    return Retriever(get_catalog())
