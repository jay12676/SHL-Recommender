"""Retrieval sanity tests (no LLM required)."""
from __future__ import annotations

from app.retrieval import get_retriever


def test_java_query_surfaces_java_tests():
    r = get_retriever()
    results = r.search("core java developer backend", top_n=20)
    names = " ".join(a.name.lower() for a in results)
    assert "java" in names


def test_excel_query_surfaces_excel_tests():
    r = get_retriever()
    results = r.search("microsoft excel spreadsheet admin assistant", top_n=20)
    assert any("excel" in a.name.lower() for a in results)


def test_core_set_always_present():
    r = get_retriever()
    results = r.search("some unrelated random query about widgets", top_n=10)
    names = {a.name for a in results}
    assert "Occupational Personality Questionnaire OPQ32r" in names


def test_empty_query_returns_core_set():
    r = get_retriever()
    results = r.search("")
    assert len(results) >= 1


def test_results_are_catalog_items():
    r = get_retriever()
    results = r.search("sql database")
    for a in results:
        assert a.url.startswith("http")
