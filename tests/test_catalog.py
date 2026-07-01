"""Catalog loader + test_type derivation tests."""
from __future__ import annotations

from app.catalog import get_catalog
from scripts.build_catalog import derive_test_type


def test_catalog_loads_and_is_nonempty():
    cat = get_catalog()
    assert len(cat) > 300


def test_every_record_has_required_fields():
    cat = get_catalog()
    for a in cat.items:
        assert a.id and a.name and a.url
        assert a.url.startswith("http")


def test_ids_are_unique():
    cat = get_catalog()
    ids = [a.id for a in cat.items]
    assert len(ids) == len(set(ids))


def test_get_by_id_roundtrip():
    cat = get_catalog()
    sample = cat.items[0]
    assert cat.get(sample.id) is sample
    assert cat.get("does-not-exist") is None


def test_test_type_mapping():
    assert derive_test_type(["Knowledge & Skills"]) == "K"
    assert derive_test_type(["Personality & Behavior"]) == "P"
    # Multiple keys are comma-joined in a stable order.
    assert derive_test_type(["Simulations", "Knowledge & Skills"]) == "K,S"
    assert derive_test_type(["Ability & Aptitude"]) == "A"
    assert derive_test_type([]) == ""


def test_known_assessment_present():
    cat = get_catalog()
    names = {a.name for a in cat.items}
    assert "Occupational Personality Questionnaire OPQ32r" in names
