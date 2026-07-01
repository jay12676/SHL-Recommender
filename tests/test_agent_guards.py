"""Agent guard tests.

These exercise the deterministic guarantees (catalog-only items, honored edits,
turn budget, fallback) with a *fake* LLM so they run offline and reproducibly.
"""
from __future__ import annotations

import pytest

import app.agent as agent
from app.agent import _count_clarifications, _is_excluded, run_chat
from app.catalog import get_catalog
from app.schemas import Message


def _opq_id() -> str:
    cat = get_catalog()
    for a in cat.items:
        if a.name == "Occupational Personality Questionnaire OPQ32r":
            return a.id
    raise AssertionError("OPQ not in catalog")


class FakeLLM:
    """Returns queued responses in order, regardless of prompt."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = 0

    def __call__(self, system, user, **kwargs):
        self.calls += 1
        return self.responses.pop(0)


def test_is_excluded():
    assert _is_excluded("Occupational Personality Questionnaire OPQ32r", ["OPQ"])
    assert _is_excluded("RESTful Web Services (New)", ["rest"])
    assert not _is_excluded("Core Java (Advanced Level) (New)", ["OPQ"])


def test_count_clarifications():
    msgs = [
        Message(role="user", content="hi"),
        Message(role="assistant", content="What role?"),
        Message(role="user", content="java"),
        Message(role="assistant", content="Here you go."),
    ]
    assert _count_clarifications(msgs) == 1


def test_recommendations_are_catalog_grounded(monkeypatch):
    java = next(a for a in get_catalog().items if "Core Java" in a.name)
    understand = {
        "action": "recommend",
        "in_scope": True,
        "injection_detected": False,
        "expanded_query": "java developer",
        "test_type_hints": ["K"],
        "job_level_hint": "",
        "must_include": [],
        "must_exclude": [],
        "clarifying_question": "",
    }
    compose = {
        "reply": "Here is a shortlist.",
        # one real id + one fabricated id that must be dropped
        "selected_ids": [java.id, "999999-not-real"],
        "end_of_conversation": False,
    }
    monkeypatch.setattr(agent, "llm_enabled", lambda: True)
    monkeypatch.setattr(agent, "complete_json", FakeLLM([understand, compose]))

    resp = run_chat([Message(role="user", content="hiring a java developer")])
    urls = {a.url for a in get_catalog().items}
    assert resp.recommendations
    for rec in resp.recommendations:
        assert rec.url in urls  # nothing fabricated leaks out


def test_honors_drop_edit(monkeypatch):
    java = next(a for a in get_catalog().items if "Core Java" in a.name)
    understand = {
        "action": "recommend",
        "expanded_query": "java developer personality",
        "test_type_hints": ["K", "P"],
        "job_level_hint": "",
        "must_include": [],
        "must_exclude": ["OPQ"],
        "clarifying_question": "",
    }
    compose = {
        # model tries to include OPQ even though user dropped it
        "selected_ids": [java.id, _opq_id()],
        "reply": "Updated.",
        "end_of_conversation": False,
    }
    monkeypatch.setattr(agent, "llm_enabled", lambda: True)
    monkeypatch.setattr(agent, "complete_json", FakeLLM([understand, compose]))

    resp = run_chat(
        [
            Message(role="user", content="hiring a java developer"),
            Message(role="assistant", content="Here is a list."),
            Message(role="user", content="drop OPQ"),
        ]
    )
    assert all("OPQ" not in r.name for r in resp.recommendations)


def test_honors_add_edit(monkeypatch):
    java = next(a for a in get_catalog().items if "Core Java" in a.name)
    understand = {
        "action": "recommend",
        "expanded_query": "java developer",
        "test_type_hints": ["K"],
        "job_level_hint": "",
        "must_include": ["personality"],
        "must_exclude": [],
        "clarifying_question": "",
    }
    compose = {
        "selected_ids": [java.id],  # model forgot to add a personality test
        "reply": "Added.",
        "end_of_conversation": False,
    }
    monkeypatch.setattr(agent, "llm_enabled", lambda: True)
    monkeypatch.setattr(agent, "complete_json", FakeLLM([understand, compose]))

    resp = run_chat(
        [
            Message(role="user", content="hiring a java developer"),
            Message(role="assistant", content="Here is a list."),
            Message(role="user", content="add personality tests"),
        ]
    )
    assert any("P" in r.test_type.split(",") for r in resp.recommendations)


def test_clarify_budget_forces_recommendation(monkeypatch):
    """After the clarify budget is exhausted, a 'clarify' is overridden to recommend."""
    understand = {"action": "clarify", "clarifying_question": "one more?"}
    compose = {
        "selected_ids": [next(a for a in get_catalog().items if "Core Java" in a.name).id],
        "reply": "Here you go.",
        "end_of_conversation": False,
    }
    monkeypatch.setattr(agent, "llm_enabled", lambda: True)
    monkeypatch.setattr(agent, "complete_json", FakeLLM([understand, compose]))

    # Two prior assistant questions => budget (2) reached.
    msgs = [
        Message(role="user", content="something"),
        Message(role="assistant", content="q1?"),
        Message(role="user", content="answer1"),
        Message(role="assistant", content="q2?"),
        Message(role="user", content="answer2"),
    ]
    resp = run_chat(msgs)
    assert resp.recommendations  # did not clarify a third time


def test_refusal_has_empty_recommendations(monkeypatch):
    understand = {
        "action": "refuse",
        "clarifying_question": "I can only help with SHL assessments.",
    }
    monkeypatch.setattr(agent, "llm_enabled", lambda: True)
    monkeypatch.setattr(agent, "complete_json", FakeLLM([understand]))
    resp = run_chat([Message(role="user", content="what's the capital of France?")])
    assert resp.recommendations == []


def test_fallback_when_llm_disabled(monkeypatch):
    monkeypatch.setattr(agent, "llm_enabled", lambda: False)
    resp = run_chat([Message(role="user", content="I need an assessment")])
    # vague single turn -> clarify (no recommendation)
    assert resp.recommendations == []
