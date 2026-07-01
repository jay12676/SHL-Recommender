"""Request/response schema enforcement tests (the non-negotiable contract)."""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.schemas import ChatRequest, ChatResponse, Recommendation


def test_request_requires_user_message():
    with pytest.raises(ValidationError):
        ChatRequest(messages=[{"role": "assistant", "content": "hi"}])


def test_request_rejects_empty_messages():
    with pytest.raises(ValidationError):
        ChatRequest(messages=[])


def test_response_defaults_to_empty_recommendations():
    r = ChatResponse(reply="hi")
    assert r.recommendations == []
    assert r.end_of_conversation is False


def test_response_caps_recommendations_at_ten():
    recs = [Recommendation(name=f"n{i}", url="http://x", test_type="K") for i in range(11)]
    with pytest.raises(ValidationError):
        ChatResponse(reply="x", recommendations=recs)


def test_response_serializes_to_contract_keys():
    r = ChatResponse(
        reply="ok",
        recommendations=[Recommendation(name="OPQ", url="http://x", test_type="P")],
        end_of_conversation=True,
    )
    data = r.model_dump()
    assert set(data) == {"reply", "recommendations", "end_of_conversation"}
    assert set(data["recommendations"][0]) == {"name", "url", "test_type"}
