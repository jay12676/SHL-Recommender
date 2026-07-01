"""Request/response models. The response shape is the non-negotiable API contract
from the assignment; Pydantic enforces it on the way out so we can never emit a
malformed body that would fail the evaluator's hard evals.
"""
from __future__ import annotations

from typing import List, Literal

from pydantic import BaseModel, Field, field_validator


class Message(BaseModel):
    role: Literal["user", "assistant", "system"]
    content: str


class ChatRequest(BaseModel):
    messages: List[Message] = Field(..., min_length=1)

    @field_validator("messages")
    @classmethod
    def must_contain_user(cls, v: List[Message]) -> List[Message]:
        if not any(m.role == "user" for m in v):
            raise ValueError("messages must contain at least one user message")
        return v


class Recommendation(BaseModel):
    name: str
    url: str
    test_type: str


class ChatResponse(BaseModel):
    reply: str
    # Empty while clarifying or refusing; 1-10 items once a shortlist is committed.
    recommendations: List[Recommendation] = Field(default_factory=list, max_length=10)
    end_of_conversation: bool = False


class HealthResponse(BaseModel):
    status: str = "ok"
