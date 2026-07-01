"""Thin Groq client wrapper.

Exposes a single ``complete_json`` helper that asks the model for a JSON object,
parses it robustly, and retries on transient failures. Raises ``LLMUnavailable``
when the model can't be reached or returns nothing usable, so the agent can fall
back to deterministic retrieval instead of 500-ing the evaluator.
"""
from __future__ import annotations

import json
import re
import time
from functools import lru_cache
from typing import Any, Dict, List

from app.config import (
    GROQ_API_KEY,
    GROQ_MODEL,
    LLM_MAX_RETRIES,
    LLM_TIMEOUT,
)


class LLMUnavailable(RuntimeError):
    """Raised when the LLM cannot produce a usable JSON response."""


@lru_cache(maxsize=1)
def _client():
    if not GROQ_API_KEY:
        raise LLMUnavailable("GROQ_API_KEY is not set")
    try:
        from groq import Groq
    except ImportError as e:  # pragma: no cover
        raise LLMUnavailable("groq package not installed") from e
    return Groq(api_key=GROQ_API_KEY, timeout=LLM_TIMEOUT, max_retries=0)


_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def _extract_json(text: str) -> Dict[str, Any]:
    text = (text or "").strip()
    if not text:
        raise LLMUnavailable("empty LLM response")
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    m = _JSON_RE.search(text)
    if not m:
        raise LLMUnavailable("no JSON object in LLM response")
    return json.loads(m.group(0))


def complete_json(
    system: str,
    user: str,
    *,
    temperature: float = 0.1,
    max_tokens: int = 1024,
    model: str | None = None,
) -> Dict[str, Any]:
    """Return a parsed JSON object from the model, or raise ``LLMUnavailable``."""
    messages: List[dict] = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    last_err: Exception | None = None
    for attempt in range(LLM_MAX_RETRIES + 1):
        try:
            resp = _client().chat.completions.create(
                model=model or GROQ_MODEL,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                response_format={"type": "json_object"},
            )
            content = resp.choices[0].message.content
            return _extract_json(content)
        except LLMUnavailable:
            raise
        except Exception as e:  # network / rate-limit / parse
            last_err = e
            if attempt < LLM_MAX_RETRIES:
                wait = _retry_after_seconds(e)
                if wait is not None:
                    time.sleep(min(wait + 0.5, 8.0))
                continue
    raise LLMUnavailable(f"LLM call failed after retries: {last_err}")


_RETRY_RE = re.compile(r"try again in ([0-9.]+)s", re.IGNORECASE)


def _retry_after_seconds(err: Exception) -> float | None:
    """If this looks like a 429 rate-limit, return how long to wait, else None."""
    msg = str(err)
    if "429" not in msg and "rate limit" not in msg.lower():
        return None
    m = _RETRY_RE.search(msg)
    return float(m.group(1)) if m else 2.0
