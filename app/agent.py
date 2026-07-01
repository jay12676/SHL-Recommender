"""Agent orchestration: turn a stateless message history into a grounded reply.

Flow per turn:
    understand (LLM) -> route -> [retrieve -> compose (LLM)] -> ground -> guard

Every externally observable guarantee (catalog-only items, honored edits, turn
budget, schema validity, never 500) is enforced deterministically in Python here,
not left to the model.
"""
from __future__ import annotations

from typing import List

from app.catalog import Assessment, get_catalog
from app.config import (
    COMPOSE_CANDIDATES,
    MAX_CLARIFYING_TURNS,
    MAX_RECOMMENDATIONS,
    llm_enabled,
)
from app.llm import LLMUnavailable, complete_json
from app.prompts import (
    COMPOSE_SYSTEM,
    UNDERSTAND_SYSTEM,
    build_compose_user,
    build_understand_user,
)
from app.retrieval import CORE_SET_NAMES, get_retriever
from app.schemas import ChatResponse, Message, Recommendation

# Topic words (from explicit "add X" edits) -> test-type code they imply.
INCLUDE_CODE = {
    "personality": "P",
    "behaviour": "P",
    "behavior": "P",
    "cognitive": "A",
    "aptitude": "A",
    "reasoning": "A",
    "ability": "A",
    "knowledge": "K",
    "technical": "K",
    "simulation": "S",
    "situational": "B",
    "judgement": "B",
    "judgment": "B",
    "competency": "C",
    "competencies": "C",
}

_DEFAULT_REFUSAL = (
    "I can only help you choose SHL assessments from our catalog, so I can't help "
    "with that. If you tell me about the role you're hiring for, I'll suggest "
    "relevant assessments."
)


def _to_recommendations(items: List[Assessment]) -> List[Recommendation]:
    return [
        Recommendation(name=a.name, url=a.url, test_type=a.test_type) for a in items
    ]


def _count_clarifications(messages: List[Message]) -> int:
    """Assistant questions already asked (proxy for clarify turns used)."""
    return sum(
        1
        for m in messages
        if m.role == "assistant" and "?" in m.content
    )


def _is_excluded(name: str, exclude_terms: List[str]) -> bool:
    n = name.lower()
    return any(t.strip() and t.strip().lower() in n for t in exclude_terms)


def _last_user_text(messages: List[Message]) -> str:
    for m in reversed(messages):
        if m.role == "user":
            return m.content
    return ""


def _all_user_text(messages: List[Message]) -> str:
    return " ".join(m.content for m in messages if m.role == "user")


def _is_vague_first_turn(messages: List[Message]) -> bool:
    """Heuristic used only in the no-LLM fallback path."""
    users = [m for m in messages if m.role == "user"]
    if len(users) != 1:
        return False
    text = users[0].content.lower()
    if len(text.split()) > 12:
        return False
    vague_markers = ("need an assessment", "need a solution", "need a test", "help me", "what do you have")
    return any(v in text for v in vague_markers) or len(text.split()) <= 5


# --------------------------------------------------------------------------
# Deterministic fallback (no LLM / LLM failure) — keeps responses schema-valid.
# --------------------------------------------------------------------------

def _deterministic_recommend(messages: List[Message]) -> ChatResponse:
    retriever = get_retriever()
    query = _all_user_text(messages)
    candidates = retriever.search(query, top_n=8)
    items = candidates[:5] if candidates else []
    if not items:
        return ChatResponse(
            reply=(
                "Could you tell me a bit more about the role and the skills you want "
                "to assess?"
            ),
            recommendations=[],
            end_of_conversation=False,
        )
    return ChatResponse(
        reply="Here are assessments from the SHL catalog that fit what you described.",
        recommendations=_to_recommendations(items),
        end_of_conversation=False,
    )


def _fallback(messages: List[Message]) -> ChatResponse:
    if _is_vague_first_turn(messages):
        return ChatResponse(
            reply=(
                "Happy to help. What role are you hiring for, and what do you most "
                "want to assess (skills, seniority, or context)?"
            ),
            recommendations=[],
            end_of_conversation=False,
        )
    return _deterministic_recommend(messages)


# --------------------------------------------------------------------------
# Main entry point
# --------------------------------------------------------------------------

def run_chat(messages: List[Message]) -> ChatResponse:
    if not llm_enabled():
        return _fallback(messages)
    try:
        return _run_chat_llm(messages)
    except LLMUnavailable:
        return _fallback(messages)


def _run_chat_llm(messages: List[Message]) -> ChatResponse:
    clarifications_used = _count_clarifications(messages)

    understanding = complete_json(
        UNDERSTAND_SYSTEM,
        build_understand_user(messages, clarifications_used, MAX_CLARIFYING_TURNS),
        temperature=0.0,
        max_tokens=600,
    )

    action = str(understanding.get("action", "recommend")).lower()
    must_exclude = [str(x) for x in understanding.get("must_exclude", []) if x]
    must_include = [str(x) for x in understanding.get("must_include", []) if x]

    # Guard: never exhaust the 8-turn cap clarifying.
    if action == "clarify" and clarifications_used >= MAX_CLARIFYING_TURNS:
        action = "recommend"

    if action == "refuse":
        question = str(understanding.get("clarifying_question", "")).strip()
        return ChatResponse(
            reply=question or _DEFAULT_REFUSAL,
            recommendations=[],
            end_of_conversation=False,
        )

    if action == "clarify":
        question = str(understanding.get("clarifying_question", "")).strip()
        return ChatResponse(
            reply=question
            or "Could you tell me more about the role and what you want to assess?",
            recommendations=[],
            end_of_conversation=False,
        )

    # --- recommend path ---
    return _recommend(messages, understanding, must_include, must_exclude)


def _recommend(
    messages: List[Message],
    understanding: dict,
    must_include: List[str],
    must_exclude: List[str],
) -> ChatResponse:
    catalog = get_catalog()
    retriever = get_retriever()

    # Combine the LLM's expanded query with the literal user text so exact skill
    # terms (SQL, Docker, Excel, "SVAR US") are never paraphrased out of retrieval.
    expanded = str(understanding.get("expanded_query", ""))
    query = f"{expanded} {_all_user_text(messages)}".strip()
    test_type_hints = [str(c) for c in understanding.get("test_type_hints", []) if c]
    job_level_hint = str(understanding.get("job_level_hint", ""))

    candidates = retriever.search(
        query, test_type_hints=test_type_hints, job_level_hint=job_level_hint
    )
    # Remove explicitly excluded items before the model ever sees them.
    candidates = [a for a in candidates if not _is_excluded(a.name, must_exclude)]
    if not candidates:
        return _deterministic_recommend(messages)

    # Trim what the LLM sees to keep token usage under the free-tier limit, while
    # always retaining the curated core-set items (which sit at the tail).
    compose_pool = list(candidates[:COMPOSE_CANDIDATES])
    pool_ids = {a.id for a in compose_pool}
    for a in candidates:
        if a.name in CORE_SET_NAMES and a.id not in pool_ids:
            compose_pool.append(a)
            pool_ids.add(a.id)

    try:
        composition = complete_json(
            COMPOSE_SYSTEM,
            build_compose_user(messages, compose_pool),
            temperature=0.1,
            max_tokens=600,
        )
    except LLMUnavailable:
        # Compose failed (e.g. rate-limited). Degrade to the top of the already
        # retrieved, edit-filtered candidates rather than naive BM25 — these are
        # the relevant items, so recall stays close to the retrieval ceiling.
        return ChatResponse(
            reply="Here are SHL assessments from the catalog that match your requirements.",
            recommendations=_to_recommendations(candidates[:8]),
            end_of_conversation=False,
        )

    cand_by_id = {a.id: a for a in candidates}
    selected_ids = [str(i) for i in composition.get("selected_ids", [])]

    chosen: List[Assessment] = []
    seen = set()
    for sid in selected_ids:
        item = cand_by_id.get(sid) or catalog.get(sid)
        if not item or item.id in seen:
            continue
        if _is_excluded(item.name, must_exclude):  # belt-and-suspenders on edits
            continue
        chosen.append(item)
        seen.add(item.id)

    chosen = _ensure_includes(chosen, candidates, must_include, seen)
    chosen = chosen[:MAX_RECOMMENDATIONS]

    if not chosen:
        # Model returned nothing usable; fall back to top candidates.
        chosen = [a for a in candidates if not _is_excluded(a.name, must_exclude)][:5]

    reply = str(composition.get("reply", "")).strip() or (
        "Here are SHL assessments that fit your requirements."
    )
    end = bool(composition.get("end_of_conversation", False))

    return ChatResponse(
        reply=reply,
        recommendations=_to_recommendations(chosen),
        end_of_conversation=end,
    )


def _ensure_includes(
    chosen: List[Assessment],
    candidates: List[Assessment],
    must_include: List[str],
    seen: set,
) -> List[Assessment]:
    """Guarantee that an explicit 'add X' edit is reflected in the shortlist."""
    if len(chosen) >= MAX_RECOMMENDATIONS:
        return chosen
    retriever = get_retriever()
    for term in must_include:
        code = INCLUDE_CODE.get(term.strip().lower())
        term_l = term.strip().lower()

        def matches(a: Assessment) -> bool:
            if code and code in a.test_type.split(","):
                return True
            return term_l in a.search_text.lower()

        if any(matches(a) for a in chosen):
            continue
        # Find the best candidate (then catalog-wide) that covers the topic.
        pool = candidates + retriever.search(term, top_n=10)
        for a in pool:
            if a.id in seen:
                continue
            if matches(a):
                chosen.append(a)
                seen.add(a.id)
                break
        if len(chosen) >= MAX_RECOMMENDATIONS:
            break
    return chosen
