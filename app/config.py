"""Runtime configuration, read once from the environment.

All knobs are environment variables so the same image runs locally and on a free
host (Render/HF Spaces) without code changes.
"""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CATALOG_PATH = Path(os.getenv("CATALOG_PATH", ROOT / "data" / "catalog.json"))

# --- LLM (Groq) ---
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
# 8b-instant has higher free-tier throughput (RPM/TPM/TPD) than 70b and is more
# than capable of the structured routing/selection tasks here. Override with
# GROQ_MODEL=llama-3.3-70b-versatile on a paid key for marginally better quality.
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
# Per-call timeout (seconds). Two calls worst-case must fit the evaluator's 30 s.
LLM_TIMEOUT = float(os.getenv("LLM_TIMEOUT", "12"))
LLM_MAX_RETRIES = int(os.getenv("LLM_MAX_RETRIES", "3"))

# --- Retrieval ---
# Embeddings are optional: if sentence-transformers/torch are unavailable or this
# flag is off, retrieval degrades gracefully to BM25-only.
USE_EMBEDDINGS = os.getenv("USE_EMBEDDINGS", "false").lower() in {"1", "true", "yes"}
EMBED_MODEL = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
N_CANDIDATES = int(os.getenv("N_CANDIDATES", "50"))
# How many candidates are actually shown to the compose LLM call (keeps token
# usage under the Groq free-tier per-minute limit). Core-set items are added on
# top of this slice.
COMPOSE_CANDIDATES = int(os.getenv("COMPOSE_CANDIDATES", "24"))

# --- Agent behavior ---
MAX_CLARIFYING_TURNS = int(os.getenv("MAX_CLARIFYING_TURNS", "2"))
MAX_RECOMMENDATIONS = 10  # hard cap from the API contract


def llm_enabled() -> bool:
    return bool(GROQ_API_KEY)
