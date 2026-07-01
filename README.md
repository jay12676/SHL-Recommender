---
title: SHL Recommender
emoji: 🧩
colorFrom: blue
colorTo: green
sdk: docker
app_port: 8000
pinned: false
---

# Conversational SHL Assessment Recommender

A stateless conversational agent that takes a user from a vague hiring intent
("I'm hiring a Java developer") to a grounded shortlist of **SHL Individual Test
Solutions** through dialogue. It clarifies when the request is too vague,
recommends 1–10 assessments once it has context, refines the shortlist when
constraints change, compares assessments on request, and refuses anything outside
the SHL catalog. Every returned URL comes from the scraped catalog — never the
model's imagination.

Built with FastAPI + Groq (free tier). See
[`APPROACH.md`](APPROACH.md) for the design write-up and
[`docs/design.md`](docs/design.md) for the full spec.

## API

Stateless: every `POST /chat` carries the full conversation history; the server
stores nothing.

### `GET /health`
```json
{ "status": "ok" }
```

### `POST /chat`
Request:
```json
{
  "messages": [
    {"role": "user", "content": "Hiring a Java developer who works with stakeholders"},
    {"role": "assistant", "content": "Sure. What is the seniority level?"},
    {"role": "user", "content": "Mid-level, around 4 years"}
  ]
}
```
Response:
```json
{
  "reply": "Here are assessments that fit a mid-level Java dev with stakeholder needs.",
  "recommendations": [
    {"name": "Core Java (Advanced Level) (New)", "url": "https://www.shl.com/...", "test_type": "K"},
    {"name": "Occupational Personality Questionnaire OPQ32r", "url": "https://www.shl.com/...", "test_type": "P"}
  ],
  "end_of_conversation": false
}
```
- `recommendations` is `[]` while clarifying or refusing; 1–10 items when a
  shortlist is committed.
- `test_type` codes: `A` Ability & Aptitude, `B` Biodata & Situational Judgment,
  `C` Competencies, `D` Development & 360, `E` Assessment Exercises,
  `K` Knowledge & Skills, `P` Personality & Behavior, `S` Simulations.

## Quickstart

```bash
# 1. Install (core, deploy-light — BM25 retrieval, no torch)
python -m pip install -r requirements.txt

# 2. Build the normalized catalog (data/catalog.json) from the scraped raw file
python -m scripts.build_catalog

# 3. Set your Groq key (free: https://console.groq.com/keys)
cp .env.example .env        # then edit GROQ_API_KEY=...
export GROQ_API_KEY=gsk_...  # or set in your shell / host dashboard

# 4. Run
uvicorn app.main:app --reload --port 8000

# health
curl localhost:8000/health
# chat
curl -X POST localhost:8000/chat -H "content-type: application/json" \
  -d '{"messages":[{"role":"user","content":"Hiring a mid-level Java developer; test core Java and SQL"}]}'
```

Without `GROQ_API_KEY` the service still runs and stays schema-valid via a
deterministic BM25 fallback (used for offline tests).

## Tests & evaluation

```bash
python -m pytest -q          # unit tests: schema, catalog, retrieval, guards
python -m eval.harness       # Mean Recall@10 over the 10 traces (needs GROQ_API_KEY)
python -m eval.probes        # behavior probes (refuse, no-rec-turn1, edits, grounding)
```
The harness paces requests to stay under the Groq free-tier RPM limit
(`EVAL_PACING` seconds between turns, default 1.5).

## How it works

```
POST /chat → validate → UNDERSTAND (Groq JSON) → route
   ├─ refuse    → reply, recommendations=[]
   ├─ clarify   → one question, recommendations=[]
   └─ recommend → RETRIEVE (BM25 [+ optional embeddings]) → COMPOSE (Groq JSON)
                  → map ids→catalog → honor edits → recommendations (1–10)
```

- **Two LLM calls max per turn** (clarify/refuse use one). Well inside the 30 s
  evaluator timeout.
- **ID-only grounding:** the compose step selects catalog *ids*; the server fills
  in `name`/`url`/`test_type` from the catalog, so a URL can never be hallucinated.
- **Deterministic guards** (in `app/agent.py`): honor explicit add/drop edits,
  cap clarifying turns so the 8-turn budget is never exhausted, drop fabricated
  ids, and fall back to BM25 (never 500) if the LLM is unavailable.

Module map:

| File | Responsibility |
|------|----------------|
| `app/main.py` | FastAPI app: `/health`, `/chat` |
| `app/schemas.py` | Pydantic request/response (enforces the contract) |
| `app/agent.py` | Orchestration + deterministic guards |
| `app/retrieval.py` | Hybrid BM25 (+ optional embeddings) retrieval |
| `app/llm.py` | Groq client: JSON mode, retries, 429 backoff |
| `app/prompts.py` | Understand + compose prompts |
| `app/catalog.py` | In-memory catalog, id → grounded record |
| `scripts/build_catalog.py` | Normalize raw catalog → `data/catalog.json` |
| `eval/` | Replay harness, behavior probes, Recall@K |

## Optional: semantic retrieval

```bash
python -m pip install -r requirements-embeddings.txt
export USE_EMBEDDINGS=true
```
Adds `sentence-transformers` (all-MiniLM-L6-v2) fused with BM25 via Reciprocal
Rank Fusion. Heavier (pulls torch); retrieval degrades to BM25-only if it can't
load.

## Deploy (free tier)

**Render** (uses `render.yaml`): create a Web Service from the repo, set
`GROQ_API_KEY` as a secret env var. Health check path `/health`. The build runs
`python -m scripts.build_catalog`.

**Docker** (Railway/Fly/any): `docker build -t shl . && docker run -p 8000:8000 -e GROQ_API_KEY=gsk_... shl`.

First `/health` on a cold free instance may take up to ~2 minutes to wake.
