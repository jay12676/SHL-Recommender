"""Replay harness: run each trace's user turns against the agent and score
Mean Recall@10 on the final shortlist.

This replays the labeled user turns deterministically (a faithful, cheap proxy
for the official LLM-simulated user) so iteration is fast and reproducible. The
agent only ever sees prior assistant *reply text* in the history, exactly as the
stateless API contract specifies.

Run:  python -m eval.harness
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import List

from app.agent import run_chat
from app.catalog import get_catalog
from app.schemas import Message
from eval.recall import recall_at_k, resolve_expected

ROOT = Path(__file__).resolve().parents[1]
TRACES = ROOT / "eval" / "traces.json"


def _url_to_id():
    return {a.url: a.id for a in get_catalog().items}


def run_trace(trace: dict) -> dict:
    url2id = _url_to_id()
    history: List[Message] = []
    final_rec_ids: List[str] = []
    turns = 0

    for user_turn in trace["user_turns"]:
        history.append(Message(role="user", content=user_turn))
        resp = run_chat(history)
        turns += 1
        history.append(Message(role="assistant", content=resp.reply))
        # Light pacing keeps the rapid replay under the free-tier RPM limit.
        time.sleep(float(os.getenv("EVAL_PACING", "2.0")))
        rec_ids = [url2id.get(r.url) for r in resp.recommendations]
        rec_ids = [r for r in rec_ids if r]
        if rec_ids:
            final_rec_ids = rec_ids  # keep the latest non-empty shortlist

    expected_ids = resolve_expected(trace["expected"])
    recall = recall_at_k(final_rec_ids, expected_ids, k=10)
    return {
        "id": trace["id"],
        "recall@10": round(recall, 3),
        "n_expected": len(expected_ids),
        "n_recommended": len(final_rec_ids),
        "turns": turns,
        "hits": [e for e in expected_ids if e in set(final_rec_ids)],
        "misses": [e for e in expected_ids if e not in set(final_rec_ids)],
    }


def main() -> int:
    traces = json.loads(TRACES.read_text(encoding="utf-8"))
    catalog = get_catalog()
    id2name = {a.id: a.name for a in catalog.items}
    results = []
    for t in traces:
        r = run_trace(t)
        results.append(r)
        miss_names = [id2name.get(m, m) for m in r["misses"]]
        print(
            f"{r['id']:>4}  recall@10={r['recall@10']:.2f}  "
            f"({len(r['hits'])}/{r['n_expected']})  recs={r['n_recommended']}  "
            f"turns={r['turns']}"
        )
        if miss_names:
            print(f"        misses: {miss_names}")
    mean = sum(r["recall@10"] for r in results) / len(results)
    print("-" * 60)
    print(f"Mean Recall@10 over {len(results)} traces: {mean:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
