"""Eval orchestration: run one config over the frozen 100-query set."""
import time
from typing import Callable, Dict, List


def run_config(queries: List[dict], retrieve_fn: Callable,
               judge_fn=None) -> List[Dict]:
    """Score every frozen query with `retrieve_fn`; optionally attach `judge_fn` output."""
    from .metrics import score_query
    rows = []
    for q in queries:
        t0 = time.perf_counter()
        ranked = retrieve_fn(q["question"])
        latency = (time.perf_counter() - t0) * 1000
        if ranked and isinstance(ranked[0], (list, tuple)) and len(ranked[0]) == 2:
            retrieved = [pid for pid, _ in ranked]
            scores = [{"passage_id": pid, "score": float(sc)} for pid, sc in ranked]
        else:
            retrieved = list(ranked)
            scores = [{"passage_id": pid, "score": None} for pid in retrieved]
        row = {"query_id": q["query_id"], "retrieved_ids": retrieved,
               "retrieved_scores": scores,
               **score_query(retrieved, q["relevant_passage_ids"], latency)}
        if judge_fn is not None:
            row["judge"] = judge_fn(q, retrieved)
        rows.append(row)
    return rows
