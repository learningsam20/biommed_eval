"""Traditional retrieval metrics: Recall@k, MRR@k, nDCG@k.

Gold `relevant_passage_ids` are used only here (never at query time). Gains are
binary: a retrieved ID is relevant or not. See docs/code-understanding.md.
"""
import math
from typing import Dict, List


def recall_at_k(retrieved: List[int], relevant: List[int], k: int) -> float:
    """Fraction of gold IDs that appear in the top-k retrieved list.

    ``|retrieved[:k] ∩ relevant| / |relevant|``. 0 if the gold set is empty.
    """
    if not relevant:
        return 0.0
    return len(set(retrieved[:k]) & set(relevant)) / len(set(relevant))


def mrr_at_k(retrieved: List[int], relevant: List[int], k: int) -> float:
    """Mean Reciprocal Rank: 1 / rank of the first gold hit in top-k, else 0."""
    rel = set(relevant)
    for rank, pid in enumerate(retrieved[:k], start=1):
        if pid in rel:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(retrieved: List[int], relevant: List[int], k: int) -> float:
    """nDCG@k with binary relevance (gain=1 for a gold ID, else 0).

    DCG = Σ 1/log2(rank+1) over gold hits in the top-k.
    IDCG = the same sum for an ideal ranking of min(|relevant|, k) hits.
    """
    rel = set(relevant)
    dcg = sum(1.0 / math.log2(rank + 1) for rank, pid in enumerate(retrieved[:k], start=1) if pid in rel)
    ideal = sum(1.0 / math.log2(rank + 1) for rank in range(1, min(len(rel), k) + 1))
    return dcg / ideal if ideal else 0.0


def score_query(retrieved: List[int], relevant: List[int], latency_ms: float) -> Dict[str, float]:
    """Per-query retrieval scores plus the retrieve-path wall time in ms."""
    return {
        "recall@5": recall_at_k(retrieved, relevant, 5),
        "recall@10": recall_at_k(retrieved, relevant, 10),
        "mrr@10": mrr_at_k(retrieved, relevant, 10),
        "ndcg@10": ndcg_at_k(retrieved, relevant, 10),
        "latency_ms": latency_ms,
    }


def aggregate(scores: List[Dict[str, float]]) -> Dict[str, float]:
    """Macro-average of score_query rows, plus p50/p95 latency and n_queries."""
    keys = ["recall@5", "recall@10", "mrr@10", "ndcg@10", "latency_ms"]
    n = max(len(scores), 1)
    out = {k: sum(s.get(k, 0.0) for s in scores) / n for k in keys}
    lat = sorted(s.get("latency_ms", 0.0) for s in scores)
    out["latency_p50"] = lat[len(lat) // 2] if lat else 0.0
    out["latency_p95"] = lat[int(len(lat) * 0.95)] if lat else 0.0
    out["n_queries"] = len(scores)
    return out
