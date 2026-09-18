"""Hybrid fusion: Reciprocal Rank Fusion + weighted score fusion."""
from typing import Dict, List, Tuple


def dedupe_by_best_score(pairs: List[Tuple[int, float]]) -> List[Tuple[int, float]]:
    """Keep highest score per passage id (needed when merging multi-query hits)."""
    best: Dict[int, float] = {}
    for pid, score in pairs:
        if pid not in best or score > best[pid]:
            best[pid] = score
    return list(best.items())


def reciprocal_rank_fusion(rank_lists: List[List[Tuple[int, float]]], k: int = 60,
                           top_k: int = 20) -> List[Tuple[int, float]]:
    """Combine ranked lists by rank, not raw score: score += 1 / (k + rank).

    Each list votes once per passage ID. Default k=60 (Cormack et al.).
    """
    scores: Dict[int, float] = {}
    for rlist in rank_lists:
        # Deduplicate within each list so one signal cannot vote twice for the same id
        seen = set()
        rank = 0
        for pid, _ in rlist:
            if pid in seen:
                continue
            seen.add(pid)
            scores[pid] = scores.get(pid, 0.0) + 1.0 / (k + rank + 1)
            rank += 1
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]


def _minmax_norm(pairs: List[Tuple[int, float]]) -> Dict[int, float]:
    """Scale scores in a list to [0, 1] so BM25 and cosine can be added."""
    if not pairs:
        return {}
    vals = [s for _, s in pairs]
    lo, hi = min(vals), max(vals)
    if hi == lo:
        return {pid: 1.0 for pid, _ in pairs}
    return {pid: (s - lo) / (hi - lo) for pid, s in pairs}


def weighted_fusion(lexical: List[Tuple[int, float]], dense: List[Tuple[int, float]],
                    w_bm25: float = 0.5, w_dense: float = 0.5,
                    top_k: int = 20) -> List[Tuple[int, float]]:
    """Min-max-normalize each signal, then ``w_bm25 * lex + w_dense * dense``."""
    ln, dn = _minmax_norm(lexical), _minmax_norm(dense)
    fused: Dict[int, float] = {}
    for pid, v in ln.items():
        fused[pid] = fused.get(pid, 0.0) + w_bm25 * v
    for pid, v in dn.items():
        fused[pid] = fused.get(pid, 0.0) + w_dense * v
    return sorted(fused.items(), key=lambda x: x[1], reverse=True)[:top_k]
