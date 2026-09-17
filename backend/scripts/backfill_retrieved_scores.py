"""Attach retrieved_scores to existing eval JSON without changing IDs or metrics.

PDF Step 1: record passage IDs and scores for every search. Older result files
had IDs only. This fills scores from the live BM25 (+ dense) indexes.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

CONFIGS = ["lexical", "dense", "hybrid_rrf", "hybrid_weighted", "hybrid_expansion", "best"]
WEIGHTED = {"hybrid_weighted", "hybrid_expansion", "best"}


def bm25_score_map(bm25, query: str) -> dict:
    from app.retrieval.bm25 import tokenize
    raw = bm25._bm25.get_scores(tokenize(query))
    return {bm25._ids[i]: float(raw[i]) for i in range(len(bm25._ids))}


def main():
    from app.config import get_settings, resolve_path
    from app.retrieval.bm25 import BM25LexicalIndex
    from app.retrieval.hybrid import reciprocal_rank_fusion, weighted_fusion

    s = get_settings()
    queries = {q["query_id"]: q for q in json.loads(resolve_path(s.EVAL_QUERY_SET).read_text())}
    bm25 = BM25LexicalIndex.load(str(resolve_path(s.BM25_INDEX_PATH)))
    dense = None
    try:
        from app.retrieval.dense import DenseRetriever
        from app.models.vector_store import get_vector_store
        dense = DenseRetriever(s.EMBEDDING_MODEL, get_vector_store(s), s.EMBEDDING_BATCH_SIZE)
        print("dense retriever ready")
    except Exception as e:
        print(f"dense unavailable ({e}) — hybrid/dense scores fall back to BM25")

    rdir = resolve_path("results")
    for cfg in CONFIGS:
        path = rdir / f"{cfg}_results.json"
        if not path.exists():
            print(f"skip missing {path}")
            continue
        rows = json.loads(path.read_text())
        for i, row in enumerate(rows):
            q = queries.get(row["query_id"])
            if not q:
                continue
            ids = list(row.get("retrieved_ids") or [])
            bmap = bm25_score_map(bm25, q["question"])
            lex = [(pid, bmap.get(pid, 0.0)) for pid in ids]
            dmap = {}
            if dense is not None and cfg != "lexical":
                dmap = dict(dense.query(q["question"], top_k=max(200, len(ids) or 1)))
            den = [(pid, float(dmap.get(pid, 0.0))) for pid in ids]
            if cfg == "lexical" or dense is None:
                scored = lex
            elif cfg == "dense":
                scored = den
            elif cfg in WEIGHTED:
                fused = dict(weighted_fusion(lex, den, s.WEIGHT_BM25, s.WEIGHT_DENSE, top_k=len(ids) or 1))
                scored = [(pid, float(fused.get(pid, 0.0))) for pid in ids]
            else:
                fused = dict(reciprocal_rank_fusion([lex, den], s.RRF_K, top_k=len(ids) or 1))
                scored = [(pid, float(fused.get(pid, 0.0))) for pid in ids]
            row["retrieved_scores"] = [{"passage_id": pid, "score": float(sc)} for pid, sc in scored]
            if (i + 1) % 25 == 0:
                print(f"  {cfg}: {i + 1}/{len(rows)}")
        path.write_text(json.dumps(rows, indent=2))
        print(f"wrote scores -> {path} ({len(rows)} queries)")


if __name__ == "__main__":
    main()
