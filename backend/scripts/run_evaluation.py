"""Run all retrieval configs over frozen eval set.
Usage: python -m scripts.run_evaluation --config all --eval-set data/eval_queries_100.json --output results/
Configs: lexical | dense | hybrid_rrf | hybrid_weighted | hybrid_expansion | best | all
"""
import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

CONFIGS = ["lexical", "dense", "hybrid_rrf", "hybrid_weighted", "hybrid_expansion", "best"]
# Strongest variant: weighted fusion + query expansion (matches hybrid_expansion)
EXPANSION_CONFIGS = {"hybrid_expansion", "best"}
WEIGHTED_CONFIGS = {"hybrid_weighted", "hybrid_expansion", "best"}


def _dedupe(pairs):
    from app.retrieval.hybrid import dedupe_by_best_score
    return dedupe_by_best_score(pairs)


def make_retrieve_fn(config: str, ctx: dict):
    from app.retrieval.hybrid import reciprocal_rank_fusion, weighted_fusion
    from app.retrieval.expansion import expand_query
    s, bm25, dense, llm = ctx["settings"], ctx["bm25"], ctx["dense"], ctx["llm"]

    def retrieve(question: str):
        queries = [question]
        if config in EXPANSION_CONFIGS and s.QUERY_EXPANSION_ENABLED and llm is not None:
            queries = expand_query(question, llm, s.EXPANSION_COUNT, s.LLM_TEMPERATURE_EXPANSION)

        lex, dens = [], []
        need_lex = config in ("lexical", "hybrid_rrf", "hybrid_weighted", "hybrid_expansion", "best")
        need_dense = config in ("dense", "hybrid_rrf", "hybrid_weighted", "hybrid_expansion", "best")

        def run_lex():
            if not (need_lex and bm25):
                return []
            out = []
            qset = queries if config in EXPANSION_CONFIGS else [question]
            with ThreadPoolExecutor(max_workers=min(4, max(1, len(qset)))) as pool:
                for hits in pool.map(lambda q: bm25.query(q, s.LEXICAL_TOP_K), qset):
                    out.extend(hits)
            return _dedupe(out)

        def run_dense():
            if not (need_dense and dense):
                return []
            qset = queries if config in EXPANSION_CONFIGS else [question]
            lists = dense.query_many(qset, s.DENSE_TOP_K)
            merged = []
            for hits in lists:
                merged.extend(hits)
            return _dedupe(merged)

        with ThreadPoolExecutor(max_workers=2) as pool:
            fut_l = pool.submit(run_lex)
            fut_d = pool.submit(run_dense)
            lex, dens = fut_l.result(), fut_d.result()

        if config == "lexical":
            ranked = sorted(lex, key=lambda x: x[1], reverse=True)[:s.HYBRID_TOP_K]
        elif config == "dense":
            ranked = sorted(dens, key=lambda x: x[1], reverse=True)[:s.HYBRID_TOP_K]
        elif config in WEIGHTED_CONFIGS:
            ranked = weighted_fusion(lex, dens, s.WEIGHT_BM25, s.WEIGHT_DENSE, s.HYBRID_TOP_K)
        else:
            ranked = reciprocal_rank_fusion([lex, dens], s.RRF_K, s.HYBRID_TOP_K)
        # Return (id, score) pairs so callers can persist both
        return ranked
    return retrieve


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="all")
    ap.add_argument("--eval-set", default="data/eval_queries_100.json")
    ap.add_argument("--output", default="results/")
    ap.add_argument("--judge", action="store_true", help="also run LLM-judge (2 LLM calls/query)")
    ap.add_argument("--limit", type=int, default=0, help="cap queries (0=all)")
    ap.add_argument("--judge-configs", default="", help="comma list to judge (default: all evaluated)")
    args = ap.parse_args()

    from app.config import get_settings
    from app.evaluation.metrics import score_query, aggregate
    s = get_settings()
    from app.config import resolve_path
    _ep = resolve_path(args.eval_set)
    queries = json.load(open(_ep))
    print(f"loaded {len(queries)} eval queries")

    from app.retrieval.bm25 import BM25LexicalIndex
    _bp = resolve_path(s.BM25_INDEX_PATH)
    bm25 = BM25LexicalIndex.load(str(_bp)) if _bp.exists() else None
    dense, llm = None, None
    need_dense = args.config in ("all", "dense", "hybrid_rrf", "hybrid_weighted", "hybrid_expansion", "best")
    need_llm = args.config in ("all", "hybrid_expansion", "best") or args.judge
    if need_dense:
        try:
            from app.retrieval.dense import DenseRetriever
            from app.models.vector_store import get_vector_store
            dense = DenseRetriever(s.EMBEDDING_MODEL, get_vector_store(s), s.EMBEDDING_BATCH_SIZE)
        except Exception as e:
            print(f"dense unavailable ({e}) — dense/hybrid configs will score 0")
    if need_llm:
        try:
            from app.generation.llm import get_llm_client
            llm = get_llm_client(s)
        except Exception as e:
            print(f"LLM unavailable ({e}) — expansion/judge disabled")
    ctx = {"settings": s, "bm25": bm25, "dense": dense, "llm": llm}

    wanted = CONFIGS if args.config == "all" else [args.config]
    outdir = resolve_path(args.output)
    outdir.mkdir(parents=True, exist_ok=True)
    _sum_path = resolve_path(args.output) / "summary.json"
    if _sum_path.exists():
        try:
            summary = json.load(open(_sum_path))
            summary.setdefault("configs", {})
        except Exception:
            summary = {"configs": {}}
    else:
        summary = {"configs": {}}
    summary.update({"judge_model": getattr(s, "JUDGE_MODEL", "same_as_llm"),
                    "judge_temperature": s.JUDGE_TEMPERATURE,
                    "embedding_model": s.EMBEDDING_MODEL, "vector_db": s.VECTOR_DB,
                    "fusion_default": s.FUSION_METHOD})
    store = ctx["dense"].store if ctx["dense"] is not None else None
    # Default: judge every evaluated config (not a subset)
    judge_cfg = (args.judge_configs.split(",") if args.judge_configs else wanted)
    judge_cfg = [c.strip() for c in judge_cfg if c.strip()]
    do_judge = args.judge and llm is not None
    if args.judge and llm is None:
        print("judge requested but LLM unavailable — skipping judge")
    if do_judge:
        from app.evaluation.judge import JUDGE_VERSION
        summary["judge_version"] = JUDGE_VERSION

    def judge_query(q, retrieved_ids):
        from app.evaluation.judge import run_judge
        from app.generation.answer import generate_grounded_answer, validate_citations
        texts = store.fetch_texts([str(i) for i in retrieved_ids[:s.HYBRID_TOP_K]]) if store else {}
        if not texts and store is None:
            import json as _j
            _pp = resolve_path("data/passages.jsonl")
            if _pp.exists():
                want = set(retrieved_ids[:s.HYBRID_TOP_K])
                with open(_pp) as _f:
                    for _line in _f:
                        _r = _j.loads(_line)
                        if _r["id"] in want:
                            texts[str(_r["id"])] = _r["passage"]
                        if len(texts) >= len(want):
                            break
        ev = [(pid, texts.get(str(pid), "")) for pid in retrieved_ids[:s.HYBRID_TOP_K]
              if texts.get(str(pid))]
        ans, cites, insuf = generate_grounded_answer(
            q["question"], ev, llm, s.LLM_TEMPERATURE_ANSWER)
        js = run_judge(q["question"], q["answer"], ans, ev, llm, s.JUDGE_TEMPERATURE)
        _, hall = validate_citations(ans, retrieved_ids)
        return {
            "answer": ans[:2000],
            "citations": cites,
            "insufficient_evidence": insuf,
            "judge_correctness": js.correctness,
            "judge_groundedness": js.groundedness,
            "judge_context_relevance": js.context_relevance,
            "citation_hallucinated": hall,
            "citation_valid": bool(insuf or cites),
        }

    eval_queries = queries[:args.limit] if args.limit else queries
    if args.limit and resolve_path(args.output) == resolve_path("results"):
        raise SystemExit("refusing to overwrite full results/ with --limit subset; pass --output results_judge<N>")

    for cfg in wanted:
        if llm is not None and hasattr(llm, "usage"):
            llm.usage.reset()
        retrieve = make_retrieve_fn(cfg, ctx)
        rows, scores = [], []
        jscores = []
        for qi, q in enumerate(eval_queries):
            t0 = time.perf_counter()
            ranked = retrieve(q["question"])  # List[(id, score)]
            lat = (time.perf_counter() - t0) * 1000
            ret_ids = [pid for pid, _ in ranked]
            ret_scores = [{"passage_id": pid, "score": float(sc)} for pid, sc in ranked]
            sc = score_query(ret_ids, q["relevant_passage_ids"], lat)
            row = {"query_id": q["query_id"], "retrieved_ids": ret_ids,
                   "retrieved_scores": ret_scores, **sc}
            if do_judge and cfg in judge_cfg:
                try:
                    j = judge_query(q, ret_ids)
                    row.update(j)
                    jscores.append(j)
                except Exception as e:
                    print(f"  judge failed q={q['query_id']}: {type(e).__name__}")
                    row["judge_error"] = True
            rows.append(row)
            scores.append(sc)
            if (qi + 1) % 10 == 0:
                print(f"  {cfg}: {qi + 1}/{len(eval_queries)}")
        agg = aggregate(scores)
        if llm is not None and hasattr(llm, "usage"):
            snap = llm.usage.snapshot()
            agg["cost_usd"] = snap["cost_usd"]
            agg["total_tokens"] = snap["total_tokens"]
            agg["llm_calls"] = snap["llm_calls"]
            # Per-query averages for the report
            nq = max(len(eval_queries), 1)
            agg["cost_usd_per_query"] = round(snap["cost_usd"] / nq, 6)
            agg["tokens_per_query"] = round(snap["total_tokens"] / nq, 1)
        else:
            agg["cost_usd"] = 0.0
            agg["cost_usd_per_query"] = 0.0
            agg["total_tokens"] = 0
            agg["llm_calls"] = 0
            agg["tokens_per_query"] = 0.0
        if jscores:
            n = len(jscores)
            agg["judge_correctness"] = sum(j["judge_correctness"] for j in jscores) / n
            agg["judge_groundedness"] = sum(j["judge_groundedness"] for j in jscores) / n
            agg["judge_context_relevance"] = sum(j["judge_context_relevance"] for j in jscores) / n
            agg["citation_valid_rate"] = sum(1 for j in jscores if j["citation_valid"]) / n
            agg["insufficient_rate"] = sum(1 for j in jscores if j["insufficient_evidence"]) / n
            agg["n_judged"] = n
        json.dump(rows, open(outdir / f"{cfg}_results.json", "w"), indent=2)
        summary["configs"][cfg] = agg
        print(f"{cfg}: " + ", ".join(f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}"
                                     for k, v in agg.items() if k != "n_queries"))
    json.dump(summary, open(outdir / "summary.json", "w"), indent=2)
    print(f"summary -> {outdir / 'summary.json'}")


if __name__ == "__main__":
    main()
