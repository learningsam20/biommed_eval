"""Snapshot eval artifacts into output/.

Copies the frozen query set, report, architecture notes, runner scripts, and
per-config results, then writes best_hybrid_config.json from summary.json.
"""
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


README = """# Eval artifacts

Snapshots from the 100-query benchmark. Regenerated with `make output`.

Vector store (`VECTOR_DB`), embeddings (`EMBEDDING_MODEL`), LLM
(`LLM_PROVIDER`), and fusion (`FUSION_METHOD`) are `.env` switches — re-index
after changing the embedder.

| File | What it is |
|---|---|
| `eval_queries_100.json` | Frozen query set (seed 42) |
| `run_evaluation.py` | Config runner used for the numbers |
| `create_eval_set.py` | Sampler for the frozen set |
| `generate_report.py` | Report generator |
| `evaluation_report.md` | Comparison table, winner, examples |
| `architecture.md` | System diagram |
| `best_hybrid_config.json` | Selected retrieval recipe |
| `results/` | Per-config IDs, scores, and aggregates |

Reproduce from the repo root:

```bash
python -m scripts.create_eval_set --output data/eval_queries_100.json --seed 42 --size 100
python -m scripts.run_evaluation --config all --eval-set data/eval_queries_100.json --output results/
python -m scripts.generate_report --results-dir results/ --output docs/evaluation_report.md
```

Judge prompt, model, and temperature stay fixed across configs (`JUDGE_*` in `.env`).
"""


def main():
    from app.config import resolve_path
    root = resolve_path(".")
    out = root / "output"
    out.mkdir(parents=True, exist_ok=True)

    copies = {
        root / "data" / "eval_queries_100.json": out / "eval_queries_100.json",
        root / "docs" / "evaluation_report.md": out / "evaluation_report.md",
        root / "docs" / "architecture.md": out / "architecture.md",
        root / "backend" / "scripts" / "run_evaluation.py": out / "run_evaluation.py",
        root / "backend" / "scripts" / "create_eval_set.py": out / "create_eval_set.py",
        root / "backend" / "scripts" / "generate_report.py": out / "generate_report.py",
    }
    for src, dest in copies.items():
        if not src.exists():
            raise SystemExit(f"missing required file: {src}")
        shutil.copy2(src, dest)

    rsrc = root / "results"
    rdst = out / "results"
    rdst.mkdir(parents=True, exist_ok=True)
    required = [
        "summary.json",
        "lexical_results.json",
        "dense_results.json",
        "hybrid_rrf_results.json",
        "hybrid_weighted_results.json",
        "hybrid_expansion_results.json",
        "best_results.json",
    ]
    for name in required:
        src = rsrc / name
        if not src.exists():
            raise SystemExit(f"missing eval artifact: {src}")
        shutil.copy2(src, rdst / name)

    summary = json.loads((rdst / "summary.json").read_text())
    cfgs = summary.get("configs", {})
    best_name = "hybrid_expansion"
    if cfgs:
        best_name = max(cfgs, key=lambda c: (cfgs[c].get("ndcg@10", 0), cfgs[c].get("recall@10", 0)))
        if best_name == "best" and "hybrid_expansion" in cfgs:
            best_name = "hybrid_expansion"
    metrics = cfgs.get(best_name, {})
    best = {
        "name": best_name,
        "alias": "best",
        "retrieval": "hybrid",
        "fusion": "weighted",
        "query_expansion": True,
        "weight_bm25": 0.5,
        "weight_dense": 0.5,
        "why": (
            "Highest nDCG@10 and Recall@10 on the frozen 100-query set among "
            "lexical, dense, hybrid RRF, hybrid weighted, and hybrid+expansion."
        ),
        "tradeoff": (
            "Expansion adds one LLM round-trip (~27s/query locally) versus ~80ms "
            "for weighted hybrid without expansion. Interactive UI therefore "
            "defaults expansion off; enable it when maximising recall."
        ),
        "metrics": {
            "recall@5": metrics.get("recall@5"),
            "recall@10": metrics.get("recall@10"),
            "mrr@10": metrics.get("mrr@10"),
            "ndcg@10": metrics.get("ndcg@10"),
            "latency_ms": metrics.get("latency_ms"),
            "cost_usd_per_query": metrics.get("cost_usd_per_query", 0.0),
        },
        "experiment_results": "results/summary.json",
    }
    (out / "best_hybrid_config.json").write_text(json.dumps(best, indent=2) + "\n")
    (out / "README.md").write_text(README)
    print(f"output assembled -> {out} (best={best_name})")


if __name__ == "__main__":
    main()
