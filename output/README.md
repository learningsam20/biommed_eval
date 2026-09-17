# Eval artifacts

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
