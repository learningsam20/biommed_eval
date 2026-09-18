# Design decisions

Providers are `.env` switches. Implementation: [specs.md](specs.md). Diagram: [docs/architecture.md](docs/architecture.md).

## Dataset

[`rag-datasets/rag-mini-bioasq`](https://huggingface.co/datasets/rag-datasets/rag-mini-bioasq) — 40,221 passages, 4,719 QA pairs. Gold IDs are scoring-only. Eval set: 100 queries, seed 42.

## Why these defaults

| Area | Switch | Default | Why |
|---|---|---|---|
| Vector store | `VECTOR_DB` | Pinecone serverless | Managed 40k-vector index; Qdrant when you want fully local. |
| Embeddings | `EMBEDDING_MODEL` / `EMBEDDING_DIM` | bge-base-en-v1.5 (768) | Strong English retrieval without a large-model encode cost. Re-index after a swap. |
| LLM | `LLM_PROVIDER` | OpenRouter (or Ollama) | One client for expansion, answers, and judge. Ollama for $0 local runs. |
| Fusion | `FUSION_METHOD` | Weighted | Higher nDCG than RRF here; RRF still implemented. |
| Expansion | UI + `QUERY_EXPANSION_ENABLED` | Off in the UI | Best recall when on; too slow for interactive search. |

## Result

Hybrid + expansion wins nDCG@10 (0.632) and Recall@10 (0.518) on 100 queries — that is the selected recipe. The 20-query LLM-judge prefers weighted hybrid for correctness; it is a trade-off, not the selector. UI default: weighted, expansion off.

Details: [docs/evaluation_report.md](docs/evaluation_report.md). Recipe: [docs/best_hybrid_config.json](docs/best_hybrid_config.json). Runner: [backend/scripts/run_evaluation.py](backend/scripts/run_evaluation.py).

## 10. Judge-framework + model speed fixes (executed)

- Model: `gemma4:latest` (9.6 GB) was the slowness bottleneck. Switched to
  `granite4.1:3b` (3.4B, Q4, 128k ctx) — passes RAGAS structured-output prompts
  (llama3.2:latest echoed JSON schemas; granite4.2:8b worked but was slow).
- OllamaClient now honors `num_ctx` (32768) + `num_predict` (4096) from `.env`
  (was hardcoded `num_predict=512`, truncating answers/judges).
- RAGAS: changed `instructor.Mode.MD_JSON` → `Mode.JSON` + `max_retries=3`,
  `timeout=180`, `num_ctx` passthrough.
- DeepEval: `OllamaModel` generation_kwargs now include `num_ctx` + `timeout=600`.
- Verified all three frameworks emit real scores on one query:
  custom 6s (1.0/1.0/1.0), ragas 40s (0.4/1.0/0.0), deepeval 58s (1.0/1.0/0.44).
  Per-query costs (warm): expansion 5s, answer 4s, custom 6s.
- Server logs cleared and both services restarted fresh (backend :5268, frontend :5269).
