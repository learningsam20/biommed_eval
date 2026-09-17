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

Hybrid + expansion wins nDCG@10 (0.632) and Recall@10 (0.518). Weighted hybrid is better on judge correctness and ~80 ms. UI default: weighted, expansion off.

Details: [docs/evaluation_report.md](docs/evaluation_report.md).
