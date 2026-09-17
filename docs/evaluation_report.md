# Evaluation report

Same **100 query IDs** for every configuration (seed 42). Gold `relevant_passage_ids` used only for scoring. Judge prompt, model, and temperature fixed across runs.

| Step 4 layer | Required | Status |
|---|---|---|
| Traditional retrieval metrics | Recall@5, Recall@10, MRR@10, nDCG@10, latency, cost | Done — 100 queries, table below |
| LLM-judge | Correctness, groundedness, context relevance; fixed settings | Done — equivalent judge (same three scores as RAGAS/DeepEval), 20-query subset of the same IDs |
| Manual review | Disagreements between judge and retrieval; citation validity in code | Done — cases below; validator in `backend/app/generation/answer.py` |

Embeddings `BAAI/bge-base-en-v1.5`. Vector store Qdrant. Judge: `gemma4:latest` via Ollama, temperature `0.0`.

## Traditional retrieval metrics

Measured with `relevant_passage_ids`. Latency is retrieval wall time. Cost is LLM token cost (`$0` on Ollama).

## Comparison table

| config | recall@5 | recall@10 | mrr@10 | ndcg@10 | latency_ms | p95 | cost_usd/q |
|---|---|---|---|---|---|---|---|
| lexical | 0.317 | 0.391 | 0.661 | 0.481 | 56.9 | 93.3 | 0.0000 |
| dense | 0.397 | 0.469 | 0.792 | 0.588 | 32.6 | 207.9 | 0.0000 |
| hybrid_rrf | 0.395 | 0.478 | 0.764 | 0.583 | 81.4 | 125.4 | 0.0000 |
| hybrid_weighted | 0.400 | 0.495 | 0.794 | 0.607 | 82.3 | 124.4 | 0.0000 |
| hybrid_expansion | 0.425 | 0.518 | 0.826 | 0.632 | 26930.4 | 29992.7 | 0.0000 |
| best | 0.425 | 0.518 | 0.826 | 0.632 | 26930.4 | 29992.7 | 0.0000 |

`best` is the strongest variant: weighted hybrid + query expansion.

## LLM-judge evaluation

Equivalent framework (`backend/app/evaluation/judge.py`): same contract as RAGAS/DeepEval — **correctness**, **groundedness**, **context relevance** on 0–1. Prompt, model, and temperature are identical for every config.

| config | correctness | groundedness | context_rel | cites valid | insufficient | n_judged |
|---|---|---|---|---|---|---|
| lexical | 0.800 | 0.660 | 0.735 | 1.000 | 0.300 | 20 |
| hybrid_weighted | 0.750 | 0.797 | 0.795 | 0.900 | 0.250 | 20 |
| hybrid_expansion | 0.575 | 0.510 | 0.730 | 0.950 | 0.250 | 20 |
| best | 0.575 | 0.510 | 0.730 | 0.950 | 0.250 | 20 |

## Winner

**`hybrid_expansion`** — highest nDCG@10 (0.632) and Recall@10 (0.518) on the comparison table.

**Why.** Hybrid beats lexical or dense alone; expansion (original query kept) lifts recall further.

**Trade-off.** ~27 s/query vs ~80 ms without expansion. Cost $0 locally; hosted models accrue `cost_usd`. UI defaults expansion off.

## Manual review

Inspected cases where the judge and retrieval metrics disagree. Citation validity is checked in code, not by the LLM.

## Success cases

- **Q2218** (`hybrid_weighted`, recall@10=1.0, correctness=1.0): How many times is CLAST faster than BLAST? — Top hit 25495907 is gold; answer cites ≈80.8× with [25495907].
- **Q85** (`hybrid_weighted`, recall@10=0.667, correctness=1.0): Which transcription factor is considered as a master regulator of lysosomal genes? — TFEB grounded with valid passage IDs.
- **Q318** (`hybrid_weighted`, recall@10=0.5, correctness=1.0): Describe the mechanism of action of drisapersen — Antisense-oligo context; answer describes exon skipping.

## Failure / disagreement cases

- **Q318** (`hybrid_expansion`, recall@10=0.5, correctness=0.0): Same question — expansion polluted context; the model answered about antipsychotics.
- **Q2731** (`hybrid_expansion`, recall@10=0.588, correctness=0.0): Solid recall, but the answer returned `INSUFFICIENT_EVIDENCE`.
- **Q3383** (`hybrid_expansion`, recall@10=0.5, correctness=0.0): Hallucinated citation IDs; validator forced insufficient evidence.
- **Q4073** (`hybrid_weighted`, recall@10=0.182, correctness=1.0): Low recall but a correct answer from partial context.
